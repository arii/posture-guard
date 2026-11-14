"""Frame annotation for the MJPEG stream (spec §dashboard): skeleton overlay,
depth distance and state banner drawn on the color frame.

OpenCV is imported lazily so this module imports on CI machines without the
hardware extras; the pipeline injects its own renderer in tests that must run
without OpenCV.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from posture_guard.scoring import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Landmark,
)

_JOINT_INDICES = (LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)

# BGR colors.
_GREEN = (80, 220, 80)
_AMBER = (60, 170, 255)
_GRAY = (160, 160, 160)
_RED = (80, 80, 230)
_BANNER_BG = (40, 40, 40)

# Wireframe stroke, keyed by the pipeline's hex wire_color (pipeline.py keeps
# the mapping to state; this stays a pure hex -> BGR table). Legacy empty
# string callers keep the historical green.
_WIRE_BGR: dict[str, tuple[int, int, int]] = {
    "#10b981": (129, 185, 16),  # emerald: good
    "#f59e0b": (11, 158, 245),  # amber: pre-sustain drift
    "#ef4444": (68, 68, 239),   # crimson: confirmed breach
    "#8b95a3": (163, 149, 139),  # slate: unreliable — drawn dashed
}
_DASHED_WIRE = "#8b95a3"
_DASH_LEN = 9  # px of stroke per dash at the annotated frame resolution
_GAP_LEN = 6


def _dashed_line(
    img: NDArray[np.uint8],
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Dashed variant of cv2.line (OpenCV has no dash style for cv2.line)."""
    import cv2

    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    drawn = 0.0
    stroke = True
    while drawn < length:
        seg = _DASH_LEN if stroke else _GAP_LEN
        seg_end = min(length, drawn + seg)
        if stroke:
            p1 = (int(start[0] + ux * drawn), int(start[1] + uy * drawn))
            p2 = (int(start[0] + ux * seg_end), int(start[1] + uy * seg_end))
            cv2.line(img, p1, p2, color, thickness)
        drawn = seg_end
        stroke = not stroke


def _banner_color(state: str) -> tuple[int, int, int]:
    if state in ("bad_chime", "bad_voice"):
        return _AMBER
    if state == "unreliable":
        return _RED
    if state == "good":
        return _GREEN
    return _GRAY


def draw_overlay(
    color: NDArray[np.uint8],
    landmarks: Sequence[Landmark] | None,
    state_text: str,
    depth_text: str,
    reason_text: str = "",
    wire_color: str = "",
) -> NDArray[np.uint8]:
    """Annotate a BGR frame in place-free copy: joints + skeleton lines, the
    state banner, the depth readout and — when a frame is unscoreable — the
    human-readable reason (e.g. which landmarks are not visible).
    Landmarks are normalized (0..1). wire_color (hex, from pipeline.wire_color_for)
    binds the skeleton stroke to the tracking state; empty keeps the legacy
    green for direct callers."""
    import cv2

    out = color.copy()
    height, width = out.shape[:2]
    stroke = _WIRE_BGR.get(wire_color, _GREEN)
    dashed = wire_color == _DASHED_WIRE

    if landmarks is not None and len(landmarks) > max(_JOINT_INDICES):
        points = {
            index: (int(landmarks[index].x * width), int(landmarks[index].y * height))
            for index in _JOINT_INDICES
        }
        lines = (
            (LEFT_EAR, RIGHT_EAR),
            (LEFT_EAR, LEFT_SHOULDER),
            (RIGHT_EAR, RIGHT_SHOULDER),
            (LEFT_SHOULDER, RIGHT_SHOULDER),
            (LEFT_SHOULDER, LEFT_HIP),
            (RIGHT_SHOULDER, RIGHT_HIP),
            (LEFT_HIP, RIGHT_HIP),
        )
        for start, end in lines:
            if dashed:
                _dashed_line(out, points[start], points[end], stroke, 2)
            else:
                cv2.line(out, points[start], points[end], stroke, 2)
        for point in points.values():
            cv2.circle(out, point, 4, stroke, -1)

    if state_text == "unreliable":
        # Demoted (demo feedback): in the high-mount framing this state is
        # near-permanent, and the big red banner obscured the live video.
        # The header badge pill and the reliability line below the metrics
        # carry the status and the missing-landmark reason instead.
        return out

    banner = _banner_color(state_text)
    banner_h = 76 if reason_text else 58
    cv2.rectangle(out, (0, 0), (width, banner_h), _BANNER_BG, -1)
    cv2.putText(
        out, state_text.upper(), (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, banner, 2
    )
    cv2.putText(
        out, depth_text, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1
    )
    if reason_text:
        cv2.putText(
            out, reason_text, (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner, 1
        )
    return out


def encode_jpeg(color: NDArray[np.uint8]) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", color)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()
