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
) -> NDArray[np.uint8]:
    """Annotate a BGR frame in place-free copy: joints + skeleton lines, the
    state banner, the depth readout and — when a frame is unscoreable — the
    human-readable reason (e.g. which landmarks are not visible).
    Landmarks are normalized (0..1)."""
    import cv2

    out = color.copy()
    height, width = out.shape[:2]

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
            cv2.line(out, points[start], points[end], _GREEN, 2)
        for point in points.values():
            cv2.circle(out, point, 4, _GREEN, -1)

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
