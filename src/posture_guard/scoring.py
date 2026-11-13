"""Posture scoring: pure functions over MediaPipe Pose landmarks.

All metric functions take landmark sequences indexed per MediaPipe Pose
convention (33 landmarks; only ears, shoulders and hips are used here) and
return signed angles in degrees. Scoring is gated: without a pose, readable
depth inside the configured band, and landmark visibility above the floor,
score_pose returns Unreliable instead of a number — the state machine never
nudges on bad data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from posture_guard.config import ScoringConfig

# MediaPipe Pose landmark indices.
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


class Landmark(Protocol):
    """Structural subset of a MediaPipe Pose landmark we consume."""

    x: float
    y: float
    visibility: float


@dataclass(frozen=True)
class SimpleLandmark:
    """Convenience Landmark for tests and synthetic poses."""

    x: float
    y: float
    visibility: float = 1.0


class UnreliableReason(Enum):
    NO_POSE = "no_pose"
    LOW_VISIBILITY = "low_visibility"
    DEPTH_UNREADABLE = "depth_unreadable"
    DEPTH_OUT_OF_BAND = "depth_out_of_band"


@dataclass(frozen=True)
class ScoredPosture:
    """Metrics for one frame. Sign conventions are symmetric: the state
    machine compares |value| against thresholds."""

    head_forward_deg: float
    trunk_lean_deg: float
    head_tilt_deg: float
    shoulder_level_deg: float
    depth_m: float


@dataclass(frozen=True)
class Unreliable:
    reason: UnreliableReason
    detail: str = ""


ScoringResult = ScoredPosture | Unreliable


def _mid(a: Landmark, b: Landmark) -> tuple[float, float]:
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


def _angle_from_vertical(dx: float, dy: float) -> float:
    """Signed angle (deg) between a body vector and the vertical image axis."""
    return math.degrees(math.atan2(dx, dy))


def measure_head_forward(landmarks: Sequence[Landmark]) -> float:
    """Head-forward: ear-mid horizontal offset from shoulder-mid, as an angle
    from vertical (spec: ear-vs-shoulder offset normalized by shoulder width —
    the ratio scaling cancels inside atan2, so the angle is width-invariant)."""
    ear_mid = _mid(landmarks[LEFT_EAR], landmarks[RIGHT_EAR])
    shoulder_mid = _mid(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    # Vector from shoulder-mid to ear-mid, measured against the downward image
    # axis (y grows down): an upright head has dy > 0 and reads 0 degrees.
    return _angle_from_vertical(ear_mid[0] - shoulder_mid[0], shoulder_mid[1] - ear_mid[1])


def measure_trunk_lean(landmarks: Sequence[Landmark]) -> float:
    """Trunk lean: shoulder-mid to hip-mid vector vs vertical."""
    shoulder_mid = _mid(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
    hip_mid = _mid(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
    return _angle_from_vertical(shoulder_mid[0] - hip_mid[0], hip_mid[1] - shoulder_mid[1])


def measure_head_tilt(landmarks: Sequence[Landmark]) -> float:
    """Head tilt: signed angle between the inter-ear vector and the
    inter-shoulder vector. Zero when the ear line is parallel to the
    shoulders; grows as the head rolls."""
    ear_vec = (
        landmarks[RIGHT_EAR].x - landmarks[LEFT_EAR].x,
        landmarks[RIGHT_EAR].y - landmarks[LEFT_EAR].y,
    )
    shoulder_vec = (
        landmarks[RIGHT_SHOULDER].x - landmarks[LEFT_SHOULDER].x,
        landmarks[RIGHT_SHOULDER].y - landmarks[LEFT_SHOULDER].y,
    )
    cross = ear_vec[0] * shoulder_vec[1] - ear_vec[1] * shoulder_vec[0]
    dot = ear_vec[0] * shoulder_vec[0] + ear_vec[1] * shoulder_vec[1]
    return math.degrees(math.atan2(cross, dot))


def measure_shoulder_level(landmarks: Sequence[Landmark]) -> float:
    """Shoulder level: signed angle of the inter-shoulder vector vs horizontal."""
    dy = landmarks[RIGHT_SHOULDER].y - landmarks[LEFT_SHOULDER].y
    dx = landmarks[RIGHT_SHOULDER].x - landmarks[LEFT_SHOULDER].x
    return math.degrees(math.atan2(dy, dx))


def valid_depth_median(samples: Sequence[float]) -> float | None:
    """Median of valid depth samples (spec: sampled over a neighborhood,
    invalid values rejected). Invalid: non-finite or <= 0. None if none valid."""
    valid = sorted(s for s in samples if math.isfinite(s) and s > 0)
    if not valid:
        return None
    n = len(valid)
    if n % 2:
        return valid[n // 2]
    return (valid[n // 2 - 1] + valid[n // 2]) / 2


def visibility_ok(landmarks: Sequence[Landmark], cfg: ScoringConfig) -> bool:
    """True when every tracked landmark clears the visibility floor."""
    tracked = (LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    return all(landmarks[i].visibility >= cfg.visibility_floor for i in tracked)


def score_pose(
    landmarks: Sequence[Landmark] | None,
    depth_m: float | None,
    cfg: ScoringConfig,
) -> ScoringResult:
    """Score one frame, or return Unreliable with the blocking reason.

    Gate order is deliberate: no pose and depth-out-of-band map to the
    no-person state; low visibility and unreadable depth map to the
    unreliable state (see state.py)."""
    if landmarks is None:
        return Unreliable(UnreliableReason.NO_POSE)
    if depth_m is None or depth_m <= 0:
        return Unreliable(UnreliableReason.DEPTH_UNREADABLE)
    if not (cfg.depth_min_m <= depth_m <= cfg.depth_max_m):
        return Unreliable(
            UnreliableReason.DEPTH_OUT_OF_BAND,
            f"depth {depth_m:.2f} m outside {cfg.depth_min_m}-{cfg.depth_max_m} m",
        )
    if not visibility_ok(landmarks, cfg):
        return Unreliable(UnreliableReason.LOW_VISIBILITY)
    return ScoredPosture(
        head_forward_deg=measure_head_forward(landmarks),
        trunk_lean_deg=measure_trunk_lean(landmarks),
        head_tilt_deg=measure_head_tilt(landmarks),
        shoulder_level_deg=measure_shoulder_level(landmarks),
        depth_m=depth_m,
    )


def zone_of(metrics: ScoredPosture, cfg: ScoringConfig) -> str:
    """Classify a score against thresholds with hysteresis.

    Returns "bad" when any metric passes its enter threshold, "good" when all
    metrics are below their exit thresholds, "neutral" in the hysteresis band
    between them."""
    enter = (
        (metrics.head_forward_deg, cfg.head_forward_deg),
        (metrics.trunk_lean_deg, cfg.trunk_lean_deg),
        (metrics.head_tilt_deg, cfg.head_tilt_deg),
        (metrics.shoulder_level_deg, cfg.shoulder_level_deg),
    )
    if any(abs(value) > threshold for value, threshold in enter):
        return "bad"
    if all(abs(value) < cfg.bad_exit(threshold) for value, threshold in enter):
        return "good"
    return "neutral"
