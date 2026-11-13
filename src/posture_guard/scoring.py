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

from posture_guard.config import BaselineDeviationConfig, ScoringConfig

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


# Tracked landmark groups for human-readable reliability detail: name and the
# (left, right) landmark pair.
_TRACKED_GROUPS: tuple[tuple[str, tuple[int, int]], ...] = (
    ("ears", (LEFT_EAR, RIGHT_EAR)),
    ("shoulders", (LEFT_SHOULDER, RIGHT_SHOULDER)),
    ("hips", (LEFT_HIP, RIGHT_HIP)),
)
_SIDES = ("left", "right")


def low_visibility_detail(landmarks: Sequence[Landmark], cfg: ScoringConfig) -> str:
    """Human-readable LOW_VISIBILITY reason: which tracked landmarks are below
    the visibility floor and by how much, e.g.
    'hips not visible (left 0.31, right 0.28) - trunk lean unavailable'."""
    parts: list[str] = []
    hips_hidden = False
    for name, (left, right) in _TRACKED_GROUPS:
        below = [
            f"{side} {landmarks[index].visibility:.2f}"
            for side, index in zip(_SIDES, (left, right), strict=True)
            if landmarks[index].visibility < cfg.visibility_floor
        ]
        if below:
            parts.append(f"{name} not visible ({', '.join(below)})")
            hips_hidden = hips_hidden or name == "hips"
    if not parts:
        return "landmark visibility below the floor"
    detail = "; ".join(parts)
    return detail + " - trunk lean unavailable" if hips_hidden else detail


@dataclass(frozen=True)
class ScoredPosture:
    """Metrics for one frame. Sign conventions are symmetric: the state
    machine compares |value| against thresholds.

    trunk_lean_available is False when the hips sit below the visibility
    floor (degraded-feedback mode): the measured trunk angle is garbage, the
    metric drops out of the BAD decision, and the dashboard reports it as
    unavailable instead of a number.
    """

    head_forward_deg: float
    trunk_lean_deg: float
    head_tilt_deg: float
    shoulder_level_deg: float
    depth_m: float
    trunk_lean_available: bool = True


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
    """Shoulder level: signed tilt of the inter-shoulder line vs horizontal,
    folded into (-90, 90).

    Folding is required because the raw atan2 over the inter-shoulder vector
    wraps to ~+/-180 when RIGHT_SHOULDER lands left of LEFT_SHOULDER in
    image coordinates (mirrored / high side-mount framing), reading level
    shoulders as a ~178 degree slouch. The shoulder line is undirected -
    only its deviation from horizontal matters - so folding into (-90, 90)
    loses nothing the scoring consumes. atan2(dy, abs(dx)) performs the
    fold while keeping the sign tied to dy alone (positive = right shoulder
    lower), so the value is identical with and without x-mirroring instead
    of flipping to the folded mirror of the true tilt."""
    dy = landmarks[RIGHT_SHOULDER].y - landmarks[LEFT_SHOULDER].y
    dx = landmarks[RIGHT_SHOULDER].x - landmarks[LEFT_SHOULDER].x
    return math.degrees(math.atan2(dy, abs(dx)))


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


def _tracked_indices(cfg: ScoringConfig) -> tuple[int, ...]:
    """Landmarks the visibility gate requires. Hips drop out when trunk lean is
    optional (scoring.trunk_lean_required: false) so the high-mount framing —
    where hips can never clear the floor — still scores ears and shoulders."""
    base = (LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)
    return base if not cfg.trunk_lean_required else base + (LEFT_HIP, RIGHT_HIP)


def visibility_ok(landmarks: Sequence[Landmark], cfg: ScoringConfig) -> bool:
    """True when every tracked landmark clears the visibility floor."""
    return all(landmarks[i].visibility >= cfg.visibility_floor for i in _tracked_indices(cfg))


def score_pose(
    landmarks: Sequence[Landmark] | None,
    depth_m: float | None,
    cfg: ScoringConfig,
) -> ScoringResult:
    """Score one frame, or return Unreliable with the blocking reason.

    Gate order is deliberate: a missing pose is the only absence signal (the
    no-person timer); every other failure — depth unreadable, depth out of
    band, low visibility — is an unscoreable frame for a person who is
    visibly present (see state.py)."""
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
        ears_shoulders_ok = all(
            landmarks[i].visibility >= cfg.visibility_floor
            for i in (LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER)
        )
        if not ears_shoulders_ok:
            # Ears and shoulders are the minimum usable set: without them no
            # metric is measurable, so the frame is unscoreable (UNRELIABLE).
            return Unreliable(
                UnreliableReason.LOW_VISIBILITY, low_visibility_detail(landmarks, cfg)
            )
        hips_ok = all(
            landmarks[i].visibility >= cfg.visibility_floor for i in (LEFT_HIP, RIGHT_HIP)
        )
        if cfg.trunk_lean_required and not hips_ok and not cfg.degraded_feedback:
            return Unreliable(
                UnreliableReason.LOW_VISIBILITY, low_visibility_detail(landmarks, cfg)
            )
        # Degraded feedback (scoring.degraded_feedback): the person is clearly
        # tracked, so score the metrics whose landmarks ARE visible (head
        # forward, head tilt, shoulder level) and drop trunk lean from the
        # decision — same pattern as trunk-lean-optional mode. Feedback keeps
        # working in the high-mount framing instead of going silent.
        return ScoredPosture(
            head_forward_deg=measure_head_forward(landmarks),
            trunk_lean_deg=measure_trunk_lean(landmarks),
            head_tilt_deg=measure_head_tilt(landmarks),
            shoulder_level_deg=measure_shoulder_level(landmarks),
            depth_m=depth_m,
            trunk_lean_available=hips_ok,
        )
    return ScoredPosture(
        head_forward_deg=measure_head_forward(landmarks),
        trunk_lean_deg=measure_trunk_lean(landmarks),
        head_tilt_deg=measure_head_tilt(landmarks),
        shoulder_level_deg=measure_shoulder_level(landmarks),
        depth_m=depth_m,
        trunk_lean_available=all(
            landmarks[i].visibility >= cfg.visibility_floor
            for i in (LEFT_HIP, RIGHT_HIP)
        ),
    )


def _metric_threshold_pairs(
    metrics: ScoredPosture, cfg: ScoringConfig
) -> list[tuple[float, float]]:
    """(value, enter threshold) pairs the BAD decision checks. Trunk lean drops
    out whenever its landmarks are not reliably visible (degraded feedback,
    or trunk lean optional) so a garbage trunk angle cannot decide the zone."""
    pairs = [
        (metrics.head_forward_deg, cfg.head_forward_deg),
        (metrics.head_tilt_deg, cfg.head_tilt_deg),
        (metrics.shoulder_level_deg, cfg.shoulder_level_deg),
    ]
    if cfg.trunk_lean_required and metrics.trunk_lean_available:
        pairs.append((metrics.trunk_lean_deg, cfg.trunk_lean_deg))
    return pairs


def zone_of(metrics: ScoredPosture, cfg: ScoringConfig) -> str:
    """Classify a score against thresholds with hysteresis.

    Returns "bad" when any metric passes its enter threshold, "good" when all
    metrics are below their exit thresholds, "neutral" in the hysteresis band
    between them."""
    pairs = _metric_threshold_pairs(metrics, cfg)
    if any(abs(value) > threshold for value, threshold in pairs):
        return "bad"
    if all(abs(value) < cfg.bad_exit(threshold) for value, threshold in pairs):
        return "good"
    return "neutral"


class PostureBaseline:
    """Exponential moving average of each scored angle over ~window_s.

    Tracks the user's own recent neutral so a whole-torso slouch — head and
    shoulders drifting back together, keeping head-vs-shoulder angles small —
    still breaches as deviation from the person's neutral. Alpha is dt/window_s
    (capped at 1), so a 30 fps stream moves the baseline ~0.1% per frame: fast
    transients leave it untouched while a sustained drift becomes the new
    reference over several minutes. Fed only scored frames (Unreliable results
    carry no metrics); resetting re-anchors to the next update.
    """

    def __init__(self, cfg: BaselineDeviationConfig) -> None:
        self._window_s = cfg.window_s
        self._metrics: ScoredPosture | None = None
        self._last_t: float | None = None

    @property
    def metrics(self) -> ScoredPosture | None:
        """Current baseline, or None before the first update."""
        return self._metrics

    def update(self, metrics: ScoredPosture, now: float) -> None:
        if self._metrics is None or self._last_t is None:
            self._metrics = metrics
            self._last_t = now
            return
        alpha = min(1.0, max(0.0, (now - self._last_t) / self._window_s))
        self._metrics = ScoredPosture(
            head_forward_deg=self._metrics.head_forward_deg
            + alpha * (metrics.head_forward_deg - self._metrics.head_forward_deg),
            trunk_lean_deg=self._metrics.trunk_lean_deg
            + alpha * (metrics.trunk_lean_deg - self._metrics.trunk_lean_deg),
            head_tilt_deg=self._metrics.head_tilt_deg
            + alpha * (metrics.head_tilt_deg - self._metrics.head_tilt_deg),
            shoulder_level_deg=self._metrics.shoulder_level_deg
            + alpha * (metrics.shoulder_level_deg - self._metrics.shoulder_level_deg),
            depth_m=self._metrics.depth_m
            + alpha * (metrics.depth_m - self._metrics.depth_m),
        )
        self._last_t = now

    def reset(self) -> None:
        self._metrics = None
        self._last_t = None


def deviation_zone(
    metrics: ScoredPosture, baseline: PostureBaseline, cfg: ScoringConfig
) -> str:
    """Classify deviation from the slow baseline, reusing the per-metric bands
    and the same hysteresis as zone_of. Before any baseline exists there is no
    reference to deviate from, so the frame reads "good" (zone_of still decides
    the zone on absolute angles)."""
    ref = baseline.metrics
    if ref is None:
        return "good"
    pairs = [
        (metrics.head_forward_deg - ref.head_forward_deg, cfg.head_forward_deg),
        (metrics.head_tilt_deg - ref.head_tilt_deg, cfg.head_tilt_deg),
        (metrics.shoulder_level_deg - ref.shoulder_level_deg, cfg.shoulder_level_deg),
    ]
    if cfg.trunk_lean_required and metrics.trunk_lean_available:
        pairs.append((metrics.trunk_lean_deg - ref.trunk_lean_deg, cfg.trunk_lean_deg))
    if any(abs(value) > threshold for value, threshold in pairs):
        return "bad"
    if all(abs(value) < cfg.bad_exit(threshold) for value, threshold in pairs):
        return "good"
    return "neutral"
