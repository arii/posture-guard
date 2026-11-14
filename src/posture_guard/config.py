"""Load config.yaml into typed dataclasses.

The dataclasses here are the single source of truth for thresholds and timings;
scoring.py and state.py take them as parameters, so a config edit is the only
edit needed to retune the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import get_type_hints

import yaml


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class BaselineDeviationConfig:
    """Slow-baseline deviation check (whole-torso slouch gate).

    An exponential moving average of each scored angle over roughly window_s
    tracks the user's own recent neutral. Deviation thresholds REUSE the
    per-metric bands (head_forward_deg, trunk_lean_deg, ...): a metric is a
    deviation breach when |metric - baseline| passes the same threshold the
    raw metric would use, with the same hysteresis. This catches slouch that
    keeps head-vs-shoulder angles small (head and shoulders drift together)
    while an absolute angle change from the person's neutral grows.
    """

    enabled: bool = True
    window_s: float = 300.0


@dataclass(frozen=True)
class ScoringConfig:
    """Thresholds and gates for posture scoring.

    A metric is bad when |value| exceeds its threshold; it only counts as good
    again once |value| falls below (threshold - hysteresis_margin_deg).
    """

    head_forward_deg: float = 8.0
    trunk_lean_deg: float = 7.0
    head_tilt_deg: float = 8.0
    shoulder_level_deg: float = 4.0
    hysteresis_margin_deg: float = 3.0
    visibility_floor: float = 0.5
    depth_min_m: float = 0.6
    depth_max_m: float = 2.5
    baseline_deviation: BaselineDeviationConfig = field(
        default_factory=BaselineDeviationConfig
    )
    # High-mount framing can keep hips permanently below the visibility floor;
    # set false to score ears/shoulders while the hips are hidden (trunk lean
    # then reports unavailable and drops out of the zone). Whenever the hips
    # ARE visible the trunk angle counts toward the zone either way.
    trunk_lean_required: bool = True
    # Degraded feedback: when trunk lean is required but only the hips fall
    # below the visibility floor, still score the ears/shoulders metrics (head
    # forward, head tilt, shoulder level) and report trunk lean as unavailable,
    # instead of gating all feedback. Ears or shoulders below the floor always
    # gates — they are the minimum usable set.
    degraded_feedback: bool = True

    def bad_exit(self, enter_threshold: float) -> float:
        """Exit threshold for a metric, per the hysteresis margin."""
        return enter_threshold - self.hysteresis_margin_deg


@dataclass(frozen=True)
class StateConfig:
    smoothing_window_s: float = 0.3
    bad_sustain_s: float = 2.0
    good_confirm_s: float = 1.0
    recovery_good_s: float = 3.0
    voice_delay_s: float = 10.0
    voice_repeat_cooldown_s: float = 30.0
    no_person_timeout_s: float = 2.0


@dataclass(frozen=True)
class AudioConfig:
    sink: str | None = None
    # Spoken-prompt wording rendered by TTS (the chime is a synthesized tone;
    # "chime_text" elsewhere in the UI/API means this spoken wording).
    voice_text: str = "Please fix your posture"
    positive_voice_text: str | None = None


@dataclass(frozen=True)
class PositiveChimeConfig:
    """Positive reinforcement: an upbeat chime after posture stays GOOD for
    `every_seconds`; the timer restarts each time it plays."""

    enabled: bool = True
    every_seconds: float = 120.0


@dataclass(frozen=True)
class FeedbackConfig:
    positive_chime: PositiveChimeConfig = field(default_factory=PositiveChimeConfig)


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    state: StateConfig = field(default_factory=StateConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)


def _build(cls, data: dict | None):  # noqa: ANN201 - factory helper
    """Construct a frozen config dataclass, ignoring keys it does not define
    and building nested dataclass-typed fields (e.g. ScoringConfig.
    baseline_deviation) from their mappings instead of passing the raw dict
    through — a raw dict leaks past the dataclass and crashes attribute
    access at startup."""
    data = data or {}
    hints = get_type_hints(cls)
    kwargs: dict[str, object] = {}
    for key, value in data.items():
        if key not in cls.__dataclass_fields__:  # type: ignore[attr-defined]
            continue
        nested = hints.get(key)
        nested_dataclass = (
            isinstance(value, dict)
            and nested is not None
            and hasattr(nested, "__dataclass_fields__")
        )
        if nested_dataclass:
            value = _build(nested, value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate config from a YAML file.

    Raises FileNotFoundError if the file is missing and yaml.YAMLError if it
    is malformed — both are surface-loud by design.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise yaml.YAMLError(f"{path}: expected a mapping at the top level")
    return AppConfig(
        camera=_build(CameraConfig, raw.get("camera")),
        scoring=_build(ScoringConfig, raw.get("scoring")),
        state=_build(StateConfig, raw.get("state")),
        audio=_build(AudioConfig, raw.get("audio")),
        feedback=_build_feedback(raw.get("feedback")),
    )


def _build_feedback(data: dict | None) -> FeedbackConfig:  # noqa: ANN201 - factory helper
    """Build FeedbackConfig; positive_chime is a nested mapping."""
    data = data or {}
    return FeedbackConfig(
        positive_chime=_build(PositiveChimeConfig, data.get("positive_chime")),
    )
