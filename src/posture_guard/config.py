"""Load config.yaml into typed dataclasses.

The dataclasses here are the single source of truth for thresholds and timings;
scoring.py and state.py take them as parameters, so a config edit is the only
edit needed to retune the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class ScoringConfig:
    """Thresholds and gates for posture scoring.

    A metric is bad when |value| exceeds its threshold; it only counts as good
    again once |value| falls below (threshold - hysteresis_margin_deg).
    """

    head_forward_deg: float = 12.0
    trunk_lean_deg: float = 10.0
    head_tilt_deg: float = 12.0
    shoulder_level_deg: float = 6.0
    hysteresis_margin_deg: float = 3.0
    visibility_floor: float = 0.5
    depth_min_m: float = 0.6
    depth_max_m: float = 2.5

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
    chime_text: str = "fix your posture"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    state: StateConfig = field(default_factory=StateConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)


def _build(cls, data: dict | None):  # noqa: ANN201 - factory helper
    """Construct a frozen config dataclass, ignoring keys it does not define."""
    data = data or {}
    known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


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
    )
