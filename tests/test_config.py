"""Config loading: YAML mappings become typed dataclasses, nested included.

Regression: a nested mapping (scoring.baseline_deviation) used to pass through
as a raw dict, crashing the state machine at startup — visible only via
load_config, never in tests that construct dataclasses directly.
"""

from __future__ import annotations

from posture_guard.config import (
    BaselineDeviationConfig,
    ScoringConfig,
    load_config,
)


def test_load_config_builds_nested_baseline_deviation(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "scoring:\n"
        "  head_forward_deg: 10.0\n"
        "  baseline_deviation:\n"
        "    enabled: false\n"
        "    window_s: 120.0\n"
    )
    config = load_config(config_file)
    baseline = config.scoring.baseline_deviation
    assert isinstance(baseline, BaselineDeviationConfig)
    assert baseline.enabled is False
    assert baseline.window_s == 120.0
    assert config.scoring.head_forward_deg == 10.0


def test_load_config_baseline_defaults_without_section(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("scoring:\n  head_tilt_deg: 9.0\n")
    config = load_config(config_file)
    assert isinstance(config.scoring.baseline_deviation, BaselineDeviationConfig)
    assert config.scoring.baseline_deviation.enabled is True


def test_load_config_unknown_keys_are_ignored(tmp_path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "scoring:\n"
        "  head_forward_deg: 10.0\n"
        "  not_a_real_field: 42\n"
    )
    config = load_config(config_file)
    assert isinstance(config.scoring, ScoringConfig)
