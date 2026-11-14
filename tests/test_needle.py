"""Tests for the dashboard posture needle aggregation (display-only math)."""

from __future__ import annotations

import pytest

from posture_guard.dashboard.needle import composite_score, metric_severity

THRESHOLDS = {
    "head_forward_deg": 12.0,
    "head_tilt_deg": 12.0,
    "shoulder_level_deg": 6.0,
    "trunk_lean_deg": 10.0,
    "hysteresis_margin_deg": 3.0,
    "trunk_lean_required": True,
}
# Exit fractions: head metrics (12-3)/12 = 0.75, shoulders (6-3)/6 = 0.5,
# trunk (10-3)/10 = 0.7. Gauge thirds: green below 100/3, amber below 200/3.


def test_centered_scores_zero():
    metrics = {
        "head_forward_deg": 0.0,
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": 0.0,
    }
    assert composite_score(metrics, THRESHOLDS) == 0.0


def test_worst_metric_sets_the_score():
    metrics = {
        "head_forward_deg": 6.0,  # severity 0.5 of exit 0.75
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": 0.0,
    }
    assert composite_score(metrics, THRESHOLDS) == pytest.approx(200.0 / 9.0)


def test_enter_threshold_maps_to_red_boundary_and_saturates():
    metrics = {
        "head_forward_deg": 12.0,
        "head_tilt_deg": 12.0,
        "shoulder_level_deg": 6.0,
        "trunk_lean_deg": 10.0,
    }
    assert composite_score(metrics, THRESHOLDS) == pytest.approx(200.0 / 3.0)
    over = {key: value * 2 for key, value in metrics.items()}
    assert composite_score(over, THRESHOLDS) == pytest.approx(100.0)


def test_unavailable_metrics_are_skipped():
    # Trunk lean without visible hips: None value contributes nothing.
    metrics = {
        "head_forward_deg": 12.0,  # at band -> red boundary
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": None,
    }
    assert composite_score(metrics, THRESHOLDS) == pytest.approx(200.0 / 3.0)


def test_trunk_counts_when_present_ignored_when_hidden():
    metrics = {
        "head_forward_deg": 0.0,
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": 20.0,  # 2x over -> red in every mode
    }
    optional = dict(THRESHOLDS, trunk_lean_required=False)
    assert composite_score(metrics, optional) == pytest.approx(100.0)
    assert composite_score(metrics, THRESHOLDS) == pytest.approx(100.0)
    hidden = dict(metrics, trunk_lean_deg=None)
    assert composite_score(hidden, optional) == 0.0


def test_missing_threshold_skips_metric_and_none_when_nothing_usable():
    metrics = {"head_forward_deg": 12.0, "trunk_lean_deg": 10.0}
    # No margin key -> exit equals the band, so severity 1 lands on GOOD_END.
    assert composite_score(metrics, {"head_forward_deg": 12.0}) == pytest.approx(
        100.0 / 3.0
    )
    assert composite_score({"trunk_lean_deg": None}, THRESHOLDS) is None
    assert composite_score({}, THRESHOLDS) is None
    assert composite_score({}, None) is None


def test_metric_severity_clamps_and_rejects_degenerate_bands():
    assert metric_severity(6.0, 12.0) == 0.5
    assert metric_severity(-12.0, 12.0) == 1.0
    assert metric_severity(24.0, 12.0) == 1.0
    assert metric_severity(None, 12.0) is None
    assert metric_severity(6.0, None) is None
    assert metric_severity(6.0, 0.0) is None
