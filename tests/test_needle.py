"""Tests for the dashboard posture needle aggregation (display-only math)."""

from __future__ import annotations

from posture_guard.dashboard.needle import composite_score, metric_severity

THRESHOLDS = {
    "head_forward_deg": 12.0,
    "head_tilt_deg": 12.0,
    "shoulder_level_deg": 6.0,
    "trunk_lean_deg": 10.0,
}


def test_centered_scores_zero():
    metrics = dict.fromkeys(THRESHOLDS, 0.0)
    assert composite_score(metrics, THRESHOLDS) == 0.0


def test_mixed_metrics_average_to_midpoint():
    metrics = {
        "head_forward_deg": 6.0,  # half of its band
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": 0.0,
    }
    assert composite_score(metrics, THRESHOLDS) == 12.5  # 100 * 0.5 / 4


def test_at_threshold_saturates_to_100():
    metrics = {key: threshold for key, threshold in THRESHOLDS.items()}
    assert composite_score(metrics, THRESHOLDS) == 100.0
    over = {key: threshold * 2 for key, threshold in THRESHOLDS.items()}
    assert composite_score(over, THRESHOLDS) == 100.0


def test_unavailable_metrics_are_skipped():
    # Trunk lean without visible hips: None value contributes nothing.
    metrics = {
        "head_forward_deg": 12.0,  # at band -> severity 1
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "trunk_lean_deg": None,
    }
    assert composite_score(metrics, THRESHOLDS) == 100.0 / 3.0  # pytest.approx-free: exact in binary


def test_missing_threshold_skips_metric_and_none_when_nothing_usable():
    metrics = {"head_forward_deg": 12.0, "trunk_lean_deg": 10.0}
    assert composite_score(metrics, {"head_forward_deg": 12.0}) == 100.0
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
