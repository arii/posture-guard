"""Scoring tests: pure metric functions, gates, and zone classification.

All angle assertions use a 1-degree tolerance per the spec acceptance table.
Expected values are the precomputed constants in conftest.py.
"""

import pytest

from conftest import GOOD_ANGLES, SLOUCHED_ANGLES
from posture_guard.config import ScoringConfig
from posture_guard.scoring import (
    ScoredPosture,
    Unreliable,
    UnreliableReason,
    measure_head_forward,
    measure_head_tilt,
    measure_shoulder_level,
    measure_trunk_lean,
    score_pose,
    valid_depth_median,
    visibility_ok,
    zone_of,
)

ANGLE_TOLERANCE_DEG = 1.0
CFG = ScoringConfig()
SLOUCHED_EXPECTED = 18.4349488  # atan2(0.02, 0.06), see conftest slouched_pose


def assert_matches(scored: ScoredPosture, expected) -> None:  # noqa: ANN001 - conftest NamedTuple
    assert scored.head_forward_deg == pytest.approx(
        expected.head_forward_deg, abs=ANGLE_TOLERANCE_DEG
    )
    assert scored.trunk_lean_deg == pytest.approx(expected.trunk_lean_deg, abs=ANGLE_TOLERANCE_DEG)
    assert scored.head_tilt_deg == pytest.approx(expected.head_tilt_deg, abs=ANGLE_TOLERANCE_DEG)
    assert scored.shoulder_level_deg == pytest.approx(
        expected.shoulder_level_deg, abs=ANGLE_TOLERANCE_DEG
    )


class TestMetrics:
    def test_good_pose_all_metrics_near_zero(self, good_pose):
        assert measure_head_forward(good_pose) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert measure_trunk_lean(good_pose) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert measure_head_tilt(good_pose) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)
        assert measure_shoulder_level(good_pose) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)

    def test_slouched_head_forward(self, slouched_pose):
        assert measure_head_forward(slouched_pose) == pytest.approx(
            SLOUCHED_EXPECTED, abs=ANGLE_TOLERANCE_DEG
        )

    def test_leaning_trunk(self, leaning_pose):
        assert measure_trunk_lean(leaning_pose) == pytest.approx(9.46, abs=ANGLE_TOLERANCE_DEG)

    def test_tilted_head(self, tilted_head_pose):
        assert measure_head_tilt(tilted_head_pose) == pytest.approx(11.31, abs=ANGLE_TOLERANCE_DEG)

    def test_uneven_shoulders(self, uneven_shoulders_pose):
        assert measure_shoulder_level(uneven_shoulders_pose) == pytest.approx(
            5.71, abs=ANGLE_TOLERANCE_DEG
        )



class TestScorePoseGates:
    def test_good_pose_scores(self, good_pose):
        result = score_pose(good_pose, depth_m=1.4, cfg=CFG)
        assert_matches(result, GOOD_ANGLES)
        assert result.depth_m == 1.4

    def test_slouched_pose_scores_bad(self, slouched_pose):
        result = score_pose(slouched_pose, depth_m=1.4, cfg=CFG)
        assert_matches(result, SLOUCHED_ANGLES)

    def test_no_pose_is_unreliable_no_pose(self):
        result = score_pose(None, depth_m=1.4, cfg=CFG)
        assert result == Unreliable(UnreliableReason.NO_POSE)

    def test_missing_depth_is_unreliable(self, good_pose):
        result = score_pose(good_pose, depth_m=None, cfg=CFG)
        assert result.reason == UnreliableReason.DEPTH_UNREADABLE

    def test_invalid_depth_is_unreliable(self, good_pose):
        result = score_pose(good_pose, depth_m=0.0, cfg=CFG)
        assert result.reason == UnreliableReason.DEPTH_UNREADABLE

    def test_person_too_far_is_out_of_band(self, good_pose):
        result = score_pose(good_pose, depth_m=2.9, cfg=CFG)
        assert result.reason == UnreliableReason.DEPTH_OUT_OF_BAND
        assert "2.9" in result.detail

    def test_person_too_close_is_out_of_band(self, good_pose):
        result = score_pose(good_pose, depth_m=0.4, cfg=CFG)
        assert result.reason == UnreliableReason.DEPTH_OUT_OF_BAND

    def test_band_edges_score(self, good_pose):
        assert isinstance(score_pose(good_pose, depth_m=CFG.depth_min_m, cfg=CFG), ScoredPosture)
        assert isinstance(score_pose(good_pose, depth_m=CFG.depth_max_m, cfg=CFG), ScoredPosture)

    def test_cropped_pose_is_unreliable_low_visibility(self, cropped_pose):
        result = score_pose(cropped_pose, depth_m=1.4, cfg=CFG)
        assert result.reason == UnreliableReason.LOW_VISIBILITY

    def test_low_visibility_never_returns_a_score(self, cropped_pose):
        assert not isinstance(score_pose(cropped_pose, depth_m=1.4, cfg=CFG), ScoredPosture)


class TestVisibility:
    def test_all_visible_passes(self, good_pose):
        assert visibility_ok(good_pose, CFG)

    def test_custom_floor(self, good_pose):
        strict = ScoringConfig(visibility_floor=1.1)
        assert not visibility_ok(good_pose, strict)


class TestDepthMedian:
    def test_rejects_invalid_and_takes_median(self):
        samples = [0.0, 1.5, 1.4, float("nan"), 0.5]
        assert valid_depth_median(samples) == 1.4

    def test_even_count_interpolates(self):
        assert valid_depth_median([1.0, 2.0]) == 1.5

    def test_all_invalid_returns_none(self):
        assert valid_depth_median([0.0, -1.0, float("inf")]) is None


class TestZoneOf:
    def test_good_pose_is_good_zone(self, good_pose):
        assert zone_of(score_pose(good_pose, 1.4, CFG), CFG) == "good"

    def test_slouched_is_bad_zone(self, slouched_pose):
        assert zone_of(score_pose(slouched_pose, 1.4, CFG), CFG) == "bad"

    def test_hysteresis_band_is_neutral(self, uneven_shoulders_pose):
        # Shoulder level ~5.7 deg: past the exit threshold (6 - 3 = 3) but
        # below the enter threshold (6) — exactly the hysteresis band.
        assert zone_of(score_pose(uneven_shoulders_pose, 1.4, CFG), CFG) == "neutral"

    def test_identical_metric_values_all_checked(self):
        # Regression: value/threshold pairs must not collapse on dict keys.
        metrics = ScoredPosture(
            head_forward_deg=0.0,
            trunk_lean_deg=0.0,
            head_tilt_deg=0.0,
            shoulder_level_deg=10.0,
            depth_m=1.4,
        )
        assert zone_of(metrics, CFG) == "bad"
