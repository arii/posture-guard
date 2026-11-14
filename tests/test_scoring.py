"""Scoring tests: pure metric functions, gates, and zone classification.

All angle assertions use a 1-degree tolerance per the spec acceptance table.
Expected values are the precomputed constants in conftest.py.
"""

import math

import pytest

from conftest import GOOD_ANGLES, SLOUCHED_ANGLES, make_pose, mirror_x
from posture_guard.config import BaselineDeviationConfig, ScoringConfig
from posture_guard.scoring import (
    HEAD_CLIP_MARGIN,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    PostureBaseline,
    ScoredPosture,
    SimpleLandmark,
    Unreliable,
    UnreliableReason,
    deviation_zone,
    head_clipped_top,
    low_visibility_detail,
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


class TestTrunkLeanOptional:
    """scoring.trunk_lean_required: false — hips may fall below the visibility
    floor (high-mount framing); the remaining metrics still score and reach BAD."""

    def test_hidden_hips_still_score_with_flag_false(self):
        pose = make_pose(
            {
                LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.3),
                RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.3),
            }
        )
        cfg = ScoringConfig(trunk_lean_required=False)
        result = score_pose(pose, 1.4, cfg)
        assert isinstance(result, ScoredPosture)  # not UNRELIABLE low_visibility

    def test_hidden_hips_score_in_degraded_mode_by_default(self):
        """scoring.degraded_feedback (default true): hips below the floor no
        longer gate all feedback — the ears/shoulders metrics still score and
        trunk lean reports unavailable."""
        pose = make_pose(
            {
                LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.3),
                RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.3),
            }
        )
        result = score_pose(pose, 1.4, ScoringConfig())
        assert isinstance(result, ScoredPosture)
        assert result.trunk_lean_available is False

    def test_hidden_hips_block_scoring_when_degraded_feedback_off(self):
        """Flag off restores the previous gating entirely."""
        pose = make_pose(
            {
                LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.3),
                RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.3),
            }
        )
        cfg = ScoringConfig(degraded_feedback=False)
        result = score_pose(pose, 1.4, cfg)
        assert result == Unreliable(
            UnreliableReason.LOW_VISIBILITY,
            "hips not visible (left 0.30, right 0.30) - trunk lean unavailable",
        )

    def test_trunk_lean_excluded_from_bad_decision_when_optional(self):
        """A garbage trunk angle cannot make the zone BAD on its own; ears and
        shoulders still can."""
        pose = make_pose(
            {
                LEFT_HIP: SimpleLandmark(x=0.10, y=0.10, visibility=0.3),
                RIGHT_HIP: SimpleLandmark(x=0.90, y=0.10, visibility=0.3),
            }
        )
        cfg = ScoringConfig(trunk_lean_required=False)
        scored = score_pose(pose, 1.4, cfg)
        assert isinstance(scored, ScoredPosture)
        assert zone_of(scored, cfg) == "good"  # trunk lean ignored entirely


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


class TestMirroredFramings:
    """Mirrored / high side-mount framings: MediaPipe's RIGHT_SHOULDER can
    land left of LEFT_SHOULDER in image coordinates (dx < 0). Regression
    tests for the shoulder-level atan2 wrap (level shoulders read ~180 deg)."""

    def test_mirrored_level_shoulders_read_near_zero(self):
        # Right shoulder (0.30) left of the left one (0.40), shoulders
        # level: the raw atan2 wraps to 180 deg and fails the +/-6 deg
        # threshold.
        pose = make_pose({RIGHT_SHOULDER: SimpleLandmark(x=0.30, y=0.60)})
        assert measure_shoulder_level(pose) == pytest.approx(0.0, abs=ANGLE_TOLERANCE_DEG)

    def test_mirrored_good_pose_scores_good_zone(self):
        # End-to-end: a mirrored good pose must score good, not bad/unreliable.
        result = score_pose(mirror_x(make_pose()), depth_m=1.4, cfg=CFG)
        assert_matches(result, GOOD_ANGLES)
        assert zone_of(result, CFG) == "good"

    def test_right_shoulder_lower_sign_preserved_under_mirroring(self):
        # Right shoulder clearly lower (dy = +0.10 over 0.20 of width):
        # the tilt must read +26.6 deg in BOTH framings per the current
        # convention (positive = right shoulder lower) — not the folded
        # mirror (-26.6) and not a wrapped value near +/-180 or +/-153.
        tilt = math.degrees(math.atan2(0.10, 0.20))  # ~26.57 deg
        upright = make_pose(
            {
                LEFT_SHOULDER: SimpleLandmark(x=0.40, y=0.60),
                RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.70),
            }
        )
        assert measure_shoulder_level(upright) == pytest.approx(tilt, abs=ANGLE_TOLERANCE_DEG)
        assert measure_shoulder_level(mirror_x(upright)) == pytest.approx(
            tilt, abs=ANGLE_TOLERANCE_DEG
        )

    def test_head_tilt_magnitude_invariant_under_mirroring(self):
        # Head tilt is a relative angle between the inter-ear and
        # inter-shoulder vectors. x-mirroring negates both vectors' x
        # components: the dot product is unchanged and the cross product
        # negates (a reflection reverses orientation), so the signed angle
        # flips sign while its magnitude — the only thing the +/-12 deg
        # threshold consumes — is invariant. Verified, not assumed.
        tilted = make_pose(
            {
                LEFT_EAR: SimpleLandmark(x=0.45, y=0.46),
                RIGHT_EAR: SimpleLandmark(x=0.55, y=0.44),
            }
        )
        upright_tilt = measure_head_tilt(tilted)
        mirrored_tilt = measure_head_tilt(mirror_x(tilted))
        assert mirrored_tilt == pytest.approx(-upright_tilt, abs=ANGLE_TOLERANCE_DEG)
        assert abs(mirrored_tilt) == pytest.approx(
            abs(upright_tilt), abs=ANGLE_TOLERANCE_DEG
        )
        assert 11 < abs(mirrored_tilt) < 12  # expected magnitude ~11.31 deg

    def test_head_forward_and_trunk_lean_do_not_wrap_under_mirroring(self):
        # Head forward and trunk lean measure deviation from vertical via
        # atan2(dx, dy) with dy > 0 for any upright pose; mirroring only
        # negates dx, so the angle negates and can never wrap to +/-180.
        slouched = make_pose(
            {
                LEFT_EAR: SimpleLandmark(x=0.48, y=0.54),
                RIGHT_EAR: SimpleLandmark(x=0.56, y=0.54),
            }
        )
        leaning = make_pose(
            {
                LEFT_SHOULDER: SimpleLandmark(x=0.45, y=0.60),
                RIGHT_SHOULDER: SimpleLandmark(x=0.65, y=0.60),
                LEFT_EAR: SimpleLandmark(x=0.51, y=0.45),
                RIGHT_EAR: SimpleLandmark(x=0.59, y=0.45),
            }
        )
        for func, pose in (
            (measure_head_forward, slouched),
            (measure_trunk_lean, leaning),
        ):
            value = func(pose)
            mirrored_value = func(mirror_x(pose))
            assert mirrored_value == pytest.approx(-value, abs=ANGLE_TOLERANCE_DEG)
            assert abs(mirrored_value) < 90  # no +/-180 wrap


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

    def test_low_visibility_detail_names_failing_landmarks(self, cropped_pose):
        # Only the right shoulder is below the floor: name it with its value.
        result = score_pose(cropped_pose, depth_m=1.4, cfg=CFG)
        assert result.detail == "shoulders not visible (right 0.10)"

    def test_low_visibility_hips_detail_names_trunk_lean(self, good_pose):
        pose = make_pose(
            {
                LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.31),
                RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.28),
            }
        )
        result = score_pose(pose, depth_m=1.4, cfg=ScoringConfig(degraded_feedback=False))
        assert result.reason == UnreliableReason.LOW_VISIBILITY
        assert result.detail == (
            "hips not visible (left 0.31, right 0.28) - trunk lean unavailable"
        )

    def test_low_visibility_detail_joins_multiple_groups(self, good_pose):
        pose = make_pose(
            {
                LEFT_EAR: SimpleLandmark(x=0.46, y=0.45, visibility=0.2),
                LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.31),
                RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.28),
            }
        )
        result = score_pose(pose, depth_m=1.4, cfg=CFG)
        assert "ears not visible (left 0.20)" in result.detail
        assert "hips not visible (left 0.31, right 0.28)" in result.detail

    def test_low_visibility_detail_helper_matches_gate(self, good_pose):
        # All visible: the helper still returns a sane string (no crash), but
        # score_pose never reaches it.
        assert isinstance(low_visibility_detail(good_pose, CFG), str)


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
        # Shoulder level ~5.71 deg: past the exit threshold (8 - 3 = 5) but
        # below the enter threshold (8) — exactly the hysteresis band.
        cfg = ScoringConfig(shoulder_level_deg=8.0)
        assert zone_of(score_pose(uneven_shoulders_pose, 1.4, cfg), cfg) == "neutral"

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


class TestBaselineDeviation:
    """Slow-baseline deviation check: an EMA (~5 min) of each angle tracks the
    user's own recent neutral, so whole-torso slouch that keeps head-vs-shoulder
    angles small still breaches as deviation from that neutral."""

    @staticmethod
    def _metrics(head_forward=0.0, trunk=0.0, tilt=0.0, shoulder=0.0) -> ScoredPosture:
        return ScoredPosture(
            head_forward_deg=head_forward,
            trunk_lean_deg=trunk,
            head_tilt_deg=tilt,
            shoulder_level_deg=shoulder,
            depth_m=1.4,
        )

    def test_baseline_anchors_to_first_frame(self):
        baseline = PostureBaseline(BaselineDeviationConfig())
        assert baseline.metrics is None
        baseline.update(self._metrics(head_forward=-2.0), 0.0)
        assert baseline.metrics is not None
        assert baseline.metrics.head_forward_deg == pytest.approx(-2.0)

    def test_steady_pose_is_no_deviation(self):
        """A person whose natural neutral is -2 degrees never deviates while
        sitting there — even though zone_of alone would call it good."""
        cfg = ScoringConfig()
        baseline = PostureBaseline(BaselineDeviationConfig())
        for i in range(300):
            baseline.update(self._metrics(head_forward=-2.0), i * 0.1)
        zone = deviation_zone(self._metrics(head_forward=-2.0), baseline, cfg)
        assert zone == "good"

    def test_sustained_drift_beyond_baseline_breaches(self):
        """Neutral calibrated at -2 deg; the user drifts to +7 deg. The raw
        angle is still inside the 8 deg band (zone_of: good) but the 9 deg
        swing from their own neutral breaches."""
        cfg = ScoringConfig()
        baseline = PostureBaseline(BaselineDeviationConfig())
        for i in range(300):
            baseline.update(self._metrics(head_forward=-2.0), i * 0.1)
        drifted = self._metrics(head_forward=7.0)
        # Raw check alone cannot reach BAD: 7 deg sits inside the 8 deg band.
        assert zone_of(drifted, cfg) != "bad"
        assert deviation_zone(drifted, baseline, cfg) == "bad"

    def test_deviation_uses_same_hysteresis_band(self):
        cfg = ScoringConfig()
        baseline = PostureBaseline(BaselineDeviationConfig())
        baseline.update(self._metrics(), 0.0)
        # 5.5 deg off the baseline: past the exit threshold (8 - 3 = 5), below
        # the enter threshold (8) — neutral, exactly like zone_of.
        assert deviation_zone(self._metrics(head_forward=5.5), baseline, cfg) == "neutral"

    def test_fast_transient_noise_does_not_breach(self):
        """A brief spike barely moves the EMA and never deviates."""
        cfg = ScoringConfig()
        baseline = PostureBaseline(BaselineDeviationConfig())
        for i in range(600):
            baseline.update(self._metrics(), i * 0.1)
        for j in range(3):  # 0.3 s of noise
            baseline.update(self._metrics(head_forward=30.0), 60.0 + j * 0.1)
        assert baseline.metrics is not None
        assert baseline.metrics.head_forward_deg < 1.0  # EMA barely moved
        assert deviation_zone(self._metrics(), baseline, cfg) == "good"

    def test_window_time_constant(self):
        """A constant offset converges to ~1 - 1/e of itself over window_s.
        Baseline is anchored on the first update, so hold a zero neutral first."""
        window_s = 300.0
        baseline = PostureBaseline(BaselineDeviationConfig(window_s=window_s))
        for i in range(300):  # 30 s of neutral anchors the baseline at 0
            baseline.update(self._metrics(), i * 0.1)
        for i in range(3001):  # window_s of 30 deg drift
            baseline.update(self._metrics(head_forward=30.0), 30.0 + i * 0.1)
        assert baseline.metrics is not None
        assert baseline.metrics.head_forward_deg == pytest.approx(
            30.0 * (1 - math.exp(-1)), rel=0.01
        )

    def test_reset_reanchors_on_next_update(self):
        baseline = PostureBaseline(BaselineDeviationConfig())
        baseline.update(self._metrics(head_forward=-2.0), 0.0)
        baseline.reset()
        assert baseline.metrics is None
        baseline.update(self._metrics(head_forward=5.0), 1.0)
        assert baseline.metrics is not None
        assert baseline.metrics.head_forward_deg == pytest.approx(5.0)


class TestDegradedFeedback:
    """scoring.degraded_feedback: UNRELIABLE only blocks feedback when the
    minimum usable set (ears AND shoulders) is not trackable. With ears and
    shoulders above the floor, the metrics they measure (head forward, head
    tilt, shoulder level) still score and trunk lean drops out of the
    decision — same pattern as trunk-lean-optional mode."""

    HIDDEN_HIPS = {
        LEFT_HIP: SimpleLandmark(x=0.44, y=0.90, visibility=0.2),
        RIGHT_HIP: SimpleLandmark(x=0.56, y=0.90, visibility=0.2),
    }

    def test_partial_visibility_reaches_bad_through_head_metrics(self):
        """Hips hidden, head slouched forward: the pose reaches BAD on the
        still-measurable head-forward metric (trunk lean is unavailable and
        excluded, and the garbage measured trunk angle cannot veto it)."""
        cfg = ScoringConfig()
        pose = make_pose(
            {
                **self.HIDDEN_HIPS,
                LEFT_EAR: SimpleLandmark(x=0.48, y=0.54),
                RIGHT_EAR: SimpleLandmark(x=0.56, y=0.54),
            }
        )
        scored = score_pose(pose, 1.4, cfg)
        assert isinstance(scored, ScoredPosture)
        assert scored.trunk_lean_available is False
        assert zone_of(scored, cfg) == "bad"

    def test_hidden_hip_garbage_angle_cannot_decide_the_zone(self):
        """Hips hidden: the measured trunk angle is garbage (hips at frame
        corners) but must not make the zone BAD on its own."""
        cfg = ScoringConfig()
        pose = make_pose(
            {
                **self.HIDDEN_HIPS,
                LEFT_HIP: SimpleLandmark(x=0.05, y=0.95, visibility=0.2),
                RIGHT_HIP: SimpleLandmark(x=0.95, y=0.05, visibility=0.2),
            }
        )
        scored = score_pose(pose, 1.4, cfg)
        assert isinstance(scored, ScoredPosture)
        assert zone_of(scored, cfg) == "good"

    def test_ears_below_floor_still_gates(self):
        """Ears are half of the minimum usable set: below the floor the frame
        is unscoreable even with degraded feedback on."""
        cfg = ScoringConfig()
        pose = make_pose(
            {
                **self.HIDDEN_HIPS,
                LEFT_EAR: SimpleLandmark(x=0.46, y=0.45, visibility=0.2),
            }
        )
        result = score_pose(pose, 1.4, cfg)
        assert result == Unreliable(
            UnreliableReason.LOW_VISIBILITY,
            "ears not visible (left 0.20); hips not visible (left 0.20, right 0.20)"
            " - trunk lean unavailable",
        )

    def test_shoulders_below_floor_still_gates(self):
        cfg = ScoringConfig()
        pose = make_pose(
            {
                LEFT_SHOULDER: SimpleLandmark(x=0.40, y=0.60, visibility=0.3),
            }
        )
        result = score_pose(pose, 1.4, cfg)
        assert isinstance(result, Unreliable)
        assert result.reason is UnreliableReason.LOW_VISIBILITY


# -- top-edge boundary check -----------------------------------------------------


def test_head_clipped_top_when_ear_in_top_margin() -> None:
    pose = make_pose({LEFT_EAR: SimpleLandmark(x=0.46, y=0.05)})
    assert head_clipped_top(pose) is True


def test_head_clipped_top_false_for_centered_pose() -> None:
    assert head_clipped_top(make_pose()) is False


def test_head_clipped_top_false_when_ear_at_margin_boundary() -> None:
    pose = make_pose({LEFT_EAR: SimpleLandmark(x=0.46, y=HEAD_CLIP_MARGIN)})
    assert head_clipped_top(pose) is False
