"""Deterministic state-machine timeline tests.

Timestamps are injected; there are no sleeps and no wall-clock reads, so the
chime/voice schedule is asserted to the exact sample. Sample grid is 10 Hz
(0.1 s). Reference schedule for the default config:

    person appears 0.0        -> GOOD confirmed at 1.0
    slouch starts 10.0        -> chime at 12.0 (bad_sustain)
                              -> voice at 22.0 (voice_delay after chime)
                              -> voice repeats at 52.0, 82.0 (30 s cooldown)
    good from 20.1 in the recovery case -> back to GOOD at 23.1
"""

from __future__ import annotations

import pytest

from conftest import make_pose
from posture_guard.config import FeedbackConfig, PositiveChimeConfig, ScoringConfig, StateConfig
from posture_guard.feedback import RuntimeFeedback
from posture_guard.scoring import (
    LEFT_EAR,
    RIGHT_EAR,
    RIGHT_SHOULDER,
    ScoredPosture,
    SimpleLandmark,
    Unreliable,
    UnreliableReason,
    score_pose,
)
from posture_guard.state import Nudge, PostureState, PostureStateMachine

CFG = ScoringConfig()
TIMINGS = StateConfig()

GOOD = score_pose(make_pose(), 1.4, CFG)


def _bad_pose_score() -> ScoredPosture:
    """A slouched pose (head-forward ~18.4 deg) that lands in the bad zone."""
    pose = make_pose(
        {
            LEFT_EAR: SimpleLandmark(x=0.48, y=0.54),
            RIGHT_EAR: SimpleLandmark(x=0.56, y=0.54),
        }
    )
    return score_pose(pose, 1.4, CFG)


NO_POSE = Unreliable(UnreliableReason.NO_POSE)
TOO_FAR = Unreliable(UnreliableReason.DEPTH_OUT_OF_BAND)
LOW_VIS = Unreliable(UnreliableReason.LOW_VISIBILITY)


def run(machine: PostureStateMachine, samples: list[tuple[float, object]]) -> list[object]:
    return [machine.update(result, now) for now, result in samples]


def grid(start: float, stop: float, step: float, result) -> list[tuple[float, object]]:  # noqa: ANN001
    now = start
    out = []
    while now <= stop + 1e-9:
        out.append((round(now, 3), result))
        now += step
    return out


class TestPositiveChime:
    """Positive reinforcement: in GOOD state, fire at most once per
    every_seconds of continuously good posture; the timer restarts on fire and
    on any departure from GOOD."""

    POSITIVE = FeedbackConfig(positive_chime=PositiveChimeConfig(enabled=True, every_seconds=10.0))

    def _machine(self, positive: FeedbackConfig | None = POSITIVE) -> PostureStateMachine:
        return PostureStateMachine(TIMINGS, CFG, positive_chime=positive)

    @staticmethod
    def _positive_times(samples, transitions):  # noqa: ANN001
        return [
            now for (now, _), transition in zip(samples, transitions, strict=False)
            if transition.nudge is Nudge.POSITIVE_CHIME
        ]

    def test_fires_only_after_sustained_good(self) -> None:
        machine = self._machine()
        samples = grid(0.0, 11.5, 0.1, GOOD)
        transitions = run(machine, samples)
        # GOOD confirmed at 1.0 -> first positive chime one full interval later.
        assert self._positive_times(samples, transitions) == [11.0]

    def test_cooldown_restarts_after_each_fire(self) -> None:
        machine = self._machine()
        samples = grid(0.0, 21.5, 0.1, GOOD)
        transitions = run(machine, samples)
        assert self._positive_times(samples, transitions) == [11.0, 21.0]

    def test_disabled_never_fires(self) -> None:
        machine = self._machine(
            FeedbackConfig(positive_chime=PositiveChimeConfig(enabled=False, every_seconds=10.0))
        )
        samples = grid(0.0, 25.0, 0.1, GOOD)
        transitions = run(machine, samples)
        assert self._positive_times(samples, transitions) == []

    def test_bad_episode_resets_the_interval(self) -> None:
        machine = self._machine()
        samples = grid(0.0, 12.0, 0.1, GOOD)
        samples += grid(12.1, 16.0, 0.1, _bad_pose_score())
        samples += grid(16.1, 29.5, 0.1, GOOD)
        transitions = run(machine, samples)
        # Recovered to GOOD at ~19.1 (recovery_good_s) -> next fire 10 s later.
        assert self._positive_times(samples, transitions) == pytest.approx([11.0, 29.2])

    def test_unreliable_frames_reset_the_interval(self) -> None:
        machine = self._machine()
        samples = grid(0.0, 9.0, 0.1, GOOD)
        samples += grid(9.1, 9.5, 0.1, LOW_VIS)
        samples += grid(9.6, 12.0, 0.1, GOOD)
        transitions = run(machine, samples)
        # GOOD re-confirmed at ~9.6 after the unreliable gap -> no fire by 12.0.
        assert self._positive_times(samples, transitions) == []


class TestLiveVoiceTimings:
    """Runtime voice-timing override: edits take effect with no restart
    (spec acceptance: verified by a state-machine test with a runtime override)."""

    def _machine(self, feedback: RuntimeFeedback) -> PostureStateMachine:
        return PostureStateMachine(TIMINGS, CFG, timings_override=feedback)

    @staticmethod
    def _voice_times(samples, transitions):  # noqa: ANN001
        return [
            now for (now, _), transition in zip(samples, transitions, strict=False)
            if transition.nudge is Nudge.VOICE
        ]

    def test_override_shortens_voice_delay_and_cooldown(self) -> None:
        feedback = RuntimeFeedback(voice_delay_s=3.0, voice_repeat_cooldown_s=6.0)
        machine = self._machine(feedback)

        samples = grid(0.0, 12.0, 0.1, _bad_pose_score())
        transitions = run(machine, samples)

        voice_times = self._voice_times(samples, transitions)
        # Chime at 2.0 (bad_sustain); voice 3 s later; repeat 6 s after that.
        assert voice_times[0] == pytest.approx(5.0)
        assert voice_times[1] == pytest.approx(11.0)

    def test_mid_run_edit_takes_effect_on_the_next_nudge(self) -> None:
        feedback = RuntimeFeedback()  # default 10 s delay at start
        machine = self._machine(feedback)

        samples = grid(0.0, 6.0, 0.1, _bad_pose_score())
        transitions = run(machine, samples)
        assert self._voice_times(samples, transitions) == []  # no voice before 6 s
        chime_times = [
            now for (now, _), transition in zip(samples, transitions, strict=False)
            if transition.nudge is Nudge.CHIME
        ]
        assert chime_times == [pytest.approx(2.0)]

        # Presenter tightens the delay mid-demo; the first sample that now
        # satisfies the edited delay fires the voice prompt.
        feedback.apply(voice_delay_s=1.5)
        samples = grid(6.0, 8.0, 0.1, _bad_pose_score())
        transitions = run(machine, samples)

        voice_times = self._voice_times(samples, transitions)
        assert voice_times == [pytest.approx(6.0)]  # 6.0 - chime 2.0 >= edited delay 1.5

    def test_no_override_keeps_config_schedule(self) -> None:
        machine = PostureStateMachine(TIMINGS, CFG)

        # Bad from t=0: chime at 2.0 (bad_sustain), voice at 12.0 (config
        # delay 10.0), repeat at 42.0 (config cooldown 30.0).
        samples = grid(0.0, 43.0, 0.1, _bad_pose_score())
        transitions = run(machine, samples)

        voice_times = self._voice_times(samples, transitions)
        assert voice_times[0] == pytest.approx(12.0)
        assert voice_times[1] == pytest.approx(42.0)


class TestNudgeSchedule:
    @staticmethod
    def _at(samples, transitions, state, nudge):  # noqa: ANN001
        """(timestamp, transition) pairs where state and nudge match."""
        return [
            (now, tr)
            for (now, _), tr in zip(samples, transitions, strict=True)
            if tr.state is state and tr.nudge is nudge
        ]

    def test_full_timeline_chime_and_voice_schedule(self):
        """Good 0.0-9.9 -> GOOD at 1.0; slouch from 10.0 -> chime exactly at
        12.0 (+2 s of bad), voice exactly at 22.0 (chime + 10 s)."""
        sm = PostureStateMachine(TIMINGS, CFG)
        samples = grid(0.0, 9.9, 0.1, GOOD) + grid(10.0, 30.0, 0.1, _bad_pose_score())
        transitions = run(sm, samples)

        good_entries = self._at(samples, transitions, PostureState.GOOD, Nudge.NONE)
        assert good_entries[0][0] == 1.0

        chimes = self._at(samples, transitions, PostureState.BAD_CHIME, Nudge.CHIME)
        assert len(chimes) == 1
        assert chimes[0][0] == 12.0
        assert chimes[0][1].seconds_bad == pytest.approx(2.0)

        voices = self._at(samples, transitions, PostureState.BAD_VOICE, Nudge.VOICE)
        assert voices[0][0] == 22.0
        assert voices[0][1].state is PostureState.BAD_VOICE

    def test_voice_repeats_at_most_every_30s(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        samples = grid(0.0, 9.9, 0.1, GOOD) + grid(10.0, 90.0, 0.1, _bad_pose_score())
        transitions = run(sm, samples)

        voices = self._at(samples, transitions, PostureState.BAD_VOICE, Nudge.VOICE)
        voice_times = [t for t, _ in voices]
        assert voice_times == [22.0, 52.0, 82.0]
        assert all(
            later - earlier >= TIMINGS.voice_repeat_cooldown_s - 1e-9
            for earlier, later in zip(voice_times, voice_times[1:], strict=False)
        )

    def test_no_voice_before_voice_delay_elapses(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        samples = grid(0.0, 9.9, 0.1, GOOD) + grid(10.0, 21.9, 0.1, _bad_pose_score())
        transitions = run(sm, samples)
        # Voice is due at 22.0 (chime 12.0 + 10 s); the timeline stops at
        # 21.9, so no VOICE may have fired yet.
        assert all(tr.nudge is not Nudge.VOICE for tr in transitions)

    def test_recovery_to_good_after_3s_of_good_is_silent(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        samples = (
            grid(0.0, 9.9, 0.1, GOOD)
            + grid(10.0, 20.0, 0.1, _bad_pose_score())
            + grid(20.1, 30.0, 0.1, GOOD)
        )
        transitions = run(sm, samples)

        recovered = [
            (now, tr)
            for (now, _), tr in zip(samples, transitions, strict=True)
            if tr.state is PostureState.GOOD and now > 12.0  # after the chime at 12.0
        ]
        # good_since = 20.1 (first good sample after the bad streak);
        # recovery needs recovery_good (3.0) from that streak start.
        assert recovered[0][0] == 23.1
        assert recovered[0][1].nudge is Nudge.NONE


class TestSmoothingAndHysteresis:
    def test_single_bad_frame_does_not_leave_good(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        transitions = run(
            sm, grid(0.0, 3.0, 0.1, GOOD) + [(3.1, _bad_pose_score())] + grid(3.2, 5.0, 0.1, GOOD)
        )
        assert all(tr.state is not PostureState.BAD_CHIME for tr in transitions)

    def test_hysteresis_band_holds_bad_state(self):
        """Uneven shoulders (~5.7 deg) sits between exit (3) and enter (6):
        neither recovered nor re-entered, so BAD_CHIME holds through 21.9 —
        silent, with voice escalation due only at 22.0."""
        neutral = score_pose(make_pose({RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.62)}), 1.4, CFG)
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 9.9, 0.1, GOOD))
        run(sm, grid(10.0, 12.0, 0.1, _bad_pose_score()))
        assert sm.state is PostureState.BAD_CHIME
        tail = run(sm, grid(12.1, 21.9, 0.1, neutral))
        assert all(tr.state is PostureState.BAD_CHIME for tr in tail)
        assert all(tr.nudge is not Nudge.VOICE for tr in tail)


class TestUnreliableAndAbsent:
    def test_low_visibility_is_silent_and_resets_bad_streak(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 10.0, 0.1, GOOD))
        run(sm, grid(10.1, 11.5, 0.1, _bad_pose_score()))
        tr = sm.update(LOW_VIS, 11.6)
        assert tr.state is PostureState.UNRELIABLE
        assert tr.nudge is Nudge.NONE

        # Escalation restarts: bad must be sustained afresh after data returns.
        samples = grid(11.7, 13.0, 0.1, _bad_pose_score())
        transitions = run(sm, samples)
        assert all(tr.nudge is not Nudge.CHIME for tr in transitions)

    def test_depth_unreadable_is_unreliable(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        tr = sm.update(Unreliable(UnreliableReason.DEPTH_UNREADABLE), 5.0)
        assert tr.state is PostureState.UNRELIABLE

    def test_no_pose_times_out_to_no_person(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 5.0, 0.1, GOOD))
        samples = grid(5.1, 8.0, 0.1, NO_POSE)
        transitions = run(sm, samples)
        no_person = [
            (now, tr)
            for (now, _), tr in zip(samples, transitions, strict=True)
            if tr.state is PostureState.NO_PERSON
        ]
        assert no_person[0][0] == pytest.approx(7.1)  # absent_since 5.1 + 2.0 timeout

    def test_person_leaving_cancels_pending_voice(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 9.9, 0.1, GOOD))
        run(sm, grid(10.0, 12.0, 0.1, _bad_pose_score()))
        assert sm.state is PostureState.BAD_CHIME
        # Chime was at 12.0; person leaves before chime_at + 10.0.
        transitions = run(sm, grid(12.1, 25.0, 0.1, NO_POSE))
        assert any(tr.state is PostureState.NO_PERSON for tr in transitions)
        assert all(tr.nudge is not Nudge.VOICE for tr in transitions)

    def test_out_of_band_depth_feeds_no_person_timer(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 5.0, 0.1, GOOD))
        transitions = run(sm, grid(5.1, 8.0, 0.1, TOO_FAR))
        assert transitions[-1].state is PostureState.NO_PERSON

    def test_person_returning_from_no_person_gets_good_confirmation(self):
        sm = PostureStateMachine(TIMINGS, CFG)
        run(sm, grid(0.0, 5.0, 0.1, GOOD))
        run(sm, grid(5.1, 8.0, 0.1, NO_POSE))
        transitions = run(sm, grid(8.1, 12.0, 0.1, GOOD))
        assert transitions[-1].state is PostureState.GOOD
        assert all(tr.nudge is not Nudge.CHIME for tr in transitions)

    def test_neutral_posture_from_no_person_transitions_to_good(self):
        """A person sitting down in neutral posture (within enter thresholds)
        must transition to GOOD and not get trapped in NO_PERSON."""
        neutral = score_pose(make_pose({RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.62)}), 1.4, CFG)
        sm = PostureStateMachine(TIMINGS, CFG)
        transitions = run(sm, grid(0.0, 2.0, 0.1, neutral))
        assert transitions[-1].state is PostureState.GOOD
