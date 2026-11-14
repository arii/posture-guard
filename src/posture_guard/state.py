"""Nudge state machine.

Consumes one ScoringResult per frame with an injected timestamp — logic never
reads the wall clock and never sleeps, so timelines are reproducible in tests.
States per the spec: NO_PERSON, GOOD, BAD_CHIME, BAD_VOICE, UNRELIABLE.

Mapping of unreliable reasons is deliberate: no pose and depth-out-of-band
mean "no person to nudge" (they feed the no-person timer); low visibility and
unreadable depth mean "do not trust this frame" (UNRELIABLE, immediate,
silent, resets any bad-posture streak so bad data can never escalate).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from posture_guard.config import FeedbackConfig, ScoringConfig, StateConfig
from posture_guard.scoring import (
    ScoringResult,
    Unreliable,
    UnreliableReason,
    zone_of,
)

_EPS = 1e-9  # float slack for window comparisons on the 0.1 s sample grid


class PostureState(Enum):
    NO_PERSON = "no_person"
    GOOD = "good"
    BAD_CHIME = "bad_chime"
    BAD_VOICE = "bad_voice"
    UNRELIABLE = "unreliable"


class Nudge(Enum):
    NONE = "none"
    CHIME = "chime"
    VOICE = "voice"
    POSITIVE_CHIME = "positive_chime"


_ABSENT_REASONS = (UnreliableReason.NO_POSE, UnreliableReason.DEPTH_OUT_OF_BAND)


class VoiceTimings(Protocol):
    """Runtime voice-timing override — RuntimeFeedback satisfies it."""

    def voice_timings(self) -> tuple[float, float]: ...


@dataclass(frozen=True)
class Transition:
    state: PostureState
    nudge: Nudge
    seconds_bad: float | None = None  # bad-posture streak length at nudge time


class PostureStateMachine:
    """Temporal smoothing (0.3 s window), hysteresis, and the nudge schedule:

    - chime once after bad posture is confirmed and sustained (default 2 s)
    - spoken prompt voice_delay_s after the chime if not corrected
    - spoken prompt repeats at most every voice_repeat_cooldown_s
    - GOOD after good_confirm_s of in-threshold posture; recovery from a bad
      state needs recovery_good_s of good posture
    - NO_PERSON after no_person_timeout_s of no pose / out-of-band depth
    """

    def __init__(
        self, timings: StateConfig, thresholds: ScoringConfig, *,
        timings_override: VoiceTimings | None = None,
        positive_chime: FeedbackConfig | None = None,
    ):
        self._t = timings
        self._th = thresholds
        self._override = timings_override
        self._positive = positive_chime if positive_chime is not None else FeedbackConfig()
        self._state = PostureState.NO_PERSON
        self._history: list[tuple[float, str]] = []  # (t, zone) samples
        self._bad_since: float | None = None
        self._good_since: float | None = None
        self._absent_since: float | None = None
        self._chime_at: float | None = None
        self._voice_at: float | None = None
        self._good_state_since: float | None = None  # start of the current GOOD episode
        self._positive_at: float | None = None  # last positive chime (cooldown timer)

    @property
    def state(self) -> PostureState:
        return self._state

    def streaks(self, now: float) -> tuple[float | None, float | None]:
        """(bad_streak_s, good_streak_s): live streak lengths for the dashboard
        hold bar. Same clocks the fire conditions compare, so the bar reaches
        100% exactly when update() fires — the button can never show
        "triggering" with no sound again. None when no streak is running."""
        bad = None if self._bad_since is None else max(0.0, now - self._bad_since)
        good = None if self._good_since is None else max(0.0, now - self._good_since)
        return bad, good

    def update(self, result: ScoringResult, now: float) -> Transition:
        if isinstance(result, Unreliable):
            return self._on_unreliable(result, now)

        self._absent_since = None  # person seen and scored
        zone = zone_of(result, self._th)
        self._history.append((now, zone))
        self._prune_history(now)

        span = now - self._history[0][0]
        smoothed_bad = span >= self._t.smoothing_window_s - _EPS and all(
            z == "bad" for _, z in self._history
        )
        smoothed_good = span >= self._t.smoothing_window_s - _EPS and all(
            z == "good" for _, z in self._history
        )
        if smoothed_bad and self._bad_since is None:
            self._bad_since = self._history[0][0]
        if (
            (smoothed_good or (self._state is PostureState.NO_PERSON and not smoothed_bad))
            and self._good_since is None
        ):
            self._good_since = self._history[0][0]
        if not smoothed_bad and self._bad_since is not None and smoothed_good:
            self._bad_since = None
        if not smoothed_good and self._good_since is not None and smoothed_bad:
            self._good_since = None

        if self._state in (PostureState.BAD_CHIME, PostureState.BAD_VOICE):
            return self._update_bad(smoothed_good, now)
        transition = self._update_idle(smoothed_bad, smoothed_good, now)
        if self._state is PostureState.GOOD and self._positive.positive_chime.enabled:
            return self._maybe_positive_chime(now)
        return transition

    # -- unreliable / absent paths -------------------------------------------------

    def _on_unreliable(self, result: Unreliable, now: float) -> Transition:
        if result.reason in _ABSENT_REASONS:
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self._t.no_person_timeout_s - _EPS:
                self._enter_no_person()
                return Transition(PostureState.NO_PERSON, Nudge.NONE)
            return Transition(self._state, Nudge.NONE)
        # LOW_VISIBILITY / DEPTH_UNREADABLE: untrustworthy frame — go silent
        # and drop the bad streak so it cannot escalate on data we do not trust.
        self._absent_since = None
        self._state = PostureState.UNRELIABLE
        self._history.clear()
        self._bad_since = None
        self._good_since = None
        self._chime_at = None
        self._voice_at = None
        self._good_state_since = None
        return Transition(PostureState.UNRELIABLE, Nudge.NONE)

    def _enter_no_person(self) -> None:
        self._state = PostureState.NO_PERSON
        self._history.clear()
        self._bad_since = None
        self._good_since = None
        self._chime_at = None
        self._voice_at = None
        self._good_state_since = None

    # -- scored paths ----------------------------------------------------------------

    def _voice_timings(self) -> tuple[float, float]:
        """Live voice timings: the runtime override when present, else config."""
        if self._override is not None:
            return self._override.voice_timings()
        return self._t.voice_delay_s, self._t.voice_repeat_cooldown_s

    def _update_idle(self, smoothed_bad: bool, smoothed_good: bool, now: float) -> Transition:
        # Bad is checked first so a slouching person who never sat well still
        # gets nudged instead of waiting forever in NO_PERSON.
        if (
            smoothed_bad
            and self._bad_since is not None
            and now - self._bad_since >= self._t.bad_sustain_s - _EPS
        ):
            seconds_bad = now - self._bad_since
            self._state = PostureState.BAD_CHIME
            self._chime_at = now
            self._voice_at = None
            self._good_state_since = None
            return Transition(PostureState.BAD_CHIME, Nudge.CHIME, seconds_bad)
        can_confirm_good = smoothed_good or (
            self._state is PostureState.NO_PERSON and not smoothed_bad
        )
        if (
            can_confirm_good
            and self._good_since is not None
            and now - self._good_since >= self._t.good_confirm_s - _EPS
        ):
            self._state = PostureState.GOOD
            return Transition(PostureState.GOOD, Nudge.NONE)
        return Transition(self._state, Nudge.NONE)

    def _update_bad(self, smoothed_good: bool, now: float) -> Transition:
        if (
            smoothed_good
            and self._good_since is not None
            and now - self._good_since >= self._t.recovery_good_s - _EPS
        ):
            self._state = PostureState.GOOD
            self._bad_since = None
            self._chime_at = None
            self._voice_at = None
            return Transition(PostureState.GOOD, Nudge.NONE)
        delay_s, repeat_s = self._voice_timings()
        if self._state is PostureState.BAD_CHIME and self._chime_at is not None:
            if now - self._chime_at >= delay_s - _EPS:
                self._state = PostureState.BAD_VOICE
                self._voice_at = now
                return Transition(PostureState.BAD_VOICE, Nudge.VOICE)
        elif (
            self._state is PostureState.BAD_VOICE
            and self._voice_at is not None
            and now - self._voice_at >= repeat_s - _EPS
        ):
            self._voice_at = now
            return Transition(PostureState.BAD_VOICE, Nudge.VOICE)
        return Transition(self._state, Nudge.NONE)

    def _maybe_positive_chime(self, now: float) -> Transition:
        """Positive reinforcement: while state is GOOD, fire the positive chime
        once per every_seconds of continuously good posture. The timer restarts
        on fire (cooldown) and on any departure from GOOD (sustained interval)."""
        if self._good_state_since is None:
            self._good_state_since = now
        every_s = self._positive.positive_chime.every_seconds
        sustained = now - self._good_state_since >= every_s - _EPS
        cooled = self._positive_at is None or now - self._positive_at >= every_s - _EPS
        if sustained and cooled:
            self._positive_at = now
            return Transition(PostureState.GOOD, Nudge.POSITIVE_CHIME)
        return Transition(PostureState.GOOD, Nudge.NONE)

    def _prune_history(self, now: float) -> None:
        cutoff = now - self._t.smoothing_window_s - _EPS
        self._history = [(t, z) for t, z in self._history if t >= cutoff]
