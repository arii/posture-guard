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

from posture_guard.config import ScoringConfig, StateConfig
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


_ABSENT_REASONS = (UnreliableReason.NO_POSE, UnreliableReason.DEPTH_OUT_OF_BAND)


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

    def __init__(self, timings: StateConfig, thresholds: ScoringConfig):
        self._t = timings
        self._th = thresholds
        self._state = PostureState.NO_PERSON
        self._history: list[tuple[float, str]] = []  # (t, zone) samples
        self._bad_since: float | None = None
        self._good_since: float | None = None
        self._absent_since: float | None = None
        self._chime_at: float | None = None
        self._voice_at: float | None = None

    @property
    def state(self) -> PostureState:
        return self._state

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
        if smoothed_good and self._good_since is None:
            self._good_since = self._history[0][0]
        if not smoothed_bad and self._bad_since is not None and smoothed_good:
            self._bad_since = None
        if not smoothed_good and self._good_since is not None and smoothed_bad:
            self._good_since = None

        if self._state in (PostureState.BAD_CHIME, PostureState.BAD_VOICE):
            return self._update_bad(smoothed_good, now)
        return self._update_idle(smoothed_bad, smoothed_good, now)

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
        return Transition(PostureState.UNRELIABLE, Nudge.NONE)

    def _enter_no_person(self) -> None:
        self._state = PostureState.NO_PERSON
        self._history.clear()
        self._bad_since = None
        self._good_since = None
        self._chime_at = None
        self._voice_at = None

    # -- scored paths ----------------------------------------------------------------

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
            return Transition(PostureState.BAD_CHIME, Nudge.CHIME, seconds_bad)
        if (
            smoothed_good
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
        if self._state is PostureState.BAD_CHIME and self._chime_at is not None:
            if now - self._chime_at >= self._t.voice_delay_s - _EPS:
                self._state = PostureState.BAD_VOICE
                self._voice_at = now
                return Transition(PostureState.BAD_VOICE, Nudge.VOICE)
        elif (
            self._state is PostureState.BAD_VOICE
            and self._voice_at is not None
            and now - self._voice_at >= self._t.voice_repeat_cooldown_s - _EPS
        ):
            self._voice_at = now
            return Transition(PostureState.BAD_VOICE, Nudge.VOICE)
        return Transition(self._state, Nudge.NONE)

    def _prune_history(self, now: float) -> None:
        cutoff = now - self._t.smoothing_window_s - _EPS
        self._history = [(t, z) for t, z in self._history if t >= cutoff]
