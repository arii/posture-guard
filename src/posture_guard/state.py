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
# Smoothing never starves: the window holds at least this many recent samples
# even when the frame rate divides the horizon awkwardly (at 24 fps a pure
# 0.3 s span never fills, which used to veto every bad streak). Anything
# older than the stall cap is stale and stays dropped.
_MIN_WINDOW_SAMPLES = 3
_STALL_CAP_S = 2.0


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
    POSITIVE_VOICE = "positive_voice"


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
        th = self._active_thresholds()
        zone = zone_of(result, th)
        self._history.append((now, zone))
        self._prune_history(now)

        zones = [zone for _, zone in self._history]
        # Fullness is sample-counted (pruning guarantees the minimum): a pure
        # time span starves whenever fps divides the horizon awkwardly, which
        # vetoed every bad streak at 24 fps.
        window_full = len(self._history) >= _MIN_WINDOW_SAMPLES
        # Majority vote, neutral abstains: a slouch hovering at a threshold
        # flickers bad/neutral (and occasionally good) frame to frame, and any
        # veto-based rule blocks the streak forever (red needle, 0% bar, no
        # sound). More bad than good sustains the bad streak; more good than
        # bad sustains good. Ties and pure-neutral windows hold the streaks.
        n_bad = zones.count("bad")
        n_good = zones.count("good")
        bad_dominant = window_full and n_bad > n_good
        good_dominant = window_full and n_good > n_bad
        if bad_dominant and self._bad_since is None:
            bad_times = [t for t, z in self._history if z == "bad"]
            self._bad_since = bad_times[0] if bad_times else self._history[0][0]
        if (
            (good_dominant or (self._state is PostureState.NO_PERSON and not bad_dominant))
            and self._good_since is None
        ):
            good_times = [t for t, z in self._history if z == "good"]
            self._good_since = good_times[0] if good_times else self._history[0][0]
        if good_dominant and self._bad_since is not None:
            self._bad_since = None
        if bad_dominant and self._good_since is not None:
            self._good_since = None

        if self._state in (PostureState.BAD_CHIME, PostureState.BAD_VOICE):
            return self._update_bad(good_dominant, now)
        transition = self._update_idle(bad_dominant, good_dominant, now)
        pos_enabled = (
            getattr(self._override, "positive_enabled", self._positive.positive_chime.enabled)
            if self._override is not None
            else self._positive.positive_chime.enabled
        )
        if self._state is PostureState.GOOD and pos_enabled:
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

    def _state_timings(self) -> tuple[float, float, float]:
        """(bad_sustain_s, good_confirm_s, positive_every_s) with runtime overrides."""
        bad_s = self._t.bad_sustain_s
        good_s = self._t.good_confirm_s
        pos_s = self._positive.positive_chime.every_seconds
        if self._override is not None:
            if getattr(self._override, "bad_sustain_s", None) is not None:
                bad_s = float(self._override.bad_sustain_s)
            if getattr(self._override, "good_confirm_s", None) is not None:
                good_s = float(self._override.good_confirm_s)
            if getattr(self._override, "positive_every_s", None) is not None:
                pos_s = float(self._override.positive_every_s)
        return bad_s, good_s, pos_s

    def _active_thresholds(self) -> ScoringConfig:
        if self._override is not None and hasattr(self._override, "scoring_config"):
            return self._override.scoring_config(self._th)
        return self._th

    def _update_idle(self, bad_dominant: bool, good_dominant: bool, now: float) -> Transition:
        bad_s, good_s, _ = self._state_timings()
        # Bad is checked first so a slouching person who never sat well still
        # gets nudged instead of waiting forever in NO_PERSON.
        if (
            bad_dominant
            and self._bad_since is not None
            and now - self._bad_since >= bad_s - _EPS
        ):
            seconds_bad = now - self._bad_since
            self._state = PostureState.BAD_CHIME
            self._chime_at = now
            self._voice_at = None
            self._good_state_since = None
            return Transition(PostureState.BAD_CHIME, Nudge.CHIME, seconds_bad)
        can_confirm_good = good_dominant or (
            self._state is PostureState.NO_PERSON and not bad_dominant
        )
        if (
            can_confirm_good
            and self._good_since is not None
            and now - self._good_since >= good_s - _EPS
        ):
            self._state = PostureState.GOOD
            return Transition(PostureState.GOOD, Nudge.NONE)
        return Transition(self._state, Nudge.NONE)

    def _update_bad(self, good_dominant: bool, now: float) -> Transition:
        if (
            good_dominant
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
        _, _, every_s = self._state_timings()
        sustained = now - self._good_state_since >= every_s - _EPS
        cooled = self._positive_at is None or now - self._positive_at >= every_s - _EPS
        if sustained and cooled:
            self._positive_at = now
            return Transition(PostureState.GOOD, Nudge.POSITIVE_CHIME)
        return Transition(PostureState.GOOD, Nudge.NONE)

    def _prune_history(self, now: float) -> None:
        cutoff = now - self._t.smoothing_window_s - _EPS
        recent = [(t, z) for t, z in self._history if t >= cutoff]
        if len(recent) < _MIN_WINDOW_SAMPLES:
            # Low frame rate (or unlucky window/fps ratio): reach back for
            # evidence instead of starving the vote; anything older than the
            # stall cap is stale and stays dropped.
            older = [
                (t, z) for t, z in self._history
                if cutoff > t >= now - _STALL_CAP_S
            ]
            recent = (older + recent)[-_MIN_WINDOW_SAMPLES:]
        self._history = recent
