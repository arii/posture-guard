"""RuntimeFeedback unit tests: seeding, mode filter, live timing reads, and
all-or-nothing settings validation (spec §runtime feedback settings)."""

from __future__ import annotations

import threading

from posture_guard.config import AppConfig, AudioConfig, StateConfig
from posture_guard.feedback import (
    MAX_PROMPT_CHARS,
    FeedbackMode,
    RuntimeFeedback,
    validate_changes,
)
from posture_guard.state import Nudge


def test_from_config_seeds_from_config_defaults() -> None:
    config = AppConfig(
        state=StateConfig(voice_delay_s=5.0, voice_repeat_cooldown_s=17.0),
        audio=AudioConfig(voice_text="sit up"),
    )

    feedback = RuntimeFeedback.from_config(config)

    assert feedback.voice_timings() == (5.0, 17.0)
    assert feedback.prompt_text() == "sit up"
    assert feedback.mode is FeedbackMode.CHIME_AND_VOICE


class TestAllows:
    def test_default_mode_allows_everything(self) -> None:
        feedback = RuntimeFeedback()
        assert feedback.allows(Nudge.CHIME)
        assert feedback.allows(Nudge.VOICE)

    def test_chime_only_drops_voice_only(self) -> None:
        feedback = RuntimeFeedback(mode=FeedbackMode.CHIME_ONLY)
        assert feedback.allows(Nudge.CHIME)
        assert not feedback.allows(Nudge.VOICE)

    def test_silent_drops_positive_chime(self) -> None:
        feedback = RuntimeFeedback(mode=FeedbackMode.SILENT)
        assert not feedback.allows(Nudge.POSITIVE_CHIME)

    def test_chime_only_allows_positive_chime(self) -> None:
        # The positive chime is a chime, not a voice prompt: chime-only keeps it.
        feedback = RuntimeFeedback(mode=FeedbackMode.CHIME_ONLY)
        assert feedback.allows(Nudge.POSITIVE_CHIME)

    def test_silent_drops_both(self) -> None:
        feedback = RuntimeFeedback(mode=FeedbackMode.SILENT)
        assert not feedback.allows(Nudge.CHIME)
        assert not feedback.allows(Nudge.VOICE)

    def test_mode_takes_effect_on_the_next_check(self) -> None:
        # No restart: apply() at runtime changes the very next decision.
        feedback = RuntimeFeedback()
        assert feedback.allows(Nudge.VOICE)
        feedback.apply(mode="silent")
        assert not feedback.allows(Nudge.VOICE)


def test_apply_and_payload_round_trip() -> None:
    feedback = RuntimeFeedback()

    payload = feedback.apply(
        mode="chime_only", voice_delay_s=3.0, voice_repeat_cooldown_s=9.0,
        chime_text="straighten up",
    )

    assert payload == {
        "mode": "chime_only",
        "voice_delay_s": 3.0,
        "voice_repeat_cooldown_s": 9.0,
        "chime_text": "straighten up",
        "session_only": True,
    }
    assert feedback.voice_timings() == (3.0, 9.0)
    assert feedback.prompt_text() == "straighten up"


class TestValidateChanges:
    def test_unknown_mode_rejected(self) -> None:
        changes, error = validate_changes({"mode": "vibrate"})
        assert error is not None and "vibrate" in error
        assert changes == {}

    def test_timings_below_one_second_rejected(self) -> None:
        for key in ("voice_delay_s", "voice_repeat_cooldown_s"):
            changes, error = validate_changes({key: 0.5})
            assert error is not None and key in error
            assert changes == {}

    def test_non_numeric_timing_rejected(self) -> None:
        _, error = validate_changes({"voice_delay_s": "soon"})
        assert error is not None

    def test_blank_and_oversized_text_rejected(self) -> None:
        _, error = validate_changes({"chime_text": "   "})
        assert error is not None
        _, error = validate_changes({"chime_text": "x" * (MAX_PROMPT_CHARS + 1)})
        assert error is not None

    def test_unknown_keys_rejected(self) -> None:
        _, error = validate_changes({"volume": 11})
        assert error is not None and "volume" in error

    def test_empty_body_rejected(self) -> None:
        _, error = validate_changes({})
        assert error is not None

    def test_valid_body_returns_changes_and_strips_text(self) -> None:
        changes, error = validate_changes(
            {"mode": "silent", "chime_text": "  sit up  ", "voice_delay_s": 2}
        )
        assert error is None
        assert changes == {
            "mode": FeedbackMode.SILENT,
            "chime_text": "sit up",
            "voice_delay_s": 2.0,
        }


def test_concurrent_reads_and_writes_stay_consistent() -> None:
    """A frame-rate reader and a dashboard writer race without corruption."""
    feedback = RuntimeFeedback()
    stop = threading.Event()
    errors: list[str] = []

    def reader() -> None:
        while not stop.is_set():
            delay, repeat = feedback.voice_timings()
            if delay < 1.0 or repeat < 1.0:
                errors.append(f"torn timing read: {delay}, {repeat}")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    for delay in range(1, 50):
        _, error = validate_changes({"voice_delay_s": delay})
        assert error is None
        feedback.apply(voice_delay_s=delay, voice_repeat_cooldown_s=delay)
    stop.set()
    thread.join(timeout=2.0)
    assert errors == []
