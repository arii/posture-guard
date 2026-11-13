"""Runtime feedback settings (spec §runtime feedback settings).

One mutable settings object shared by the pipeline (mode filter), the state
machine (voice timings) and the dashboard API (read/apply). Seeded from
AppConfig so config.yaml defaults win on startup; edits are session-only by
design — config.yaml stays the startup source of truth.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum

from posture_guard.config import AppConfig
from posture_guard.state import Nudge

MAX_PROMPT_CHARS = 120
MIN_TIMING_S = 1.0


class FeedbackMode(Enum):
    CHIME_AND_VOICE = "chime_and_voice"
    CHIME_ONLY = "chime_only"
    SILENT = "silent"


@dataclass
class RuntimeFeedback:
    """Thread-safe runtime feedback settings.

    Locks are coarse but correct: writers are rare (dashboard form posts) and
    readers are per-frame, so an uncontended lock beats lock-free trickery.
    """

    mode: FeedbackMode = FeedbackMode.CHIME_AND_VOICE
    voice_delay_s: float = 10.0
    voice_repeat_cooldown_s: float = 30.0
    chime_text: str = "fix your posture"
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @classmethod
    def from_config(cls, config: AppConfig) -> RuntimeFeedback:
        """Seed from config.yaml defaults (spec: pre-configured defaults win on startup)."""
        return cls(
            mode=FeedbackMode.CHIME_AND_VOICE,
            voice_delay_s=config.state.voice_delay_s,
            voice_repeat_cooldown_s=config.state.voice_repeat_cooldown_s,
            chime_text=config.audio.voice_text,
        )

    # -- reads (hot path: every frame) -------------------------------------------

    def allows(self, nudge: Nudge) -> bool:
        """Whether this nudge may play under the current mode.

        SILENT drops both channels; CHIME_ONLY drops the voice. Test endpoints
        bypass this filter by design (they are hardware-verification commands).
        """
        with self._lock:
            mode = self.mode
        if mode is FeedbackMode.SILENT:
            return False
        return not (mode is FeedbackMode.CHIME_ONLY and nudge is Nudge.VOICE)

    def voice_timings(self) -> tuple[float, float]:
        """(voice_delay_s, voice_repeat_cooldown_s) as one consistent read."""
        with self._lock:
            return self.voice_delay_s, self.voice_repeat_cooldown_s

    def prompt_text(self) -> str:
        with self._lock:
            return self.chime_text

    # -- writes (dashboard API) ------------------------------------------------------

    def apply(self, **changes: object) -> dict[str, object]:
        """Apply validated changes and return the resulting settings payload.

        Callers validate first (see validate_changes); apply itself trusts its
        input so the dashboard endpoint can compose both. The payload is built
        after the lock is released — to_payload takes the same non-reentrant
        lock, so calling it under the lock would deadlock.
        """
        with self._lock:
            if "mode" in changes:
                mode = changes["mode"]
                self.mode = mode if isinstance(mode, FeedbackMode) else FeedbackMode(mode)  # type: ignore[arg-type]
            if "voice_delay_s" in changes:
                self.voice_delay_s = float(changes["voice_delay_s"])  # type: ignore[arg-type]
            if "voice_repeat_cooldown_s" in changes:
                self.voice_repeat_cooldown_s = float(changes["voice_repeat_cooldown_s"])  # type: ignore[arg-type]
            if "chime_text" in changes:
                self.chime_text = str(changes["chime_text"])  # type: ignore[arg-type]
        return self.to_payload()

    def to_payload(self) -> dict[str, object]:
        """Settings for the WS snapshot / API echo; mode is the enum value."""
        with self._lock:
            return {
                "mode": self.mode.value,
                "voice_delay_s": self.voice_delay_s,
                "voice_repeat_cooldown_s": self.voice_repeat_cooldown_s,
                "chime_text": self.chime_text,
                # UI renders this so the presenter knows edits are not persisted
                "session_only": True,
            }


def validate_changes(raw: dict[str, object]) -> tuple[dict[str, object], str | None]:
    """Validate a settings PATCH body.

    Returns (validated_changes, error). error is None exactly when the body
    contains at least one known key and every present key passes validation —
    partial application is impossible because callers only apply on error=None.
    """
    if not raw:
        return {}, "empty settings body"
    changes: dict[str, object] = {}
    if "mode" in raw:
        try:
            changes["mode"] = FeedbackMode(raw["mode"])
        except ValueError:
            valid = ", ".join(m.value for m in FeedbackMode)
            return {}, f"unknown mode {raw['mode']!r} (valid: {valid})"
    if "voice_delay_s" in raw:
        value, err = _timing(raw["voice_delay_s"])
        if err:
            return {}, f"voice_delay_s: {err}"
        changes["voice_delay_s"] = value
    if "voice_repeat_cooldown_s" in raw:
        value, err = _timing(raw["voice_repeat_cooldown_s"])
        if err:
            return {}, f"voice_repeat_cooldown_s: {err}"
        changes["voice_repeat_cooldown_s"] = value
    if "chime_text" in raw:
        text = raw["chime_text"]
        if not isinstance(text, str):
            return {}, "chime_text: expected a string"
        if not text.strip():
            return {}, "chime_text: must not be empty"
        if len(text) > MAX_PROMPT_CHARS:
            return {}, f"chime_text: over {MAX_PROMPT_CHARS} characters"
        changes["chime_text"] = text.strip()
    unknown = set(raw) - set(changes)
    if unknown:
        return {}, f"unknown setting(s): {', '.join(sorted(unknown))}"
    return changes, None


def _timing(value: object) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "expected a number (seconds)"
    seconds = float(value)
    if seconds < MIN_TIMING_S:
        return None, f"must be at least {MIN_TIMING_S} s"
    return seconds, None
