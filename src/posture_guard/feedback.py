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

from posture_guard.config import AppConfig, ScoringConfig
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
    positive_voice_text: str | None = None
    positive_every_s: float | None = None
    bad_sustain_s: float | None = None
    good_confirm_s: float | None = None
    head_forward_deg: float | None = None
    trunk_lean_deg: float | None = None
    head_tilt_deg: float | None = None
    shoulder_level_deg: float | None = None
    hysteresis_margin_deg: float | None = None
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @classmethod
    def from_config(cls, config: AppConfig) -> RuntimeFeedback:
        """Seed from config.yaml defaults (spec: pre-configured defaults win on startup)."""
        pos_txt = getattr(config.audio, "positive_voice_text", "Good posture, keep it up!")
        return cls(
            mode=FeedbackMode.CHIME_AND_VOICE,
            voice_delay_s=config.state.voice_delay_s,
            voice_repeat_cooldown_s=config.state.voice_repeat_cooldown_s,
            chime_text=config.audio.voice_text,
            positive_voice_text=pos_txt,
            positive_every_s=config.feedback.positive_chime.every_seconds,
            bad_sustain_s=config.state.bad_sustain_s,
            good_confirm_s=config.state.good_confirm_s,
            head_forward_deg=config.scoring.head_forward_deg,
            trunk_lean_deg=config.scoring.trunk_lean_deg,
            head_tilt_deg=config.scoring.head_tilt_deg,
            shoulder_level_deg=config.scoring.shoulder_level_deg,
            hysteresis_margin_deg=config.scoring.hysteresis_margin_deg,
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
        voice_nudges = (Nudge.VOICE, Nudge.POSITIVE_VOICE)
        return not (mode is FeedbackMode.CHIME_ONLY and nudge in voice_nudges)

    def voice_timings(self) -> tuple[float, float]:
        """(voice_delay_s, voice_repeat_cooldown_s) as one consistent read."""
        with self._lock:
            return self.voice_delay_s, self.voice_repeat_cooldown_s

    def prompt_text(self) -> str:
        with self._lock:
            return self.chime_text

    def positive_text(self) -> str:
        with self._lock:
            return self.positive_voice_text or "Good posture, keep it up!"

    def scoring_config(self, base: ScoringConfig) -> ScoringConfig:
        with self._lock:
            hf = (
                self.head_forward_deg
                if self.head_forward_deg is not None
                else base.head_forward_deg
            )
            tl = self.trunk_lean_deg if self.trunk_lean_deg is not None else base.trunk_lean_deg
            ht = self.head_tilt_deg if self.head_tilt_deg is not None else base.head_tilt_deg
            sl = (
                self.shoulder_level_deg
                if self.shoulder_level_deg is not None
                else base.shoulder_level_deg
            )
            hm = (
                self.hysteresis_margin_deg
                if self.hysteresis_margin_deg is not None
                else base.hysteresis_margin_deg
            )
            return ScoringConfig(
                head_forward_deg=hf,
                trunk_lean_deg=tl,
                head_tilt_deg=ht,
                shoulder_level_deg=sl,
                hysteresis_margin_deg=hm,
                visibility_floor=base.visibility_floor,
                depth_min_m=base.depth_min_m,
                depth_max_m=base.depth_max_m,
                trunk_lean_required=base.trunk_lean_required,
            )

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
                raw_mode = changes["mode"]
                self.mode = (
                    raw_mode
                    if isinstance(raw_mode, FeedbackMode)
                    else FeedbackMode(raw_mode)  # type: ignore[arg-type]
                )
            if "voice_delay_s" in changes:
                self.voice_delay_s = float(changes["voice_delay_s"])  # type: ignore[arg-type]
            if "voice_repeat_cooldown_s" in changes:
                self.voice_repeat_cooldown_s = float(changes["voice_repeat_cooldown_s"])  # type: ignore[arg-type]
            if "chime_text" in changes:
                self.chime_text = str(changes["chime_text"])  # type: ignore[arg-type]
            if "positive_voice_text" in changes:
                self.positive_voice_text = str(changes["positive_voice_text"])  # type: ignore[arg-type]
            if "positive_every_s" in changes:
                self.positive_every_s = float(changes["positive_every_s"])  # type: ignore[arg-type]
            if "bad_sustain_s" in changes:
                self.bad_sustain_s = float(changes["bad_sustain_s"])  # type: ignore[arg-type]
            if "good_confirm_s" in changes:
                self.good_confirm_s = float(changes["good_confirm_s"])  # type: ignore[arg-type]
            for deg_key in (
                "head_forward_deg",
                "trunk_lean_deg",
                "head_tilt_deg",
                "shoulder_level_deg",
                "hysteresis_margin_deg",
            ):
                if deg_key in changes:
                    setattr(self, deg_key, float(changes[deg_key]))  # type: ignore[arg-type]
        return self.to_payload()

    def to_payload(self) -> dict[str, object]:
        """Settings for the WS snapshot / API echo; mode is the enum value."""
        with self._lock:
            payload: dict[str, object] = {
                "mode": self.mode.value,
                "voice_delay_s": self.voice_delay_s,
                "voice_repeat_cooldown_s": self.voice_repeat_cooldown_s,
                "chime_text": self.chime_text,
                "session_only": True,
            }
            extra_keys = (
                "positive_voice_text",
                "positive_every_s",
                "bad_sustain_s",
                "good_confirm_s",
                "head_forward_deg",
                "trunk_lean_deg",
                "head_tilt_deg",
                "shoulder_level_deg",
                "hysteresis_margin_deg",
            )
            for k in extra_keys:
                val = getattr(self, k, None)
                if val is not None:
                    payload[k] = val
            return payload


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
    if "positive_voice_text" in raw:
        text = raw["positive_voice_text"]
        if not isinstance(text, str):
            return {}, "positive_voice_text: expected a string"
        if not text.strip():
            return {}, "positive_voice_text: must not be empty"
        if len(text) > MAX_PROMPT_CHARS:
            return {}, f"positive_voice_text: over {MAX_PROMPT_CHARS} characters"
        changes["positive_voice_text"] = text.strip()
    for timing_key, min_limit in (
        ("positive_every_s", 1.0),
        ("bad_sustain_s", 0.2),
        ("good_confirm_s", 0.2),
    ):
        if timing_key in raw:
            value, err = _timing(raw[timing_key], min_s=min_limit)
            if err:
                return {}, f"{timing_key}: {err}"
            changes[timing_key] = value
    for deg_key in (
        "head_forward_deg",
        "trunk_lean_deg",
        "head_tilt_deg",
        "shoulder_level_deg",
        "hysteresis_margin_deg",
    ):
        if deg_key in raw:
            val = raw[deg_key]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or float(val) <= 0.0:
                return {}, f"{deg_key}: expected a positive number of degrees"
            changes[deg_key] = float(val)
    unknown = set(raw) - set(changes)
    if unknown:
        return {}, f"unknown setting(s): {', '.join(sorted(unknown))}"
    return changes, None


def _timing(value: object, min_s: float = MIN_TIMING_S) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "expected a number (seconds)"
    seconds = float(value)
    if seconds < min_s:
        return None, f"must be at least {min_s} s"
    return seconds, None
