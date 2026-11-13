"""Audio worker: chime-then-voice nudge playback (spec §audio.py).

Pre-renders every prompt to WAV at startup — the chime is synthesized tone
pairs (no dependency), speech via pico2wave with espeak-ng as fallback — so
playback is a single paplay call on the configured sink. One daemon thread
owns the queue: prompts serialize, never overlap, and enqueue() never blocks
the camera or state-machine threads. External commands all flow through an
injected runner, so tests stub the subprocess layer entirely.
"""

from __future__ import annotations

import logging
import math
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from posture_guard.config import AudioConfig
from posture_guard.state import Nudge

SAMPLE_RATE = 44100

logger = logging.getLogger(__name__)

Runner = Callable[[Sequence[str]], "CommandResult"]
Which = Callable[[str], str | None]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""


@dataclass(frozen=True)
class AudioStatus:
    """Operational status, safe to render on the dashboard."""

    ready: bool  # at least one prompt is playable
    sink: str | None  # resolved sink name, None = paplay default
    tts_backend: str | None  # "pico2wave" | "espeak-ng" | None
    error: str | None = None


def _render_chime(path: Path) -> None:
    """Two-tone chime (660 -> 990 Hz) with exponential decay, 16-bit mono WAV."""
    samples: list[float] = []
    for freq, seconds in ((660.0, 0.15), (990.0, 0.25)):
        total = int(SAMPLE_RATE * seconds)
        for i in range(total):
            decay = math.exp(-3.0 * i / total)
            samples.append(0.6 * decay * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(b"".join(struct.pack("<h", int(s * 32767)) for s in samples))


class AudioWorker:
    """Serialized nudge playback on a dedicated daemon thread.

    Duplicate pending nudges are dropped (the state machine already enforces
    the voice cooldown); test_nudge() bypasses dedupe so the dashboard button
    always fires the full chime + voice sequence for hardware verification.
    """

    def __init__(
        self,
        config: AudioConfig,
        *,
        sound_dir: Path | None = None,
        paplay_cmd: str = "paplay",
        pactl_cmd: str = "pactl",
        which: Which = shutil.which,
        runner: Runner | None = None,
    ):
        self._config = config
        self._paplay = paplay_cmd
        self._pactl = pactl_cmd
        self._which = which
        self._runner: Runner = runner or self._run_subprocess
        self._sound_dir = Path(sound_dir or tempfile.mkdtemp(prefix="posture-guard-audio-"))
        self._queue: queue.Queue[tuple[Nudge, Path] | None] = queue.Queue()
        self._pending: set[Nudge] = set()
        self._pending_lock = threading.Lock()
        self._sounds: dict[Nudge, Path] = {}
        self._sounds_lock = threading.Lock()  # guards _sounds swaps vs enqueue reads
        self._playing_path: Path | None = None  # WAV the playback thread is reading
        self._voice_seq = 0  # names re-rendered voice WAVs uniquely
        self._sink: str | None = None
        self._tts_backend: str | None = None
        self._error: str | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> AudioStatus:
        """Pre-render prompts, resolve the sink, and start the playback thread."""
        self._render_sounds()
        self._sink = self._config.sink or self._discover_usb_sink()
        self._thread = threading.Thread(
            target=self._playback_loop, name="audio-worker", daemon=True
        )
        self._thread.start()
        return self.status

    def close(self) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def status(self) -> AudioStatus:
        return AudioStatus(
            ready=bool(self._sounds),
            sink=self._sink,
            tts_backend=self._tts_backend,
            error=self._error,
        )

    # -- public API (caller-side, never blocks on playback) -------------------------

    def enqueue(self, nudge: Nudge, *, force: bool = False) -> None:
        """Queue a nudge for playback. Returns immediately; a duplicate of a
        nudge that is queued or currently playing is dropped."""
        with self._sounds_lock:
            path = self._sounds.get(nudge)
        if path is None:
            return  # nothing rendered for this nudge (e.g. no TTS backend)
        with self._pending_lock:
            if not force and nudge in self._pending:
                return
            self._pending.add(nudge)
        self._queue.put((nudge, path))

    def test_nudge(self) -> None:
        """Dashboard hook: force-play chime then voice, bypassing dedupe."""
        self.play_chime()
        self.play_voice()

    def play_chime(self) -> None:
        """Dashboard hook: force-play the chime only, bypassing dedupe and mode."""
        self.enqueue(Nudge.CHIME, force=True)

    def play_voice(self) -> None:
        """Dashboard hook: force-play the spoken prompt only, bypassing dedupe
        and mode (spec: test endpoints are hardware-verification commands)."""
        self.enqueue(Nudge.VOICE, force=True)

    # -- internals -------------------------------------------------------------------

    def render_prompt(self, text: str) -> str | None:
        """Re-render the spoken prompt for edited text (spec §settings API).

        Renders to a fresh file, then swaps it in under the sounds lock: a
        queued prompt keeps its old path and finishes cleanly, the next enqueue
        picks up the new wording. Returns the backend used, or None when no TTS
        backend rendered the prompt (worker error field then names the cause).
        """
        with self._sounds_lock:
            self._voice_seq += 1
            path = self._sound_dir / f"voice-{self._voice_seq}.wav"
        backend = self._render_voice(text, path)
        if backend is None:
            path.unlink(missing_ok=True)
            return None
        self._tts_backend = backend
        self._error = None  # a successful re-render supersedes a stale backend error
        with self._sounds_lock:
            previous = self._sounds.get(Nudge.VOICE)
            self._sounds[Nudge.VOICE] = path
        self._discard_stale(previous)
        return backend

    def _discard_stale(self, path: Path | None) -> None:
        """Delete a replaced voice WAV once no playback can still be reading it.
        A file being played right now is unlinked by the playback loop instead."""
        if path is None or path == self._playing_path:
            return
        path.unlink(missing_ok=True)

    def _render_sounds(self) -> None:
        self._sound_dir.mkdir(parents=True, exist_ok=True)
        chime_path = self._sound_dir / "chime.wav"
        _render_chime(chime_path)
        self._sounds[Nudge.CHIME] = chime_path
        voice_path = self._sound_dir / "voice.wav"
        backend = self._render_voice(self._config.voice_text, voice_path)
        if backend is not None:
            self._sounds[Nudge.VOICE] = voice_path
            self._tts_backend = backend

    def _render_voice(self, text: str, path: Path) -> str | None:
        """Render the spoken prompt to path; pico2wave first, espeak-ng fallback.
        Renders only — callers install the result into _sounds under the lock."""
        for backend in ("pico2wave", "espeak-ng"):
            binary = self._which(backend)
            if binary is None:
                continue
            result = self._runner([binary, "-w", str(path), text])
            if result.returncode == 0:
                return backend
        self._error = "no TTS backend available (install pico2wave or espeak-ng)"
        return None

    def _discover_usb_sink(self) -> str | None:
        """First USB audio sink from pactl, or None to use the paplay default."""
        if self._which(self._pactl) is None:
            return None
        result = self._runner([self._pactl, "list", "short", "sinks"])
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if fields and "usb" in fields[0].lower():
                return fields[0]
        return None

    def _playback_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            nudge, path = item
            try:
                self._playing_path = path
                self._play(path)
            finally:
                self._playing_path = None
                self._queue.task_done()
                with self._pending_lock:
                    self._pending.discard(nudge)

    def _play(self, path: Path) -> None:
        """Play on the configured sink; on a non-zero exit, retry once with the
        paplay default sink before surfacing both attempts as status error."""
        sinks: tuple[str | None, ...] = (
            (self._sink, None) if self._sink is not None else (None,)
        )
        failures: list[str] = []
        for sink in sinks:
            cmd = [self._paplay]
            if sink is not None:
                cmd += ["--sink", sink]
            cmd.append(str(path))
            try:
                result = self._runner(cmd)
            except OSError as exc:
                # A missing or unlaunchable player (FileNotFoundError etc.) is
                # surfaced as status, never raised out of the worker thread —
                # a dead sink must not kill nudging.
                self._error = f"{self._paplay} unavailable: {exc}"
                return
            if result.returncode == 0:
                self._error = None  # a successful play supersedes a stale error
                logger.info("played %s on sink %s", path.name, sink or "default")
                return
            failures.append(f"sink {sink or 'default'}: exited {result.returncode}")
            logger.warning(
                "paplay failed for %s on %s (exit %d)",
                path.name,
                sink or "default",
                result.returncode,
            )
        self._error = f"{self._paplay} failed for {path.name}: " + "; ".join(failures)

    @staticmethod
    def _run_subprocess(cmd: Sequence[str]) -> CommandResult:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return CommandResult(returncode=completed.returncode, stdout=completed.stdout)
