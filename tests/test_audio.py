"""Audio worker tests.

A stubbed command runner stands in for paplay, pactl and the TTS binaries, so
queue serialization, dedupe and sink targeting are verified with no audio
hardware — per the spec's acceptance criterion for audio.py.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from posture_guard.audio import AudioWorker, CommandResult
from posture_guard.config import AudioConfig
from posture_guard.state import Nudge


def wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> bool:
    """Poll a condition without sleeps baked into the assertions."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class FakeRunner:
    """Records every external command; blocks the first paplay until released."""

    def __init__(self, pactl_sinks: str | None = None, fail_paplay: bool = False):
        self.calls: list[list[str]] = []
        self.pactl_sinks = pactl_sinks
        self.fail_paplay = fail_paplay
        self.first_play_started = threading.Event()
        self.release_first_play = threading.Event()

    def __call__(self, cmd: list[str]) -> CommandResult:
        self.calls.append(list(cmd))
        if cmd[0] == "pactl":
            return CommandResult(returncode=0, stdout=self.pactl_sinks or "")
        if cmd[0] == "paplay":
            if self.fail_paplay:
                return CommandResult(returncode=1)
            if not self.first_play_started.is_set():
                self.first_play_started.set()
                self.release_first_play.wait(timeout=5)
        return CommandResult(returncode=0)

    def play_calls(self) -> list[list[str]]:
        return [cmd for cmd in self.calls if cmd[0] == "paplay"]

    def render_calls(self, backend: str) -> list[list[str]]:
        # cmd[0] is the resolved binary path (e.g. /usr/bin/pico2wave).
        return [cmd for cmd in self.calls if cmd[0].endswith(backend)]


def make_which(available: set[str]) -> Callable[[str], str | None]:
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in available else None

    return which


PACTL_SINKS = (
    "alsa_output.pci-0000_00_1f.3.analog-stereo\tmodule-alsa-card.c\tsink\n"
    "alsa_output.usb-C-Media_Electronics_Box-00.analog-stereo\tmodule-alsa-card.c\tsink\n"
)


def make_worker(
    tmp_path,
    runner: FakeRunner,
    available: set[str] = frozenset({"pico2wave"}),
    config: AudioConfig | None = None,
) -> AudioWorker:
    return AudioWorker(
        config or AudioConfig(),
        sound_dir=tmp_path / "sounds",
        which=make_which(available),
        runner=runner,
    )


def test_startup_pre_renders_chime_and_voice_prompt(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    status = worker.start()
    try:
        assert (tmp_path / "sounds" / "chime.wav").stat().st_size > 0
        assert status.tts_backend == "pico2wave"
        assert status.ready
        assert len(runner.render_calls("pico2wave")) == 1
    finally:
        worker.close()


def test_tts_falls_back_to_espeak_ng(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner, available={"espeak-ng"})
    status = worker.start()
    try:
        assert status.tts_backend == "espeak-ng"
        assert len(runner.render_calls("espeak-ng")) == 1
        assert runner.render_calls("pico2wave") == []
    finally:
        worker.close()


def test_no_tts_backend_leaves_chime_playable(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner, available=set())
    status = worker.start()
    try:
        assert status.tts_backend is None
        assert status.ready  # chime is always synthesized
        assert "TTS" in (status.error or "")
        worker.enqueue(Nudge.CHIME)
        assert wait_until(lambda: len(runner.play_calls()) == 1)
        worker.enqueue(Nudge.VOICE)  # nothing rendered -> silent no-op
        assert len(runner.play_calls()) == 1
    finally:
        runner.release_first_play.set()
        worker.close()


def test_queue_serializes_playback_never_overlapping(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    worker.start()
    try:
        worker.enqueue(Nudge.CHIME)
        assert runner.first_play_started.wait(timeout=5)
        # Second nudge queued while the first is mid-playback: recorded but
        # the worker must still be inside the first paplay call.
        worker.enqueue(Nudge.VOICE)
        assert len(runner.play_calls()) == 1
        runner.release_first_play.set()
        assert wait_until(lambda: len(runner.play_calls()) == 2)
        # Order is FIFO: chime first, then voice.
        assert runner.play_calls()[0][1].endswith("chime.wav")
        assert runner.play_calls()[1][1].endswith("voice.wav")
    finally:
        worker.close()


def test_enqueue_returns_immediately_while_playback_blocks(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    worker.start()
    try:
        worker.enqueue(Nudge.CHIME)
        assert runner.first_play_started.wait(timeout=5)
        start = time.monotonic()
        worker.enqueue(Nudge.VOICE)
        assert time.monotonic() - start < 1.0  # never blocked on the sink
    finally:
        runner.release_first_play.set()
        worker.close()


def test_duplicate_pending_nudges_are_deduplicated(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    worker.start()
    try:
        worker.enqueue(Nudge.CHIME)
        assert runner.first_play_started.wait(timeout=5)
        # Two duplicates while the chime is queued/in-flight: both dropped.
        worker.enqueue(Nudge.CHIME)
        worker.enqueue(Nudge.CHIME)
        runner.release_first_play.set()
        assert wait_until(lambda: len(runner.play_calls()) == 1)
        worker._queue.join()
        assert len(runner.play_calls()) == 1
        # Once no chime is pending, a new event plays again.
        worker.enqueue(Nudge.CHIME)
        assert wait_until(lambda: len(runner.play_calls()) == 2)
    finally:
        worker.close()


def test_playback_targets_configured_sink(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner, config=AudioConfig(sink="alsa_usb_2"))
    worker.start()
    try:
        assert worker.status.sink == "alsa_usb_2"
        worker.enqueue(Nudge.CHIME)
        assert wait_until(lambda: len(runner.play_calls()) == 1)
        cmd = runner.play_calls()[0]
        assert cmd[1:3] == ["--sink", "alsa_usb_2"]
    finally:
        worker.close()


def test_default_sink_discovered_from_pactl_usb_listing(tmp_path):
    runner = FakeRunner(pactl_sinks=PACTL_SINKS)
    worker = make_worker(tmp_path, runner, available={"pico2wave", "pactl"})
    status = worker.start()
    try:
        assert status.sink == "alsa_output.usb-C-Media_Electronics_Box-00.analog-stereo"
    finally:
        worker.close()


def test_paplay_failure_is_surfaced_as_status_not_exception(tmp_path):
    runner = FakeRunner(fail_paplay=True)
    worker = make_worker(tmp_path, runner)
    worker.start()
    try:
        worker.enqueue(Nudge.CHIME)
        assert wait_until(lambda: worker.status.error is not None)
        assert "paplay" in worker.status.error
    finally:
        worker.close()


def test_test_nudge_forces_chime_then_voice_past_dedupe(tmp_path):
    runner = FakeRunner()
    worker = make_worker(tmp_path, runner)
    worker.start()
    try:
        runner.release_first_play.set()  # never block: both plays must complete
        worker.test_nudge()
        assert wait_until(lambda: len(runner.play_calls()) == 2)
        assert runner.play_calls()[0][1].endswith("chime.wav")
        assert runner.play_calls()[1][1].endswith("voice.wav")
    finally:
        runner.release_first_play.set()
        worker.close()
