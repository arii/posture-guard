"""Tests for frame sources: everything here runs with no camera attached."""

import time

import numpy as np
import pytest

from posture_guard.camera import (
    CameraWorker,
    ColorDepth,
    FrameSource,
    RealSenseFrameSource,
    SourceStatus,
    SyntheticFrameSource,
    open_frame_source,
    usb_bandwidth_warning,
)


def wait_until(predicate, timeout_s: float = 5.0) -> bool:
    """Poll a condition without sleeps baked into the assertions."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_synthetic_source_satisfies_frame_source_protocol():
    source = SyntheticFrameSource()
    assert isinstance(source, FrameSource)


def test_synthetic_source_emits_aligned_color_and_depth():
    source = SyntheticFrameSource(width=320, height=240)
    frame = source.read()
    assert frame is not None
    assert frame.color.shape == (240, 320, 3)
    assert frame.color.dtype == np.uint8
    assert frame.depth.shape == (240, 320)
    assert frame.depth.dtype == np.uint16


def test_synthetic_source_depth_is_inside_scoring_band():
    source = SyntheticFrameSource(depth_m=1.4)
    frame = source.read()
    assert frame is not None
    assert frame.depth[0, 0] == pytest.approx(1400)  # millimetres, Z16


def test_synthetic_source_timestamps_advance_at_30fps():
    source = SyntheticFrameSource()
    first = source.read()
    second = source.read()
    assert first is not None and second is not None
    assert second.timestamp_s == pytest.approx(first.timestamp_s + 1 / 30.0)


def test_synthetic_source_status_reports_connected():
    assert SyntheticFrameSource().status.connected


def test_synthetic_source_status_carries_fallback_reason():
    """A synthetic source built with a reason reports it as status error, so
    the dashboard shows why the real camera was not used."""
    source = SyntheticFrameSource(reason="camera unavailable after 3 attempts: No device connected")
    assert source.status.kind == "synthetic"
    assert source.status.connected
    assert "No device connected" in (source.status.error or "")


def test_open_frame_source_closes_on_exit():
    source = SyntheticFrameSource()
    with open_frame_source(source) as opened:
        assert opened.read() is not None
    # SyntheticFrameSource.close() is a no-op; the contract under test is that
    # exit does not raise and the source remains usable-by-contract.


def test_realsense_source_reads_none_before_start():
    source = RealSenseFrameSource()
    assert source.read() is None
    assert not source.status.connected


# -- missing-camera and USB states (no SDK import needed: _open_pipeline is stubbed) --


def test_missing_camera_surfaces_as_state_not_exception(monkeypatch):
    def no_device(width, height, fps):
        raise RuntimeError("No device connected")

    monkeypatch.setattr("posture_guard.camera._open_pipeline", no_device)
    source = RealSenseFrameSource(open_backoff_s=0.0)
    assert source.start() is False  # returns a state, never raises
    assert not source.status.connected
    assert "No device connected" in (source.status.error or "")
    assert source.read() is None


def test_failed_open_names_attempt_count_and_reason(monkeypatch):
    """The failed-open reason string is readable and names the last error."""

    def no_device(width, height, fps):
        raise RuntimeError("No device connected")

    monkeypatch.setattr("posture_guard.camera._open_pipeline", no_device)
    source = RealSenseFrameSource(open_backoff_s=0.0)
    assert source.start() is False
    assert source.status.error == (
        "camera unavailable after 3 attempts: No device connected"
    )
    assert source.status.kind == "realsense"


def test_open_retries_bounded_then_succeeds(monkeypatch):
    """A flaky open (transient failure) recovers within the attempt budget."""
    attempts = {"count": 0}

    def flaky_device(width, height, fps):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("USB transfer error")
        return object(), object(), "3.2"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", flaky_device)
    source = RealSenseFrameSource(open_attempts=3, open_backoff_s=0.0)
    assert source.start() is True
    assert attempts["count"] == 3  # two failures, then success on attempt 3
    assert source.status.connected
    assert source.status.usb_type == "3.2"
    assert source.status.error is None


def test_open_gives_up_after_attempt_budget(monkeypatch):
    attempts = {"count": 0}

    def always_fails(width, height, fps):
        attempts["count"] += 1
        raise RuntimeError("No device connected")

    monkeypatch.setattr("posture_guard.camera._open_pipeline", always_fails)
    source = RealSenseFrameSource(open_attempts=3, open_backoff_s=0.0)
    assert source.start() is False
    assert attempts["count"] == 3  # bounded: no endless open loop


def test_usb2_connection_is_reported_with_bandwidth_warning(monkeypatch):
    def usb2_device(width, height, fps):
        return object(), object(), "2.1"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", usb2_device)
    source = RealSenseFrameSource()
    assert source.start() is True
    assert source.status.connected
    assert source.status.usb_type == "2.1"
    warning = usb_bandwidth_warning(source.status.usb_type)
    assert warning is not None
    assert "USB 2.1" in warning


def test_usb3_connection_has_no_bandwidth_warning(monkeypatch):
    def usb3_device(width, height, fps):
        return object(), object(), "3.2"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", usb3_device)
    source = RealSenseFrameSource()
    assert source.start() is True
    assert usb_bandwidth_warning(source.status.usb_type) is None


def test_usb_bandwidth_warning_handles_unknown_type():
    assert usb_bandwidth_warning(None) is None


# -- reopen on wait_for_frames timeout (no SDK import needed: stubbed) --


class _HangingPipeline:
    """Fake pipeline whose wait_for_frames always times out (RuntimeError)."""

    def __init__(self) -> None:
        self.stopped = False
        self.wait_calls = 0

    def wait_for_frames(self, timeout_ms: int) -> object:
        self.wait_calls += 1
        raise RuntimeError(f"Frame didn't arrived within {timeout_ms}")

    def stop(self) -> None:
        self.stopped = True


class _StubAlign:
    def process(self, frames: object) -> object:  # pragma: no cover - unused here
        raise AssertionError("aligned frames are unreachable while frames time out")


class _WorkingAlign:
    """Fake align: hands back one preloaded color+depth pair per process()."""

    def __init__(self) -> None:
        self.color = np.zeros((2, 2, 3), dtype=np.uint8)
        self.depth = np.zeros((2, 2), dtype=np.uint16)

    def process(self, frames: object) -> object:
        return self

    def get_color_frame(self) -> object:
        return _FakeFrame(self.color, timestamp_ms=1000.0)

    def get_depth_frame(self) -> object:
        return _FakeFrame(self.depth)


class _FakeFrame:
    def __init__(self, data, timestamp_ms: float = 0.0):
        self._data = data
        self._timestamp_ms = timestamp_ms

    def get_timestamp(self) -> float:
        return self._timestamp_ms

    def get_data(self):
        return self._data


class _DeliveringPipeline:
    """Fake pipeline that returns a frameset (never times out)."""

    def __init__(self) -> None:
        self.wait_calls = 0

    def wait_for_frames(self, timeout_ms: int) -> object:
        self.wait_calls += 1
        return object()

    def stop(self) -> None: ...


def test_frame_timeout_sets_readable_error_and_reopens(monkeypatch):
    """Repeated wait_for_frames timeouts become a status error and a reopen,
    never a permanent stall inside read()."""
    opens = {"count": 0}
    pipelines: list[_HangingPipeline] = []

    def hanging_device(width, height, fps):
        opens["count"] += 1
        pipeline = _HangingPipeline()
        pipelines.append(pipeline)
        return pipeline, _StubAlign(), "3.2"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", hanging_device)
    source = RealSenseFrameSource(frame_timeout_ms=50)
    assert source.start() is True
    assert source.read() is None  # open ok, then the first wait times out
    assert not source.status.connected
    assert "frame timeout" in (source.status.error or "")
    assert pipelines[0].stopped  # the stalled pipeline was closed, not leaked

    assert source.read() is None  # next read retries the open, then times out
    assert opens["count"] == 2
    assert pipelines[1] is not pipelines[0]  # a fresh pipeline was opened


def test_frame_timeout_recovers_when_frames_return(monkeypatch):
    """After a timeout, a successful reopen serves frames again and clears
    the error."""
    opens = {"count": 0}

    def recovering_device(width, height, fps):
        opens["count"] += 1
        if opens["count"] == 1:
            return _HangingPipeline(), _StubAlign(), "3.2"
        return _DeliveringPipeline(), _WorkingAlign(), "3.2"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", recovering_device)
    source = RealSenseFrameSource(frame_timeout_ms=50)
    assert source.start() is True
    assert source.read() is None  # first open succeeds, then the wait times out
    assert "frame timeout" in (source.status.error or "")

    frame = source.read()  # second open delivers frames again
    assert frame is not None
    assert source.status.connected
    assert source.status.error is None
    assert opens["count"] == 2


# -- CameraWorker: bounded latest-frame buffer over any FrameSource --


class _ScriptedSource:
    """Emits a fixed frame sequence, then None — for buffer-boundary tests."""

    def __init__(self, frames: list[ColorDepth]):
        self._frames = list(frames)
        self._status = SourceStatus(connected=True)
        self.closed = False

    def read(self) -> ColorDepth | None:
        if self._frames:
            return self._frames.pop(0)
        return None

    @property
    def status(self) -> SourceStatus:
        return self._status

    def close(self) -> None:
        self.closed = True


def _frame(timestamp_s: float) -> ColorDepth:
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint16)
    return ColorDepth(color=color, depth=depth, timestamp_s=timestamp_s)


def test_camera_worker_satisfies_frame_source_protocol():
    assert isinstance(CameraWorker(SyntheticFrameSource()), FrameSource)


def test_camera_worker_drops_stale_frames_and_serves_latest():
    source = _ScriptedSource([_frame(1.0), _frame(2.0), _frame(3.0), _frame(4.0)])
    worker = CameraWorker(source)
    worker.start()
    try:
        # Single-slot buffer: by the time a frame is available, the newest
        # scripted frame (or a later one) is what read() serves.
        assert wait_until(lambda: (f := worker.read()) is not None and f.timestamp_s >= 4.0)
    finally:
        worker.close()
    assert source.closed


def test_camera_worker_read_consumes_the_slot():
    source = _ScriptedSource([_frame(1.0)])
    worker = CameraWorker(source)
    worker.start()
    try:
        assert wait_until(lambda: worker.read() is not None)
    finally:
        worker.close()
    # Nothing new was produced; the slot was consumed and stays empty.
    assert worker.read() is None


def test_camera_worker_delegates_status_from_source():
    source = _ScriptedSource([_frame(1.0)])
    worker = CameraWorker(source)
    worker.start()
    try:
        assert wait_until(lambda: worker.read() is not None)
        assert worker.status.connected
        assert worker.status.error is None
    finally:
        worker.close()


def test_camera_worker_capture_error_surfaces_as_state():
    class _BrokenSource:
        def read(self):
            raise RuntimeError("device disconnected")

        @property
        def status(self) -> SourceStatus:
            return SourceStatus(connected=False, error=None, kind="realsense")

        def close(self) -> None:
            return None

    worker = CameraWorker(_BrokenSource())
    worker.start()
    try:
        assert wait_until(lambda: worker.status.error is not None)
        assert "device disconnected" in worker.status.error
        assert not worker.status.connected
        assert worker.status.kind == "realsense"  # source kind survives the error
        assert worker.read() is None
    finally:
        worker.close()


def test_source_kinds_identify_the_active_implementation():
    """Each source reports its kind, so the dashboard can never show a bare
    status without naming what is active."""
    assert SyntheticFrameSource().status.kind == "synthetic"
    assert RealSenseFrameSource().status.kind == "realsense"
    assert CameraWorker(SyntheticFrameSource()).status.kind == "synthetic"


def test_camera_worker_auto_starts_on_read():
    """CameraWorker implements FrameSource protocol autonomously without
    requiring callers to invoke start() explicitly."""
    source = _ScriptedSource([_frame(1.0)])
    worker = CameraWorker(source)
    try:
        assert wait_until(lambda: worker.read() is not None)
    finally:
        worker.close()
