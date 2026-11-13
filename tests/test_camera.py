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
    source = RealSenseFrameSource()
    assert source.start() is False  # returns a state, never raises
    assert not source.status.connected
    assert "No device connected" in (source.status.error or "")
    assert source.read() is None


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
            return SourceStatus(connected=False, error=None)

        def close(self) -> None:
            return None

    worker = CameraWorker(_BrokenSource())
    worker.start()
    try:
        assert wait_until(lambda: worker.status.error is not None)
        assert "device disconnected" in worker.status.error
        assert not worker.status.connected
        assert worker.read() is None
    finally:
        worker.close()
