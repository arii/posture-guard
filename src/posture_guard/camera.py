"""Frame sources: a `FrameSource` interface with a synthetic implementation for
hardware-free development/tests, a lazy-importing RealSense implementation, and
a `CameraWorker` that runs any source on a background thread behind a bounded
latest-frame buffer.

Capture contract (spec §camera.py): color 640x480 BGR + depth 640x480 Z16 at
30 fps, depth aligned to color. Missing camera and USB-2 connections surface as
source *status*, never as exceptions thrown from read().
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import pyrealsense2 as rs  # noqa: F401 - type-checking only


@dataclass(frozen=True)
class ColorDepth:
    """One aligned color+depth frame pair.

    color: HxWx3 uint8 BGR. depth: HxW uint16, Z16 millimetres.
    """

    color: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.uint16]
    timestamp_s: float


@dataclass(frozen=True)
class SourceStatus:
    """Operational status of a frame source, safe to render on the dashboard."""

    connected: bool
    usb_type: str | None = None  # "2.1" | "3.x" | None when unknown
    error: str | None = None


def usb_bandwidth_warning(usb_type: str | None) -> str | None:
    """Dashboard warning for a USB-2 connection, or None when bandwidth is fine.

    USB 2 cannot carry aligned depth+color at full frame rate without dropping
    frames — librealsense's most common Linux failure mode — so it is surfaced
    as a state the dashboard can show, not an error.
    """
    if usb_type is not None and usb_type.startswith("2"):
        return f"USB {usb_type} connection - depth may be unstable; use a USB 3 port"
    return None


@runtime_checkable
class FrameSource(Protocol):
    """Newest-frame reader. read() returns None when no fresh frame exists."""

    def read(self) -> ColorDepth | None: ...

    @property
    def status(self) -> SourceStatus: ...

    def close(self) -> None: ...


class SyntheticFrameSource:
    """Deterministic frames with no camera attached.

    Emits a flat color frame and a constant-depth depth frame (default 1.4 m,
    inside the scoring band) so pose workers, dashboards and tests run anywhere.
    """

    def __init__(self, width: int = 640, height: int = 480, depth_m: float = 1.4):
        self._width = width
        self._height = height
        self._depth_mm = int(depth_m * 1000)
        self._frame_index = 0
        self._status = SourceStatus(connected=True, usb_type=None, error=None)

    def read(self) -> ColorDepth | None:
        import numpy as np

        self._frame_index += 1
        color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        color[:, :, 0] = 40  # dark blue backdrop so frames are visibly "live"
        depth = np.full((self._height, self._width), self._depth_mm, dtype=np.uint16)
        return ColorDepth(color=color, depth=depth, timestamp_s=self._frame_index / 30.0)

    @property
    def status(self) -> SourceStatus:
        return self._status

    def close(self) -> None:
        return None


def _open_pipeline(width: int, height: int, fps: int) -> tuple[object, object, str]:
    """Open a RealSense pipeline; returns (pipeline, align, usb_type).

    Module-level so tests can stub it. The pyrealsense2 import lives here,
    keeping the SDK import lazy for CI and dev machines without the camera.
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    # Align depth to color so both frames share one pixel grid.
    align = rs.align(rs.stream.color)
    usb_type = str(profile.get_device().get_info(rs.camera_info.usb_type_descriptor))
    return pipeline, align, usb_type


class RealSenseFrameSource:
    """pyrealsense2 pipeline wrapper.

    pyrealsense2 is imported lazily so machines without the SDK (CI, dev
    laptops) can still import this module for the synthetic source and tests.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self._width = width
        self._height = height
        self._fps = fps
        self._status = SourceStatus(connected=False, error="not started")
        self._pipeline: object | None = None
        self._align: object | None = None

    def start(self) -> bool:
        """Open the pipeline. Missing-camera and USB problems become status
        states (connected=False, error set) — start() never raises."""
        try:
            pipeline, align, usb_type = _open_pipeline(self._width, self._height, self._fps)
        except Exception as err:  # SDK raises RuntimeError subclass on no device
            self._status = SourceStatus(connected=False, error=f"camera unavailable: {err}")
            return False
        self._pipeline = pipeline
        self._align = align
        self._status = SourceStatus(connected=True, usb_type=usb_type, error=None)
        return True

    def read(self) -> ColorDepth | None:
        import numpy as np

        if self._pipeline is None:
            return None
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        return ColorDepth(
            color=np.asanyarray(color_frame.get_data()),
            depth=np.asanyarray(depth_frame.get_data()),
            timestamp_s=color_frame.get_timestamp() / 1000.0,
        )

    @property
    def status(self) -> SourceStatus:
        return self._status

    def close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError as err:  # pipeline already stopped
                self._status = SourceStatus(connected=False, error=str(err))
                return
            self._pipeline = None
            self._status = SourceStatus(connected=False, error=None)


class CameraWorker:
    """Runs a FrameSource on a background capture thread behind a bounded
    latest-frame buffer.

    The buffer is a single lock-protected slot: a slow consumer always reads
    the newest frame and stale frames are dropped, so pose scoring can never
    lag behind the camera. Implements the FrameSource protocol itself, so it
    drops in wherever a FrameSource is expected. Capture errors are recorded
    as status states; the loop backs off and keeps retrying.
    """

    def __init__(self, source: FrameSource):
        self._source = source
        self._latest: ColorDepth | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._capture_loop, name="camera-worker", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._source.read()
            except Exception as err:
                # Surface as a state; the dashboard shows the error banner.
                self._error = f"capture failed: {err}"
                time.sleep(0.1)
                continue
            if frame is None:
                time.sleep(0.005)  # yield; real sources block inside read()
                continue
            with self._lock:
                self._latest = frame  # single slot: older frames are dropped

    def read(self) -> ColorDepth | None:
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    @property
    def status(self) -> SourceStatus:
        if self._error is not None:
            base = self._source.status
            return SourceStatus(connected=False, usb_type=base.usb_type, error=self._error)
        return self._source.status

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._source.close()


@contextmanager
def open_frame_source(source: FrameSource) -> Iterator[FrameSource]:
    """Context manager so callers never leak a camera pipeline."""
    try:
        yield source
    finally:
        source.close()
