"""Frame sources: a `FrameSource` interface with a synthetic implementation for
hardware-free development/tests, and a lazy-importing RealSense implementation.

Capture contract (spec §camera.py): color 640x480 BGR + depth 640x480 Z16 at
30 fps, depth aligned to color. Missing camera and USB-2 connections surface as
source *status*, never as exceptions thrown from read().
"""

from __future__ import annotations

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

    def start(self) -> None:
        """Open the pipeline. Raises on failure — callers surface the error."""
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        profile = self._pipeline.start(config)
        # Align depth to color so both frames share one pixel grid.
        self._align = rs.align(rs.stream.color)
        usb = profile.get_device().get_info(rs.info.usb_type_descriptor)
        self._status = SourceStatus(connected=True, usb_type=str(usb), error=None)

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


@contextmanager
def open_frame_source(source: FrameSource) -> Iterator[FrameSource]:
    """Context manager so callers never leak a camera pipeline."""
    try:
        yield source
    finally:
        source.close()
