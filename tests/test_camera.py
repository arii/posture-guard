"""Tests for frame sources: everything here runs with no camera attached."""

import numpy as np
import pytest

from posture_guard.camera import (
    FrameSource,
    RealSenseFrameSource,
    SyntheticFrameSource,
    open_frame_source,
)


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
