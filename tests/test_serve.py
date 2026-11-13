"""Tests for source selection in serve.py: the one-time startup log that names
the selected source and reason (stubbed _open_pipeline, no SDK, no hardware).
"""

from __future__ import annotations

import logging

import pytest

from posture_guard.camera import SyntheticFrameSource
from posture_guard.config import AppConfig
from posture_guard.dashboard.serve import build_source


@pytest.fixture
def log_source_selection(caplog):
    caplog.set_level(logging.INFO, logger="posture_guard.dashboard.serve")
    return caplog


def test_realsense_start_logs_active_source_with_usb(log_source_selection, monkeypatch):
    def usb3_device(width, height, fps):
        return object(), object(), "3.2"

    monkeypatch.setattr("posture_guard.camera._open_pipeline", usb3_device)
    source = build_source(AppConfig())
    try:
        assert isinstance(source, SyntheticFrameSource) is False
        assert "RealSense source active (usb 3.2)" in log_source_selection.text
    finally:
        source.close()


def test_synthetic_fallback_logs_reason(log_source_selection, monkeypatch):
    def no_device(width, height, fps):
        raise RuntimeError("No device connected")

    monkeypatch.setattr("posture_guard.camera._open_pipeline", no_device)
    source = build_source(AppConfig())

    assert isinstance(source, SyntheticFrameSource)
    text = log_source_selection.text
    assert "synthetic source (reason:" in text
    # The reason names the exhausted retry budget and the SDK's last error.
    assert "camera unavailable after 3 attempts: No device connected" in text
    # The fallback source itself carries the reason, so it reaches the
    # dashboard status too, not just the log.
    assert "No device connected" in (source.status.error or "")
    assert source.status.kind == "synthetic"
