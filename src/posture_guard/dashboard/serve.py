"""Runnable entry point for the whole system (spec: make run serves the
dashboard).

    python -m posture_guard.dashboard.serve [--config config.yaml] [--port 8000]

Runs the full capture -> pose -> scoring -> state -> dashboard/audio pipeline.
With a RealSense attached it uses the camera; otherwise it falls back to the
synthetic frame source so the dashboard is verifiable with no hardware.
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from posture_guard.audio import AudioWorker
from posture_guard.camera import (
    CameraWorker,
    FrameSource,
    RealSenseFrameSource,
    SyntheticFrameSource,
)
from posture_guard.config import AppConfig, load_config
from posture_guard.dashboard.app import create_app
from posture_guard.dashboard.hub import DashboardHub
from posture_guard.pipeline import PosturePipeline

logger = logging.getLogger(__name__)


def build_source(config: AppConfig) -> FrameSource:
    """RealSense when it starts cleanly, synthetic fallback otherwise.

    Either way the selection is logged exactly once with its reason, so a
    silent fallback to the synthetic source is impossible.
    """
    realsense = RealSenseFrameSource(config.camera.width, config.camera.height, config.camera.fps)
    if realsense.start():
        logger.info("RealSense source active (usb %s)", realsense.status.usb_type or "unknown")
        return CameraWorker(realsense)
    reason = realsense.status.error or "camera unavailable"
    logger.info("synthetic source (reason: %s)", reason)
    print(f"synthetic source (reason: {reason})", file=sys.stderr)
    return SyntheticFrameSource(config.camera.width, config.camera.height, reason=reason)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Posture Guard dashboard")
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    hub = DashboardHub()

    audio = AudioWorker(config.audio)
    audio.start()

    pipeline = PosturePipeline(build_source(config), config, hub=hub, audio=audio)
    pipeline.start()

    try:
        uvicorn.run(create_app(hub, audio), host=args.host, port=args.port, log_level="info")
    finally:
        pipeline.close()
        audio.close()


if __name__ == "__main__":
    main()
