"""Pipeline integration: the whole frame path on synthetic data, no camera.

Timelines are driven by frame timestamps (30 fps grid), so the chime/voice
schedule below is the real state machine's schedule with zero sleeps in the
deterministic tests; only the threaded smoke test touches the wall clock.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pytest

from conftest import make_pose
from posture_guard.camera import ColorDepth, SourceStatus, SyntheticFrameSource
from posture_guard.config import AppConfig, ScoringConfig
from posture_guard.dashboard.hub import DashboardHub
from posture_guard.pipeline import PosturePipeline, placement_warnings
from posture_guard.pose import PoseDetection, SyntheticPoseDetector, center_depth_m
from posture_guard.scoring import (
    RIGHT_SHOULDER,
    SimpleLandmark,
    Unreliable,
    UnreliableReason,
    score_pose,
)
from posture_guard.state import Nudge


class RecordingAudio:
    """NudgeSink double recording the enqueue order."""

    def __init__(self) -> None:
        self.enqueued: list[Nudge] = []

    def enqueue(self, nudge: Nudge, *, force: bool = False) -> None:
        self.enqueued.append(nudge)


def fake_jpeg(
    color: np.ndarray, landmarks: object, state_text: str, depth_text: str
) -> bytes:
    return f"{state_text}|{depth_text}".encode()


class MutableDetector:
    """Synthetic detector whose pose the timeline can switch mid-run."""

    def __init__(self, pose: Sequence[SimpleLandmark] | None, depth_m: float = 1.4):
        self.pose = pose
        self.depth_m = depth_m

    def __call__(self, frame: ColorDepth) -> PoseDetection:
        return PoseDetection(self.pose, self.depth_m)


def make_frame(index: int, depth_m: float = 1.4) -> ColorDepth:
    color = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), int(depth_m * 1000), dtype=np.uint16)
    return ColorDepth(color=color, depth=depth, timestamp_s=index / 30.0)


def feed(pipeline: PosturePipeline, detector: MutableDetector, seconds: float, index: int) -> int:
    """Feed 30 fps frames for a wall of simulated seconds; returns next index."""
    for _ in range(int(seconds * 30)):
        pipeline.process_once(make_frame(index, detector.depth_m))
        index += 1
    return index


def build_pipeline(
    detector: MutableDetector,
) -> tuple[PosturePipeline, DashboardHub, RecordingAudio]:
    hub = DashboardHub()
    audio = RecordingAudio()
    pipeline = PosturePipeline(
        SyntheticFrameSource(),
        AppConfig(),
        hub=hub,
        audio=audio,
        detector=detector,
        renderer=fake_jpeg,
    )
    return pipeline, hub, audio


def test_good_pose_reaches_good_state_with_zeroed_metrics() -> None:
    detector = MutableDetector(make_pose())
    pipeline, hub, _ = build_pipeline(detector)

    feed(pipeline, detector, 2.0, 0)

    snapshot, _ = hub.latest_metrics()
    assert snapshot.state == "good"
    assert snapshot.depth_m == pytest.approx(1.4)
    assert snapshot.head_forward_deg == pytest.approx(0.0, abs=1.0)
    assert snapshot.trunk_lean_deg == pytest.approx(0.0, abs=1.0)
    assert len(snapshot.landmarks) == 6
    assert snapshot.warnings == ()
    assert snapshot.camera_connected is True


def test_full_nudge_timeline_chime_then_voice_then_recovery() -> None:
    detector = MutableDetector(make_pose())
    pipeline, hub, audio = build_pipeline(detector)

    index = feed(pipeline, detector, 2.0, 0)  # settle into GOOD
    assert hub.latest_metrics()[0].state == "good"

    detector.pose = make_pose(
        {
            7: SimpleLandmark(x=0.48, y=0.54),  # slouch: head-forward ~18.4 deg
            8: SimpleLandmark(x=0.56, y=0.54),
        }
    )
    index = feed(pipeline, detector, 2.5, index)  # bad sustained > 2 s
    assert hub.latest_metrics()[0].state == "bad_chime"
    assert audio.enqueued == [Nudge.CHIME]

    index = feed(pipeline, detector, 10.0, index)  # voice fires 10 s after the chime
    assert hub.latest_metrics()[0].state == "bad_voice"
    assert audio.enqueued == [Nudge.CHIME, Nudge.VOICE]

    index = feed(pipeline, detector, 1.0, index)  # still inside the 30 s voice cooldown
    assert audio.enqueued == [Nudge.CHIME, Nudge.VOICE]

    detector.pose = make_pose()
    feed(pipeline, detector, 4.0, index)  # recovery needs 3 s of good
    assert hub.latest_metrics()[0].state == "good"
    assert audio.enqueued == [Nudge.CHIME, Nudge.VOICE]  # silence on recovery


def test_out_of_band_depth_surfaces_as_warning_and_no_person() -> None:
    detector = MutableDetector(make_pose(), depth_m=2.9)  # beyond 2.5 m band
    pipeline, hub, _ = build_pipeline(detector)

    feed(pipeline, detector, 3.0, 0)

    snapshot, _ = hub.latest_metrics()
    assert snapshot.state == "no_person"
    assert snapshot.head_forward_deg is None
    assert snapshot.warnings == ("person too far - 2.9 m (band 0.6-2.5 m)",)


def test_cropped_person_surfaces_low_visibility_warning() -> None:
    cropped = make_pose({RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.60, visibility=0.1)})
    detector = MutableDetector(cropped)
    pipeline, hub, _ = build_pipeline(detector)

    feed(pipeline, detector, 1.0, 0)

    snapshot, _ = hub.latest_metrics()
    assert snapshot.state == "unreliable"
    assert snapshot.warnings == (
        "person not fully in frame - keep shoulders and hips visible",
    )


def test_fps_converges_to_frame_rate() -> None:
    detector = MutableDetector(make_pose())
    pipeline, hub, _ = build_pipeline(detector)

    feed(pipeline, detector, 2.0, 0)

    snapshot, _ = hub.latest_metrics()
    assert snapshot.fps == pytest.approx(30.0, rel=0.01)


def test_threaded_pipeline_runs_against_synthetic_source() -> None:
    hub = DashboardHub()
    audio = RecordingAudio()
    pipeline = PosturePipeline(
        SyntheticFrameSource(),
        AppConfig(),
        hub=hub,
        audio=audio,
        detector=SyntheticPoseDetector(),
        renderer=fake_jpeg,
    )
    pipeline.start()
    try:
        deadline = time.monotonic() + 5.0
        while hub.latest_metrics() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        snapshot, _ = hub.latest_metrics()
        assert snapshot is not None
        assert snapshot.state in ("no_person", "good")
        assert hub.latest_video() is not None
    finally:
        pipeline.close()


# -- placement_warnings unit coverage -------------------------------------------


def test_usb2_connection_surfaces_bandwidth_warning() -> None:
    camera = SourceStatus(connected=True, usb_type="2.1")
    score = score_pose(make_pose(), 1.4, ScoringConfig())

    warnings = placement_warnings(score, 1.4, camera, ScoringConfig())

    assert warnings == ("USB 2.1 connection - depth may be unstable; use a USB 3 port",)


def test_unreadable_depth_surfaces_warning() -> None:
    camera = SourceStatus(connected=True)
    result = Unreliable(UnreliableReason.DEPTH_UNREADABLE)

    warnings = placement_warnings(result, None, camera, ScoringConfig())

    assert warnings == ("depth unreadable - check the camera view",)


def test_disconnected_camera_surfaces_error() -> None:
    camera = SourceStatus(connected=False, error="camera unavailable: boom")
    score = score_pose(make_pose(), 1.4, ScoringConfig())

    warnings = placement_warnings(score, 1.4, camera, ScoringConfig())

    assert warnings == ("camera unavailable: camera unavailable: boom",)


def test_center_depth_median_ignores_invalid_values() -> None:
    depth = np.full((480, 640), 1400, dtype=np.uint16)
    depth[200:280, 240:400] = 0  # invalid zeros inside the center box

    assert center_depth_m(depth) == pytest.approx(1.4)
