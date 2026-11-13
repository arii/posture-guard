"""Pipeline wiring (spec: one process, four threads over bounded buffers):
capture -> pose -> scoring -> state -> dashboard/audio.

process_once() is the whole pipeline for one frame — pure control flow with
the state machine clocked on frame timestamps, so tests drive synthetic frames
through the real scoring and state code with no sleeps and no hardware.
start() runs it on a daemon thread against any FrameSource. Everything is
injectable: detector, renderer, audio sink and hub are constructor parameters.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from typing import Protocol

from numpy.typing import NDArray

from posture_guard.camera import ColorDepth, FrameSource, SourceStatus, usb_bandwidth_warning
from posture_guard.config import AppConfig, ScoringConfig
from posture_guard.dashboard.hub import DashboardHub, MetricsSnapshot, ScoringThresholds
from posture_guard.feedback import RuntimeFeedback
from posture_guard.pose import PoseDetection, PoseDetector, default_detector, tracked_landmarks
from posture_guard.scoring import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    Landmark,
    ScoredPosture,
    ScoringResult,
    Unreliable,
    UnreliableReason,
    score_pose,
)
from posture_guard.state import Nudge, PostureState, PostureStateMachine, Transition


class NudgeSink(Protocol):
    """The audio surface the pipeline needs — AudioWorker satisfies it.

    status is optional: stubs without it simply report audio as not ready on
    the dashboard payload.
    """

    def enqueue(self, nudge: Nudge, *, force: bool = False) -> None: ...


FrameRenderer = Callable[
    [NDArray, Sequence[Landmark] | None, str, str, str], bytes
]


def default_frame_renderer(
    color: NDArray,
    landmarks: Sequence[Landmark] | None,
    state_text: str,
    depth_text: str,
    reason_text: str = "",
) -> bytes:
    """Skeleton overlay + banner + JPEG encode; OpenCV imported lazily."""
    from posture_guard.dashboard.annotate import draw_overlay, encode_jpeg

    return encode_jpeg(
        draw_overlay(color, landmarks, state_text, depth_text, reason_text)
    )


def low_visibility_warning(
    landmarks: Sequence[Landmark] | None, cfg: ScoringConfig
) -> str:
    """Name the landmark groups below the visibility floor, with the geometry
    fix. At close range (e.g. 0.7 m) the D435i vertical FOV covers ~0.5 m, so
    seated hips can never be in frame — trunk lean is then unavailable and the
    fix is to move back, not to zoom or tilt."""
    if landmarks is None:
        return "person not fully in frame - sit facing the camera, 1.0-1.5 m away"
    floor = cfg.visibility_floor

    def group_missing(indices: tuple[int, int]) -> bool:
        # Either landmark of a pair below the floor means the group is not
        # reliably tracked — name it.
        return any(landmarks[i].visibility < floor for i in indices)

    missing: list[str] = []
    if group_missing((LEFT_EAR, RIGHT_EAR)):
        missing.append("head")
    if group_missing((LEFT_SHOULDER, RIGHT_SHOULDER)):
        missing.append("shoulders")
    if group_missing((LEFT_HIP, RIGHT_HIP)):
        missing.append("hips")
    if not missing:
        return "person not fully in frame - sit facing the camera, 1.0-1.5 m away"
    named = " and ".join(missing)
    if "hips" in missing:
        return f"{named} not visible - trunk lean unavailable, move back to 1.0-1.5 m"
    return f"{named} not visible - sit facing the camera, 1.0-1.5 m away"


def placement_warnings(
    result: ScoringResult,
    depth_m: float | None,
    camera: SourceStatus,
    cfg: ScoringConfig,
    landmarks: Sequence[Landmark] | None = None,
) -> tuple[str, ...]:
    """Camera-placement guidance carried on the WS payload (spec: the placement
    guide doubles as a live checklist — each failure carries its own warning)."""
    warnings: list[str] = []
    if not camera.connected:
        warnings.append(f"camera unavailable: {camera.error or 'not started'}")
    else:
        usb = usb_bandwidth_warning(camera.usb_type)
        if usb:
            warnings.append(usb)
    if isinstance(result, Unreliable):
        if result.reason is UnreliableReason.DEPTH_OUT_OF_BAND and depth_m is not None:
            band = f"{cfg.depth_min_m}-{cfg.depth_max_m} m"
            if depth_m > cfg.depth_max_m:
                warnings.append(f"person too far - {depth_m:.1f} m (band {band})")
            else:
                warnings.append(f"person too close - {depth_m:.1f} m (band {band})")
        elif result.reason is UnreliableReason.LOW_VISIBILITY:
            warnings.append(low_visibility_warning(landmarks, cfg))
        elif result.reason is UnreliableReason.DEPTH_UNREADABLE:
            warnings.append("depth unreadable - check the camera view")
        elif result.reason is UnreliableReason.NO_POSE:
            warnings.append("waiting for a person - sit fully in frame")
    return tuple(warnings)


def _hips_below_floor(
    landmarks: Sequence[Landmark] | None, cfg: ScoringConfig
) -> bool:
    """True when either hip landmark sits below the visibility floor."""
    if landmarks is None:
        return False
    return any(
        landmarks[i].visibility < cfg.visibility_floor
        for i in (LEFT_HIP, RIGHT_HIP)
    )


def _audio_fields(audio: NudgeSink | None) -> dict[str, object]:
    """AudioWorker.status fields for the snapshot; stubs without a status
    property report the audio worker as not ready."""
    status = getattr(audio, "status", None)
    if status is None:
        return {
            "audio_ready": False,
            "audio_sink": None,
            "audio_tts_backend": None,
            "audio_error": None,
        }
    return {
        "audio_ready": status.ready,
        "audio_sink": status.sink,
        "audio_tts_backend": status.tts_backend,
        "audio_error": status.error,
    }


class PosturePipeline:
    """Drives one frame through pose, scoring, state, audio and dashboard."""

    def __init__(
        self,
        source: FrameSource,
        config: AppConfig,
        *,
        hub: DashboardHub | None = None,
        audio: NudgeSink | None = None,
        detector: PoseDetector | None = None,
        renderer: FrameRenderer | None = None,
        feedback: RuntimeFeedback | None = None,
    ):
        self._source = source
        self._config = config
        self._hub = hub
        self._audio = audio
        self._feedback = feedback
        self._detector: PoseDetector = detector if detector is not None else default_detector()
        self._renderer = renderer or default_frame_renderer
        self._state_machine = PostureStateMachine(
            config.state, config.scoring, timings_override=feedback
        )
        self._thresholds = ScoringThresholds(
            head_forward_deg=config.scoring.head_forward_deg,
            trunk_lean_deg=config.scoring.trunk_lean_deg,
            head_tilt_deg=config.scoring.head_tilt_deg,
            shoulder_level_deg=config.scoring.shoulder_level_deg,
            hysteresis_margin_deg=config.scoring.hysteresis_margin_deg,
            visibility_floor=config.scoring.visibility_floor,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = 0.0
        self._last_timestamp: float | None = None
        self._nudge_seq = 0
        self._nudge_event: dict[str, object] | None = None

    @property
    def state(self) -> PostureState:
        return self._state_machine.state

    def process_once(self, frame: ColorDepth) -> Transition:
        """Run the full frame path once. Clock is the frame timestamp, keeping
        state-machine timelines reproducible in tests."""
        detection = self._detector(frame)
        result = score_pose(detection.landmarks, detection.depth_m, self._config.scoring)
        transition = self._state_machine.update(result, frame.timestamp_s)
        if transition.nudge is not Nudge.NONE:
            # Mode filter is the single enforcement point (spec): the machine
            # keeps sole emission authority; silent/chime-only drop the audio
            # here while state logic advances identically.
            allowed = self._feedback is None or self._feedback.allows(transition.nudge)
            if allowed and self._audio is not None:
                self._audio.enqueue(transition.nudge)
            self._nudge_event = self._emit_nudge_event(transition.nudge, allowed)
        self._update_fps(frame.timestamp_s)
        if self._hub is not None:
            self._publish(detection, result, transition.state, frame)
        return transition

    # -- dashboard publication -----------------------------------------------------

    def _emit_nudge_event(self, nudge: Nudge, allowed: bool) -> dict[str, object]:
        """Build the visible reminder event. Voice events carry the prompt text
        so the banner can read 'Reminder: <text>'; sequence numbers come from
        the hub when wired so snapshot and test-endpoint events never collide."""
        text = (
            self._feedback.prompt_text()
            if nudge is Nudge.VOICE and self._feedback is not None
            else None
        )
        if self._hub is not None:
            return self._hub.next_nudge_event(
                nudge.value, muted=not allowed, text=text
            )
        self._nudge_seq += 1
        event: dict[str, object] = {
            "kind": nudge.value,
            "seq": self._nudge_seq,
            "muted": not allowed,  # dropped by mode: still a visible event
        }
        if text:
            event["text"] = text
        return event

    def _publish(
        self,
        detection: PoseDetection,
        result: ScoringResult,
        state: PostureState,
        frame: ColorDepth,
    ) -> None:
        angles: ScoredPosture | None = result if isinstance(result, ScoredPosture) else None
        trunk_deg = angles.trunk_lean_deg if angles else None
        trunk_note: str | None = None
        if (
            angles is not None
            and not self._config.scoring.trunk_lean_required
            and _hips_below_floor(detection.landmarks, self._config.scoring)
        ):
            # Trunk lean optional + hips invisible: the measured angle is
            # garbage, so report it as unavailable instead of a number.
            trunk_deg = None
            trunk_note = "\u2014 trunk lean unavailable (hips not visible)"
        camera = self._source.status
        depth_m = detection.depth_m
        snapshot = MetricsSnapshot(
            state=state.value,
            fps=self._fps,
            depth_m=depth_m,
            head_forward_deg=angles.head_forward_deg if angles else None,
            trunk_lean_deg=trunk_deg,
            head_tilt_deg=angles.head_tilt_deg if angles else None,
            shoulder_level_deg=angles.shoulder_level_deg if angles else None,
            camera_connected=camera.connected,
            camera_usb_type=camera.usb_type,
            camera_error=camera.error,
            camera_source=camera.kind,
            warnings=placement_warnings(
                result, depth_m, camera, self._config.scoring, detection.landmarks
            ),
            landmarks=(
                tracked_landmarks(detection.landmarks)
                if detection.landmarks is not None
                else ()
            ),
            **_audio_fields(self._audio),
            thresholds=self._thresholds,
            unreliable_detail=(
                result.detail if isinstance(result, Unreliable) and result.detail else None
            ),
            trunk_note=trunk_note,
            feedback=(
                self._feedback.to_payload()
                if self._feedback is not None
                else None
            ),
            nudge_event=self._nudge_event,
        )
        self._hub.publish_metrics(snapshot)
        state_text = state.value.replace("_", " ")
        depth_text = f"depth {depth_m:.1f} m" if depth_m is not None else "depth -"
        reason_text = result.detail if isinstance(result, Unreliable) else ""
        self._hub.publish_video(
            self._renderer(frame.color, detection.landmarks, state_text, depth_text, reason_text)
        )

    # -- fps and threaded run ---------------------------------------------------------

    def _update_fps(self, timestamp_s: float) -> None:
        if self._last_timestamp is not None:
            delta = timestamp_s - self._last_timestamp
            if delta > 0:
                instant = 1.0 / delta
                self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
        self._last_timestamp = timestamp_s

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="posture-pipeline", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._source.read()
            if frame is None:
                time.sleep(0.005)  # no fresh frame; the hub keeps only the newest
                continue
            self.process_once(frame)

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._source.close()
