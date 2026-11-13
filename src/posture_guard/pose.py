"""Pose bridge: color+depth frame in, landmarks + readable depth out (spec
section: pose worker).

MediaPipe Pose runs on the newest color frame; depth is sampled over a central
neighborhood and reduced with the scoring module's median helper, so a person
at unusable range is rejected before scoring ever runs. The MediaPipe import
is lazy — CI and dev machines without the hardware extra run the synthetic
detector instead, which is what makes the no-camera pipeline runnable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from posture_guard.scoring import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SimpleLandmark,
    valid_depth_median,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from posture_guard.camera import ColorDepth


@dataclass(frozen=True)
class PoseDetection:
    """Landmarks (normalized MediaPipe order) plus median depth in meters.

    landmarks is None when no pose was found; depth_m is None when the depth
    frame has no valid samples. Either alone makes the frame unscoreable.
    """

    landmarks: Sequence[SimpleLandmark] | None
    depth_m: float | None


PoseDetector = Callable[["ColorDepth"], PoseDetection]


def center_depth_m(
    depth: npt.NDArray[np.uint16], box_fraction: float = 0.25
) -> float | None:
    """Median depth (m) of a central neighborhood; invalid values rejected.

    The center box assumes a seated, centered subject — good enough to gate
    scoring without a per-landmark depth lookup.
    """
    import numpy as np

    height, width = depth.shape
    box_h = max(1, int(height * box_fraction))
    box_w = max(1, int(width * box_fraction))
    y0 = (height - box_h) // 2
    x0 = (width - box_w) // 2
    region = depth[y0 : y0 + box_h, x0 : x0 + box_w].astype(np.float64) * 0.001
    return valid_depth_median(region.ravel().tolist())


def _good_pose() -> list[SimpleLandmark]:
    """A level, upright seated pose: ears over shoulder-mid, trunk vertical."""
    pose = [SimpleLandmark(x=0.5, y=0.5) for _ in range(33)]
    pose[LEFT_SHOULDER] = SimpleLandmark(x=0.40, y=0.60)
    pose[RIGHT_SHOULDER] = SimpleLandmark(x=0.60, y=0.60)
    pose[LEFT_EAR] = SimpleLandmark(x=0.46, y=0.45)
    pose[RIGHT_EAR] = SimpleLandmark(x=0.54, y=0.45)
    pose[LEFT_HIP] = SimpleLandmark(x=0.44, y=0.90)
    pose[RIGHT_HIP] = SimpleLandmark(x=0.56, y=0.90)
    return pose


class SyntheticPoseDetector:
    """Fixed good pose at the frame's measured depth — the no-hardware path."""

    def __init__(self, pose: Sequence[SimpleLandmark] | None = None):
        self._pose = _good_pose() if pose is None else list(pose)

    def __call__(self, frame: ColorDepth) -> PoseDetection:
        return PoseDetection(self._pose, center_depth_m(frame.depth))


class MediaPipePoseDetector:
    """MediaPipe Pose over the color frame; SDK imported lazily."""

    def __init__(self, model_complexity: int = 1):
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._pose = mp.solutions.pose.Pose(model_complexity=model_complexity)

    def __call__(self, frame: ColorDepth) -> PoseDetection:
        rgb = self._cv2.cvtColor(frame.color, self._cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        depth_m = center_depth_m(frame.depth)
        if result.pose_landmarks is None:
            return PoseDetection(None, depth_m)
        landmarks = [
            SimpleLandmark(x=lm.x, y=lm.y, visibility=lm.visibility)
            for lm in result.pose_landmarks.landmark
        ]
        return PoseDetection(landmarks, depth_m)


def default_detector() -> PoseDetector:
    """MediaPipe when installed, the synthetic pose otherwise."""
    try:
        return MediaPipePoseDetector()
    except ImportError:
        return SyntheticPoseDetector()


# Tracked landmark indices, in the order the dashboard payload reports them.
TRACKED_INDICES = (
    LEFT_EAR,
    RIGHT_EAR,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)


def tracked_landmarks(landmarks: Sequence[SimpleLandmark]) -> tuple[tuple[float, float], ...]:
    """Normalized (x, y) of the ears, shoulders and hips for the overlay."""
    return tuple(
        (landmarks[index].x, landmarks[index].y) for index in TRACKED_INDICES
    )
