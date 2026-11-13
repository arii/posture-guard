"""Pose detector selection: MediaPipe when usable, synthetic otherwise."""

from __future__ import annotations

from posture_guard import pose as pose_module
from posture_guard.pose import SyntheticPoseDetector, default_detector


def test_default_detector_degrades_on_incompatible_mediapipe(monkeypatch):
    """An incompatible mediapipe install (e.g. 1.x, which removed
    mp.solutions) raises AttributeError, not ImportError — the detector must
    degrade to the synthetic pose instead of crashing startup."""

    def boom(*args: object, **kwargs: object) -> None:
        raise AttributeError("module 'mediapipe' has no attribute 'solutions'")

    monkeypatch.setattr(pose_module, "MediaPipePoseDetector", boom)
    assert isinstance(default_detector(), SyntheticPoseDetector)
