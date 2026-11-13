"""Shared pose fixtures.

Geometry uses normalized image coordinates (x right, y down). Expected angles
are precomputed constants — not derived with the same atan2 helpers the code
uses — so the tests can catch formula regressions. Tolerance is 1 degree per
the spec's acceptance criterion.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest

from posture_guard.scoring import (
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    SimpleLandmark,
)

NUM_LANDMARKS = 33


class ExpectedAngles(NamedTuple):
    head_forward_deg: float
    trunk_lean_deg: float
    head_tilt_deg: float
    shoulder_level_deg: float


def make_pose(overrides: dict[int, SimpleLandmark] | None = None) -> list[SimpleLandmark]:
    """A good seated pose: ears directly above shoulder mid, trunk vertical,
    all landmarks fully visible."""
    pose = [SimpleLandmark(x=0.5, y=0.5) for _ in range(NUM_LANDMARKS)]
    pose[LEFT_SHOULDER] = SimpleLandmark(x=0.40, y=0.60)
    pose[RIGHT_SHOULDER] = SimpleLandmark(x=0.60, y=0.60)
    pose[LEFT_EAR] = SimpleLandmark(x=0.46, y=0.45)
    pose[RIGHT_EAR] = SimpleLandmark(x=0.54, y=0.45)
    pose[LEFT_HIP] = SimpleLandmark(x=0.44, y=0.90)
    pose[RIGHT_HIP] = SimpleLandmark(x=0.56, y=0.90)
    for index, landmark in (overrides or {}).items():
        pose[index] = landmark
    return pose


# Precomputed expected angles (atan2 constants, not computed from the code).
GOOD_ANGLES = ExpectedAngles(0.0, 0.0, 0.0, 0.0)
SLOUCHED_ANGLES = ExpectedAngles(
    head_forward_deg=18.4349488,  # atan2(0.02, 0.06)
    trunk_lean_deg=0.0,
    head_tilt_deg=0.0,
    shoulder_level_deg=0.0,
)
LEANING_ANGLES = ExpectedAngles(
    head_forward_deg=0.0,
    trunk_lean_deg=9.4623222,  # atan2(0.05, 0.30)
    head_tilt_deg=0.0,
    shoulder_level_deg=0.0,
)
TILTED_HEAD_ANGLES = ExpectedAngles(
    head_forward_deg=0.0,
    trunk_lean_deg=0.0,
    head_tilt_deg=11.3099325,  # atan2(0.004, 0.02): ears rolled vs level shoulders
    shoulder_level_deg=0.0,
)
UNEVEN_SHOULDERS_ANGLES = ExpectedAngles(
    head_forward_deg=0.0,
    trunk_lean_deg=0.0,
    head_tilt_deg=5.7105931,  # ear line parallel to the tilted shoulder line
    shoulder_level_deg=5.7105931,  # atan2(0.02, 0.20)
)


@pytest.fixture
def good_pose() -> list[SimpleLandmark]:
    return make_pose()


@pytest.fixture
def slouched_pose() -> list[SimpleLandmark]:
    """Head shifted forward (+x) 0.02 with only 0.06 of vertical support:
    head-forward ~18.4 deg, everything else level."""
    return make_pose(
        {
            LEFT_EAR: SimpleLandmark(x=0.48, y=0.54),
            RIGHT_EAR: SimpleLandmark(x=0.56, y=0.54),
        }
    )


@pytest.fixture
def leaning_pose() -> list[SimpleLandmark]:
    """Whole upper body shifted laterally: trunk lean ~9.5 deg, head centered
    over the (shifted) shoulders."""
    return make_pose(
        {
            LEFT_SHOULDER: SimpleLandmark(x=0.45, y=0.60),
            RIGHT_SHOULDER: SimpleLandmark(x=0.65, y=0.60),
            LEFT_EAR: SimpleLandmark(x=0.51, y=0.45),
            RIGHT_EAR: SimpleLandmark(x=0.59, y=0.45),
        }
    )


@pytest.fixture
def tilted_head_pose() -> list[SimpleLandmark]:
    """Ears rolled ~11.3 deg relative to level shoulders."""
    return make_pose(
        {
            LEFT_EAR: SimpleLandmark(x=0.45, y=0.46),
            RIGHT_EAR: SimpleLandmark(x=0.55, y=0.44),
        }
    )


@pytest.fixture
def uneven_shoulders_pose() -> list[SimpleLandmark]:
    """Right shoulder 0.02 lower than left: shoulder level ~5.7 deg."""
    return make_pose(
        {
            RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.62),
            LEFT_EAR: SimpleLandmark(x=0.46, y=0.46),
            RIGHT_EAR: SimpleLandmark(x=0.54, y=0.46),
        }
    )


@pytest.fixture
def cropped_pose() -> list[SimpleLandmark]:
    """Good pose but the right shoulder is barely visible."""
    return make_pose(
        {RIGHT_SHOULDER: SimpleLandmark(x=0.60, y=0.60, visibility=0.1)}
    )
