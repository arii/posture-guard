"""Hardware-extra smoke check: asserts the real pyrealsense2 and mediapipe
packages expose the attributes camera.py and pose.py rely on.

Not named test_*.py on purpose: it requires the `[hardware]` extra, so it is
run only by the dedicated CI job (and locally with that extra installed) —
plain `pytest` on a machine without camera SDKs must stay green.

Exits nonzero with a clear message when a required attribute is missing, so
API drift can never hide behind stubbed tests.
"""

import sys

failures: list[str] = []

try:
    import pyrealsense2 as rs
except ImportError as err:
    print(f"FAIL: pyrealsense2 import failed ({err}); is the [hardware] extra installed?")
    sys.exit(1)

# camera.py usage (see _open_pipeline / RealSenseFrameSource.read)
for attr in (
    "pipeline",
    "config",
    "align",
    "stream",
    "format",
    "camera_info",
):
    if not hasattr(rs, attr):
        failures.append(f"pyrealsense2 has no attribute {attr!r}")
if hasattr(rs, "camera_info"):
    if not hasattr(rs.camera_info, "usb_type_descriptor"):
        failures.append("rs.camera_info has no 'usb_type_descriptor' member")
    else:
        print(f"usb_type_descriptor OK: {rs.camera_info.usb_type_descriptor!r}")
if hasattr(rs, "stream"):
    for member in ("color", "depth"):
        if not hasattr(rs.stream, member):
            failures.append(f"rs.stream has no {member!r} member")
if hasattr(rs, "format"):
    for member in ("bgr8", "z16"):
        if not hasattr(rs.format, member):
            failures.append(f"rs.format has no {member!r} member")

try:
    import mediapipe as mp
except ImportError as err:
    print(f"FAIL: mediapipe import failed ({err}); is the [hardware] extra installed?")
    sys.exit(1)

# pose.py usage (MediaPipePoseDetector)
if not hasattr(mp, "solutions"):
    failures.append("mediapipe has no attribute 'solutions'")
elif not hasattr(mp.solutions, "pose"):
    failures.append("mp.solutions has no 'pose' module")
elif not hasattr(mp.solutions.pose, "Pose"):
    failures.append("mp.solutions.pose has no 'Pose' class")

if failures:
    print("FAIL: hardware-extra API smoke check:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("hardware-extra API smoke check passed")
