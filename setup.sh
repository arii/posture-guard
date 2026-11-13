#!/usr/bin/env bash
# Posture Guard setup for a clean Ubuntu/Debian-family Linux desktop.
#
# 1. Installs system packages (PulseAudio/PipeWire client tools, TTS, ffmpeg).
# 2. Creates a Python venv and installs posture-guard with the hardware extra.
# 3. Prints a hardware preflight report: camera, USB type, depth stream,
#    audio sinks, TTS renderability.
#
# Re-run at any time; every step is idempotent.

set -euo pipefail

cd "$(dirname "$0")"

# --- 1. System packages ------------------------------------------------------
# pulseaudio-utils provides pactl/paplay (works under PulseAudio and the
# PipeWire compatibility layer). libttspico-utils provides the pico2wave
# binary (the preferred TTS engine); it lives in Ubuntu's multiverse and
# Debian's non-free component. espeak-ng is the fallback the audio worker
# uses when pico2wave is absent.
if command -v sudo >/dev/null 2>&1; then
    SUDO=sudo
else
    SUDO=""
fi

echo "==> Installing system packages"
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libpulse0 pulseaudio-utils ffmpeg libttspico-utils espeak-ng \
    || $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libpulse0 pulseaudio-utils ffmpeg espeak-ng

# --- 1b. RealSense udev rules ------------------------------------------------
# The rules file grants non-root access to the RealSense USB device. Installing
# it is best-effort: without root we print the manual command instead of failing.
RULES_SRC="99-realsense-libusb.rules"
RULES_DST="/etc/udev/rules.d/99-realsense-libusb.rules"
if [[ -f "$RULES_SRC" ]]; then
    if [[ -f "$RULES_DST" ]] && cmp -s "$RULES_SRC" "$RULES_DST"; then
        echo "==> RealSense udev rules already installed"
    elif [[ "$(id -u)" -eq 0 ]]; then
        echo "==> Installing RealSense udev rules"
        cp "$RULES_SRC" "$RULES_DST"
        udevadm control --reload-rules && udevadm trigger
    else
        echo "==> Skipping udev rule install (not root) — to install manually run:"
        echo "      sudo cp \"$(pwd)/$RULES_SRC\" $RULES_DST"
        echo "      sudo udevadm control --reload-rules && sudo udevadm trigger"
    fi
fi

# --- 2. Python environment ---------------------------------------------------
echo "==> Creating venv and installing Python dependencies"
python3 -m venv .venv
# shellcheck disable=SC1091  # venv activate only exists after setup runs
. .venv/bin/activate
pip install --quiet --upgrade pip
# hardware extra pulls pyrealsense2, mediapipe and opencv-python; dev adds
# pytest/ruff so `make test` works on the same box.
pip install --quiet -e '.[hardware,dev]'

# --- 3. Hardware preflight report -------------------------------------------
echo
echo "==> Hardware preflight report"
.venv/bin/python - <<'PY'
"""Preflight checks: camera, USB type, depth stream, audio sinks, TTS."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


print("-" * 62)

# Camera + USB type + depth stream (one pipeline probe covers all three).
camera_ok, usb, depth_ok = False, "n/a", False
try:
    import pyrealsense2 as rs

    ctx = rs.context()
    devices = list(ctx.query_devices())
    if devices:
        camera_ok = True
        dev = devices[0]
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(dev.get_info(rs.camera_info.serial_number))
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        try:
            profile = pipe.start(cfg)
            for _ in range(15):  # a few frames; depth needs time to settle
                frames = pipe.wait_for_frames(timeout_ms=5000)
                if frames.get_depth_frame() is not None:
                    depth_ok = True
                    break
        finally:
            try:
                pipe.stop()
            except RuntimeError:
                pass
except Exception as exc:  # noqa: BLE001 — report, never crash, the preflight
    camera_note = str(exc).strip().splitlines()[0] if str(exc).strip() else "unknown error"
else:
    camera_note = f"{len(list(rs.context().query_devices()))} device(s)"

print(f"camera detected .............. {mark(camera_ok)}   ({camera_note})")
print(f"USB connection type .......... {'PASS' if usb.startswith('3') else 'WARN'}   ({usb})"
      + ("" if usb.startswith("3") else "  — USB 2 may drop frames on depth streams"))
print(f"depth stream readable ........ {mark(depth_ok)}")

# Audio sinks via pactl (PulseAudio or PipeWire's pipewire-pulse).
sinks: list[str] = []
if shutil.which("pactl") is None:
    print("audio sinks .................. FAIL   (pactl not found — is pulseaudio-utils installed?)")
else:
    try:
        out = subprocess.run(
            ["pactl", "list", "short", "sinks"], capture_output=True, text=True, timeout=10
        )
        sinks = [line.split("\t")[1] for line in out.stdout.splitlines() if line.strip()]
        if sinks:
            print(f"audio sinks .................. PASS   ({', '.join(sinks)})")
        else:
            print("audio sinks .................. FAIL   (no sinks — is the audio server running?)")
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"audio sinks .................. FAIL   ({exc})")

# TTS: pico2wave preferred, espeak-ng fallback — actually render a WAV.
tts_engine = shutil.which("pico2wave") or shutil.which("espeak-ng")
tts_ok = False
if tts_engine is None:
    print("TTS renderable ............... FAIL   (neither pico2wave nor espeak-ng found)")
else:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
            wav_path = wav.name
        subprocess.run(
            [tts_engine, "-w", wav_path, "test"], capture_output=True, timeout=30, check=True
        )
        tts_ok = Path(wav_path).stat().st_size > 0
    finally:
        Path(wav_path).unlink(missing_ok=True)
    print(f"TTS renderable ............... {mark(tts_ok)}   ({Path(tts_engine).name})")

print("-" * 62)
print("Next: open the README 'Hardware runbook' section and follow it step by step.")
PY

echo
echo "Setup complete. Activate the venv with:  . .venv/bin/activate"
echo "Serve the dashboard with:                make run"
