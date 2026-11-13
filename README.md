# Posture Guard

Linux desktop posture corrector: watches you through an Intel RealSense D400-series
camera, scores posture from head/shoulder geometry (gated by depth), and nudges you
audibly — chime first, then voice — while a local web dashboard shows the live
skeleton so you can verify tracking and tune camera placement.

See the approved spec (OD-10) for architecture and acceptance criteria.

## Layout

```
src/posture_guard/
    config.py    # config.yaml -> typed dataclasses (single source of truth)
    camera.py    # FrameSource interface + synthetic + RealSense sources
    scoring.py   # pure posture metrics, depth gate, visibility floor
    state.py     # nudge state machine (injected timestamps, no sleeps)
tests/          # pytest: pose fixtures + deterministic timeline tests
config.yaml     # thresholds, durations, distance band, camera settings
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check .
pytest
```

Hardware setup (RealSense SDK, audio tools, dashboard) lands with the
camera/audio and dashboard slices; `setup.sh` and the hardware runbook ship
with the final slice.
