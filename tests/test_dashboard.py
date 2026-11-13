"""Dashboard API tests (spec acceptance: TestClient against the fake hub).

No camera, no encoder, no audio hardware: the hub is fed synthetic snapshots
and JPEG bytes, and audio is a recording stub behind the NudgeSink protocol.
"""

from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from posture_guard.dashboard.app import create_app
from posture_guard.dashboard.hub import DashboardHub, MetricsSnapshot, ScoringThresholds


class RecordingAudio:
    """NudgeSink double that counts test-nudge calls."""

    def __init__(self) -> None:
        self.calls = 0

    def test_nudge(self) -> None:
        self.calls += 1


def make_snapshot(**overrides: object) -> MetricsSnapshot:
    defaults: dict[str, object] = {
        "state": "good",
        "fps": 30.0,
        "depth_m": 1.4,
        "head_forward_deg": 0.0,
        "trunk_lean_deg": 0.0,
        "head_tilt_deg": 0.0,
        "shoulder_level_deg": 0.0,
        "camera_connected": True,
        "audio_ready": True,
        "warnings": (),
        "landmarks": ((0.4, 0.6), (0.6, 0.6)),
        "thresholds": ScoringThresholds(
            head_forward_deg=12.0,
            trunk_lean_deg=10.0,
            head_tilt_deg=12.0,
            shoulder_level_deg=6.0,
            hysteresis_margin_deg=3.0,
            visibility_floor=0.5,
        ),
    }
    defaults.update(overrides)
    return MetricsSnapshot(**defaults)  # type: ignore[arg-type]


def test_nudge_endpoint_enqueues_test_prompt() -> None:
    hub, audio = DashboardHub(), RecordingAudio()
    client = TestClient(create_app(hub, audio))

    response = client.post("/nudge/test")

    assert response.status_code == 200
    assert response.json() == {"status": "queued"}
    assert audio.calls == 1


def test_mjpeg_streams_annotated_frames_from_shared_buffer() -> None:
    hub = DashboardHub()
    hub.publish_video(b"jpeg-bytes-one")
    stop = threading.Event()
    client = TestClient(create_app(hub, RecordingAudio(), should_stop=stop.is_set))

    with client.stream("GET", "/video.mjpg") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"jpeg-bytes-one" in next(response.iter_raw())
        # End the generator so the stream closes cleanly (see create_app).
        stop.set()


def test_ws_pushes_metrics_json_schema() -> None:
    hub = DashboardHub()
    hub.publish_metrics(make_snapshot(state="good", warnings=("person too far",)))
    client = TestClient(create_app(hub, RecordingAudio()))

    with client.websocket_connect("/ws") as ws:
        payload = ws.receive_json()

    assert payload["type"] == "metrics"
    assert payload["state"] == "good"
    assert payload["fps"] == 30.0
    assert payload["depth_m"] == 1.4
    assert set(payload["metrics"]) == {
        "head_forward_deg",
        "trunk_lean_deg",
        "head_tilt_deg",
        "shoulder_level_deg",
    }
    # Scoring thresholds ride alongside the values so the UI never hardcodes
    # them (spec: healthy bands render from config).
    assert payload["thresholds"] == {
        "head_forward_deg": 12.0,
        "trunk_lean_deg": 10.0,
        "head_tilt_deg": 12.0,
        "shoulder_level_deg": 6.0,
        "hysteresis_margin_deg": 3.0,
        "visibility_floor": 0.5,
    }
    assert payload["camera"] == {
        "connected": True,
        "usb_type": None,
        "error": None,
        "source": None,
    }
    assert payload["audio"]["ready"] is True
    assert payload["warnings"] == ["person too far"]
    assert payload["landmarks"] == [[0.4, 0.6], [0.6, 0.6]]


def test_ws_pushes_each_new_snapshot() -> None:
    hub = DashboardHub()
    hub.publish_metrics(make_snapshot(state="no_person"))
    client = TestClient(create_app(hub, RecordingAudio()))

    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["state"] == "no_person"
        hub.publish_metrics(make_snapshot(state="bad_chime"))
        assert ws.receive_json()["state"] == "bad_chime"


def test_ws_unreliable_detail_rides_the_payload() -> None:
    hub = DashboardHub()
    detail = "hips not visible (left 0.31, right 0.28) - trunk lean unavailable"
    hub.publish_metrics(make_snapshot(state="unreliable", unreliable_detail=detail))
    client = TestClient(create_app(hub, RecordingAudio()))

    with client.websocket_connect("/ws") as ws:
        payload = ws.receive_json()

    assert payload["unreliable_detail"] == detail


def test_index_serves_placement_guide_diagrams() -> None:
    """The placement card embeds both guide SVGs inline (collapsed by default)."""
    hub = DashboardHub()
    client = TestClient(create_app(hub, RecordingAudio()))

    response = client.get("/")

    assert response.status_code == 200
    page = response.text
    assert 'id="placement-guide"' in page
    assert "<details" in page  # collapsed so video/metrics stay above the fold
    # Both scenes from the angled placement guide, as inline SVG.
    assert 'id="placement-guide-top"' in page
    assert 'id="placement-guide-side"' in page
    assert "aria-label=\"Top view: camera positions by angle" in page
    assert "aria-label=\"Side view: camera above and beside the monitor" in page
    # High-mount-first text (PR #5) stays alongside the diagrams.
    assert "Mount the camera HIGH" in page
    assert "Alternative for deep desks" in page


def test_dead_ws_client_is_dropped() -> None:
    hub = DashboardHub()
    hub.publish_metrics(make_snapshot())
    client = TestClient(create_app(hub, RecordingAudio()))

    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        deadline = time.monotonic() + 5.0
        while hub.client_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert hub.client_count() == 1
    # Leaving the context closes the socket: the handler must drop the client.
    deadline = time.monotonic() + 5.0
    while hub.client_count() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert hub.client_count() == 0


def test_index_page_is_served() -> None:
    client = TestClient(create_app(DashboardHub(), RecordingAudio()))

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # The page is plain DOM driven by the WS payload: overlay canvas, metric
    # tiles, state badge and the test-nudge button.
    for fragment in ("<canvas", "/ws", "/video.mjpg", "/nudge/test", "badge"):
        assert fragment in response.text


def test_index_has_band_labels_and_unreliable_detail_targets() -> None:
    """Every metric tile carries a healthy-band element and the status row
    carries the unreliable-reason target — both filled from the WS payload."""
    client = TestClient(create_app(DashboardHub(), RecordingAudio()))

    response = client.get("/")

    assert response.status_code == 200
    page = response.text
    for key in ("head_forward_deg", "trunk_lean_deg", "head_tilt_deg", "shoulder_level_deg"):
        assert f'id="b-{key}"' in page  # band label target per metric tile
    assert 'id="unreliable-detail"' in page
    # Band text is composed from the server thresholds, not hardcoded.
    assert "good within" in page
    assert "payload.thresholds" in page
