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
from posture_guard.feedback import FeedbackMode, RuntimeFeedback


class RecordingAudio:
    """NudgeSink double that counts test-nudge calls and re-renders."""

    def __init__(self) -> None:
        self.calls = 0
        self.chime_calls = 0
        self.voice_calls = 0
        self.rendered: list[str] = []

    def test_nudge(self) -> None:
        self.calls += 1

    def play_chime(self) -> None:
        self.chime_calls += 1

    def play_voice(self) -> None:
        self.voice_calls += 1

    def render_prompt(self, text: str) -> str | None:
        self.rendered.append(text)
        return "pico2wave"


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


def test_nudge_endpoints_publish_visible_reminder_events() -> None:
    """Every reminder is visible, not just audible: the test endpoints publish
    banner events through the hub (voice carries the prompt text), with
    sequence numbers from the same counter as state-machine nudges."""
    hub, audio = DashboardHub(), RecordingAudio()
    client = TestClient(create_app(hub, audio, feedback=RuntimeFeedback(chime_text="sit up")))

    client.post("/nudge/test")
    test_event, test_seq = hub.latest_direct_event()
    client.post("/nudge/chime")
    chime_event, chime_seq = hub.latest_direct_event()
    client.post("/nudge/voice")
    voice_event, voice_seq = hub.latest_direct_event()

    assert test_event["kind"] == "test" and test_event["text"] == "chime + voice"
    assert chime_event["kind"] == "chime"
    assert voice_event == {"kind": "voice", "seq": voice_event["seq"], "text": "sit up"}
    assert test_seq < chime_seq < voice_seq  # one shared counter, no collisions


def test_ws_pushes_test_reminder_events_without_a_snapshot() -> None:
    """The banner must fire on click, not on the next metrics frame: direct
    events go out as their own WS message type."""
    hub, audio = DashboardHub(), RecordingAudio()
    client = TestClient(create_app(hub, audio))

    with client.websocket_connect("/ws") as ws:
        hub.publish_nudge_event("chime")
        payload = ws.receive_json()

    assert payload["type"] == "nudge_event"
    assert payload["kind"] == "chime"
    assert "seq" in payload


def test_separate_chime_and_voice_endpoints() -> None:
    """Each channel fires independently; mode/silence cannot block them
    (spec: test endpoints are hardware-verification commands)."""
    hub, audio = DashboardHub(), RecordingAudio()
    client = TestClient(create_app(hub, audio, feedback=RuntimeFeedback(mode=FeedbackMode.SILENT)))

    assert client.post("/nudge/chime").json() == {"status": "queued"}
    assert client.post("/nudge/voice").json() == {"status": "queued"}

    assert audio.chime_calls == 1
    assert audio.voice_calls == 1
    assert audio.calls == 0  # combined test endpoint untouched


def test_feedback_settings_apply_at_runtime() -> None:
    hub, audio = DashboardHub(), RecordingAudio()
    feedback = RuntimeFeedback()
    client = TestClient(create_app(hub, audio, feedback=feedback))

    response = client.post(
        "/feedback/settings",
        json={"mode": "chime_only", "voice_delay_s": 4.0,
              "voice_repeat_cooldown_s": 12.0, "chime_text": "  sit up  "},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "chime_only"
    assert feedback.voice_timings() == (4.0, 12.0)
    assert feedback.prompt_text() == "sit up"
    assert audio.rendered == ["sit up"]  # prompt re-rendered live, no restart


def test_feedback_settings_reject_invalid_body_without_partial_apply() -> None:
    hub, audio = DashboardHub(), RecordingAudio()
    feedback = RuntimeFeedback()
    client = TestClient(create_app(hub, audio, feedback=feedback))

    response = client.post("/feedback/settings", json={"mode": "vibrate"})

    assert response.status_code == 422
    assert "vibrate" in response.json()["detail"]
    assert feedback.mode is FeedbackMode.CHIME_AND_VOICE  # nothing applied
    assert audio.rendered == []


def test_feedback_settings_require_wired_feedback() -> None:
    client = TestClient(create_app(DashboardHub(), RecordingAudio()))

    response = client.post("/feedback/settings", json={"mode": "silent"})

    assert response.status_code == 503


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
        "trunk_lean_required": True,
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
    assert payload["nudge_event"] is None


def test_ws_payload_carries_feedback_settings_and_nudge_event() -> None:
    """Settings ride the snapshot like thresholds; a nudge event is keyed by
    sequence so the UI can re-trigger the banner on repeated nudges."""
    hub = DashboardHub()
    event = {"kind": "chime", "seq": 3, "muted": False}
    hub.publish_metrics(
        make_snapshot(
            feedback={"mode": "chime_only", "session_only": True},
            nudge_event=event,
        )
    )
    client = TestClient(create_app(hub, RecordingAudio()))

    with client.websocket_connect("/ws") as ws:
        payload = ws.receive_json()

    assert payload["feedback"] == {"mode": "chime_only", "session_only": True}
    assert payload["nudge_event"] == event


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
    # tiles, state badge, the test-nudge button and the feedback controls.
    for fragment in (
        "<canvas", "/ws", "/video.mjpg", "/nudge/test", "badge",
        "/nudge/chime", "/nudge/voice", "/feedback/settings",
    ):
        assert fragment in response.text


def test_index_has_feedback_controls_and_visual_event_targets() -> None:
    """Mode selector, per-channel buttons, settings form with the session-only
    marker, and the synchronized nudge-event banner targets (spec §UI)."""
    client = TestClient(create_app(DashboardHub(), RecordingAudio()))

    response = client.get("/")

    assert response.status_code == 200
    page = response.text
    assert 'name="feedback-mode"' in page
    for mode in ("chime_and_voice", "chime_only", "silent"):
        assert f'value="{mode}"' in page
    for element_id in ("play-chime", "play-voice", "apply-settings", "set-chime_text",
                       "set-voice_delay_s", "set-voice_repeat_cooldown_s",
                       "nudge-event", "feedback-error"):
        assert f'id="{element_id}"' in page
    assert "session-only" in page  # UI states when a setting is not persisted
    # Plain language: 'reminder' everywhere, never 'nudge' (task scope 2).
    assert "Test reminder" in page
    assert "Test nudge" not in page
    assert "Reminder: " in page
    # Audio line carries the full status: ready/TTS/sink plus an error target.
    assert 'id="audio-detail"' in page
    assert 'id="audio-error"' in page
    assert "tts_backend" in page
    # Trunk-lean-optional note target on the trunk tile band.
    assert "trunk_note" in page


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


def test_docs_page_is_served_and_linked_from_index() -> None:
    """The How-it-works page lives at /docs and the main view links to it."""
    client = TestClient(create_app(DashboardHub(), RecordingAudio()))

    response = client.get("/docs")
    index = client.get("/")

    assert response.status_code == 200
    page = response.text
    assert "How it works" in page
    # All five sections present: pipeline, model, privacy, story, roadmap.
    for heading in ("The pipeline", "The model", "Privacy", "Where it came from", "Roadmap"):
        assert heading in page
    assert index.status_code == 200
    assert 'href="/docs"' in index.text
