"""FastAPI dashboard app (spec §dashboard).

Surfaces over the shared hub, all clients sharing one upstream buffer:
GET /video.mjpg streams annotated JPEGs; WS /ws pushes the metrics JSON and
drops dead clients; POST /nudge/test, /nudge/chime and /nudge/voice fire test
prompts; POST /feedback/settings applies runtime feedback settings.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from posture_guard.dashboard.hub import DashboardHub
from posture_guard.feedback import RuntimeFeedback, validate_changes

MJPEG_BOUNDARY = "frame"


class NudgeSink(Protocol):
    """The audio surface the dashboard needs — AudioWorker satisfies it."""

    def test_nudge(self) -> None: ...
    def play_chime(self) -> None: ...
    def play_voice(self) -> None: ...
    def render_prompt(self, text: str) -> str | None: ...


def create_app(
    hub: DashboardHub,
    audio: NudgeSink,
    *,
    feedback: RuntimeFeedback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> FastAPI:
    """Build the dashboard app.

    should_stop, when provided, ends the MJPEG generator's poll loop — used by
    tests (the TestClient close does not propagate a disconnect to a sync
    generator) and available for clean shutdown.
    """
    app = FastAPI(title="Posture Guard")

    @app.get("/video.mjpg")
    def video_mjpg() -> StreamingResponse:
        def frames() -> Iterator[bytes]:
            last_seq = -1
            while should_stop is None or not should_stop():
                latest = hub.latest_video()
                if latest is not None and latest[1] != last_seq:
                    jpeg, seq = latest
                    last_seq = seq
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                else:
                    time.sleep(0.04)  # poll for the next frame; hub holds one slot

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    @app.websocket("/ws")
    async def ws_metrics(websocket: WebSocket) -> None:
        await websocket.accept()
        hub.add_client(websocket)
        last_seq = -1
        direct_seq = -1
        try:
            while True:
                latest = hub.latest_metrics()
                if latest is not None and latest[1] != last_seq:
                    snapshot, seq = latest
                    last_seq = seq
                    await websocket.send_json(snapshot.to_payload())
                # Test-endpoint reminders publish without a snapshot; push them
                # straight through so the banner fires on click, not on the
                # next metrics frame.
                direct = hub.latest_direct_event()
                if direct is not None and direct[1] != direct_seq:
                    event, seq = direct
                    direct_seq = seq
                    await websocket.send_json({"type": "nudge_event", **event})
                # Short receive timeout doubles as disconnect detection, so a
                # dead client is dropped even when no new metrics arrive.
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=0.02)
                except asyncio.TimeoutError:
                    continue
                if message.get("type") == "websocket.disconnect":
                    break
        except (RuntimeError, WebSocketDisconnect):
            pass  # client went away mid-send: dropped in finally, never fatal
        finally:
            hub.drop_client(websocket)

    @app.post("/nudge/test")
    def nudge_test() -> dict[str, str]:
        audio.test_nudge()
        hub.publish_nudge_event("test", text="chime + voice")
        return {"status": "queued"}

    @app.post("/nudge/chime")
    def nudge_chime() -> dict[str, str]:
        audio.play_chime()
        hub.publish_nudge_event("chime")
        return {"status": "queued"}

    @app.post("/nudge/voice")
    def nudge_voice() -> dict[str, str]:
        audio.play_voice()
        hub.publish_nudge_event(
            "voice", text=feedback.prompt_text() if feedback is not None else None
        )
        return {"status": "queued"}

    @app.post("/feedback/settings")
    def feedback_settings(body: dict[str, object]) -> dict[str, object]:
        """Apply runtime feedback settings (mode, voice timings, prompt text).

        Validation is all-or-nothing: an invalid body applies nothing and
        returns 422 with the reason. Prompt-text changes re-render the spoken
        WAV live (audio.render_prompt); no restart anywhere.
        """
        if feedback is None:
            raise HTTPException(status_code=503, detail="feedback settings not wired up")
        changes, error = validate_changes(body)
        if error is not None:
            raise HTTPException(status_code=422, detail=error)
        text_changed = "chime_text" in changes
        payload = feedback.apply(**changes)
        if text_changed:
            audio.render_prompt(str(changes["chime_text"]))
        return payload

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    return app

