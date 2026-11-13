"""Shared dashboard state: the one bounded buffer between pipeline threads and
the FastAPI event loop (spec §dashboard).

Every publish replaces the previous value — subscribers always read the newest
annotated frame and metrics snapshot, never a backlog. Snapshot objects are
frozen, so the event loop can serialize them to WebSocket clients without
copying under the lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


@dataclass(frozen=True)
class MetricsSnapshot:
    """One frame's dashboard state, rendered as the WS /ws JSON schema."""

    state: str = "no_person"
    fps: float = 0.0
    depth_m: float | None = None
    head_forward_deg: float | None = None
    trunk_lean_deg: float | None = None
    head_tilt_deg: float | None = None
    shoulder_level_deg: float | None = None
    camera_connected: bool = False
    camera_usb_type: str | None = None
    camera_error: str | None = None
    audio_ready: bool = False
    audio_sink: str | None = None
    audio_tts_backend: str | None = None
    audio_error: str | None = None
    warnings: tuple[str, ...] = ()
    landmarks: tuple[tuple[float, float], ...] = field(default=())

    def to_payload(self) -> dict[str, object]:
        return {
            "type": "metrics",
            "state": self.state,
            "fps": round(self.fps, 1),
            "depth_m": _round_or_none(self.depth_m, 3),
            "metrics": {
                "head_forward_deg": _round_or_none(self.head_forward_deg),
                "trunk_lean_deg": _round_or_none(self.trunk_lean_deg),
                "head_tilt_deg": _round_or_none(self.head_tilt_deg),
                "shoulder_level_deg": _round_or_none(self.shoulder_level_deg),
            },
            "camera": {
                "connected": self.camera_connected,
                "usb_type": self.camera_usb_type,
                "error": self.camera_error,
            },
            "audio": {
                "ready": self.audio_ready,
                "sink": self.audio_sink,
                "tts_backend": self.audio_tts_backend,
                "error": self.audio_error,
            },
            "warnings": list(self.warnings),
            "landmarks": [[round(x, 4), round(y, 4)] for x, y in self.landmarks],
        }


class DashboardHub:
    """Latest-value store plus WebSocket client registry.

    All methods are thread-safe; pipeline threads publish and the asyncio
    event loop consumes. Each publish bumps a sequence number so consumers
    can cheaply detect freshness.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._metrics: MetricsSnapshot | None = None
        self._metrics_seq = 0
        self._clients: set[object] = set()

    # -- published values -------------------------------------------------------------

    def publish_video(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg = jpeg
            self._jpeg_seq += 1

    def latest_video(self) -> tuple[bytes, int] | None:
        """(jpeg, seq) of the newest annotated frame, or None before the first."""
        with self._lock:
            if self._jpeg is None:
                return None
            return self._jpeg, self._jpeg_seq

    def publish_metrics(self, snapshot: MetricsSnapshot) -> None:
        with self._lock:
            self._metrics = snapshot
            self._metrics_seq += 1

    def latest_metrics(self) -> tuple[MetricsSnapshot, int] | None:
        with self._lock:
            if self._metrics is None:
                return None
            return self._metrics, self._metrics_seq

    # -- websocket clients -------------------------------------------------------------

    def add_client(self, client: object) -> None:
        with self._lock:
            self._clients.add(client)

    def drop_client(self, client: object) -> None:
        with self._lock:
            self._clients.discard(client)

    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)
