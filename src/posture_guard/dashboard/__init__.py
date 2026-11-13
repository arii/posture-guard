"""FastAPI dashboard: MJPEG video, WebSocket metrics, test-nudge endpoint."""

from posture_guard.dashboard.app import create_app
from posture_guard.dashboard.hub import DashboardHub, MetricsSnapshot

__all__ = ["DashboardHub", "MetricsSnapshot", "create_app"]
