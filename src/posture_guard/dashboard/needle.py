"""Pure composite-score math for the dashboard posture needle.

Display-only: mirrors the client-side computation in index.html so the
aggregation is importable and unit-testable in Python (tests/test_needle.py).
No changes to scoring or the state machine.
"""

from __future__ import annotations

Metrics = dict[str, float | None]


def metric_severity(value: float | None, threshold: float | None) -> float | None:
    """Severity of one metric: |value| as a fraction of its band, clamped to 1.

    Returns None when the metric is unavailable (no reading, or no threshold
    for it, e.g. trunk lean when hips are not visible).
    """
    if value is None or threshold is None or threshold <= 0:
        return None
    return min(1.0, abs(value) / threshold)


def composite_score(
    metrics: Metrics, thresholds: dict[str, float] | None
) -> float | None:
    """Aggregate metrics to 0-100: 0 = centered in all bands, 100 = at/over.

    Unavailable metrics are skipped; returns None when nothing is usable
    (no person, or no thresholds received yet).
    """
    if not thresholds:
        return None
    severities = [
        severity
        for key, value in metrics.items()
        if (severity := metric_severity(value, thresholds.get(key))) is not None
    ]
    if not severities:
        return None
    return 100.0 * sum(severities) / len(severities)
