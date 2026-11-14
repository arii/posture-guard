"""Pure composite-score math for the dashboard posture needle.

Display-only: mirrors the client-side computation in index.html so the
aggregation is importable and unit-testable in Python (tests/test_needle.py).
No changes to scoring or the state machine.

The score tracks the WORST metric's severity, mapped onto the gauge so the
needle color always agrees with the scoring zone (scoring.zone_of): green is
good, amber is the ok/neutral hysteresis band, red is bad (past a threshold).
A mean would disagree in both directions (one maxed metric reads green while
the chime fires; every metric at 95% reads red while silent), so the mean is
deliberately not used here.
"""

from __future__ import annotations

Metrics = dict[str, float | None]

# Gauge thirds: below GOOD_END is green, below BAD_START is amber, at or above
# BAD_START is red. The mapping anchors the worst metric's exit fraction to
# GOOD_END and its enter threshold (severity 1) to BAD_START.
GOOD_END = 100.0 / 3.0
BAD_START = 200.0 / 3.0

MARGIN_KEY = "hysteresis_margin_deg"


def metric_severity(value: float | None, threshold: float | None) -> float | None:
    """Severity of one metric: |value| as a fraction of its band, clamped to 1.

    Returns None when the metric is unavailable (no reading, or no threshold
    for it, e.g. trunk lean when hips are not visible).
    """
    if value is None or threshold is None or threshold <= 0:
        return None
    return min(1.0, abs(value) / threshold)


def composite_score(
    metrics: Metrics, thresholds: dict[str, float | bool] | None
) -> float | None:
    """Needle position 0-100 from the worst metric's severity.

    The worst metric's exit fraction (its threshold minus the hysteresis
    margin) maps to GOOD_END and severity 1 maps to BAD_START; past the
    threshold the red zone fills up to 2x severity. Trunk lean counts whenever
    it has a value: the pipeline sends None exactly while the hips are hidden,
    mirroring the scoring zone's trunk_available rule, so a present value is
    always trustworthy. Returns None when nothing is usable (no person, or no
    thresholds received yet).
    """
    if not thresholds:
        return None
    margin = thresholds.get(MARGIN_KEY, 0.0)
    margin_deg = float(margin) if isinstance(margin, (int, float)) else 0.0
    worst: tuple[float, float] | None = None  # (severity, exit_fraction)
    for key, value in metrics.items():
        if value is None:
            continue
        threshold = thresholds.get(key)
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            continue
        band = float(threshold)
        severity = abs(value) / band
        exit_frac = max(0.0, (band - margin_deg) / band)
        if worst is None or severity > worst[0]:
            worst = (severity, exit_frac)
    if worst is None:
        return None
    severity, exit_frac = worst
    if severity <= exit_frac:
        return GOOD_END * (severity / exit_frac) if exit_frac > 0 else 0.0
    if severity <= 1.0:
        span = 1.0 - exit_frac
        return GOOD_END + (BAD_START - GOOD_END) * (
            (severity - exit_frac) / span if span > 0 else 1.0
        )
    return BAD_START + (100.0 - BAD_START) * min(severity - 1.0, 1.0)
