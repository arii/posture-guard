// Author: Ariel Anders
// Copyright (c) 2026 Ariel Anders
// SPDX-License-Identifier: MIT
//
// Pure gauge maths: which zone a frame is in, and the needle angle, label and
// skeleton colour that go with it. Kept free of DOM access so it can be unit
// tested under node; app.js does the drawing.

import { State } from "./posture-core.js";

/** The dial's arcs in index.html, in needle degrees (0 = far left). */
export const ZONE_DEGREES = Object.freeze({
  good: [0, 60],
  drift: [60, 120],
  bad: [120, 180],
});
const INSET_DEG = 4; // keep the needle off the arc boundaries, so the zone is never ambiguous

/** Skeleton colours, the same as the dial arcs. */
export const ZONE_COLOR = Object.freeze({
  good: "#4cc36a",
  drift: "#ffb340",
  bad: "#ff5c5c",
  none: "#8b95a3",
});

const LABEL = Object.freeze({ good: "upright", drift: "drifting", bad: "slouching" });

/**
 * One zone per frame, so the needle, the label and the skeleton always agree (#97).
 * The zone follows the state machine: a reminder state is "bad" even if only one metric
 * is past its threshold, and stays "bad" until recovery is confirmed. "drift" is the
 * slouch building toward the reminder (the engine's hold.kind "slipping"), the same
 * time the amber fill runs, so amber always comes with the fill (frame 05, #98).
 * Settling in (scored but not yet good or slouched) and unscored frames have no zone: grey.
 */
export function gaugeZone(state, scored, slipping) {
  if (state === State.BAD_CHIME || state === State.BAD_VOICE) return "bad";
  if (!scored) return "none";
  if (slipping) return "drift";
  if (state === State.GOOD) return "good";
  return "none"; // settling in: grey (frame 02, #98)
}

/**
 * The needle's angle in degrees: the engine's needle value mapped inside the zone's arc,
 * so it still moves with posture but never leaves the arc its label names.
 */
export function zoneAngle(zone, needle) {
  const [lo, hi] = ZONE_DEGREES[zone] || [0, 180];
  const f = Math.min(1, Math.max(0, (needle + 1) / 2));
  if (!ZONE_DEGREES[zone]) return lo + f * (hi - lo);
  return lo + INSET_DEG + f * (hi - lo - 2 * INSET_DEG);
}

/** Zone of a needle angle on the dial, for checks. */
export function zoneOfAngle(deg) {
  if (deg < ZONE_DEGREES.good[1]) return "good";
  if (deg < ZONE_DEGREES.drift[1]) return "drift";
  return "bad";
}

/** Label, angle and colour for a frame. */
export function gaugeView(state, scored, slipping, needle) {
  const zone = gaugeZone(state, scored, slipping);
  return {
    zone,
    label: LABEL[zone] ?? null,
    angle: needle === null || needle === undefined ? null : zoneAngle(zone, needle),
    color: ZONE_COLOR[zone],
  };
}

/** Time constant (s) of the needle glide: long enough to read as a sweep, short enough to track a slouch. */
export const NEEDLE_GLIDE_TAU_S = 0.35;

/**
 * The needle angle to draw: `shown` glides toward `target` (exponential, frame-rate independent),
 * so a change of zone sweeps across the dial instead of snapping up to 120 degrees in one frame.
 * Display only: the zone, label and colour still come from the state machine (gaugeView), and
 * `shown` null (first frame, or the needle was hidden) starts at the target.
 */
export function glideAngle(shown, target, dtSeconds, tau = NEEDLE_GLIDE_TAU_S) {
  if (shown === null || shown === undefined || !Number.isFinite(shown)) return target;
  if (!Number.isFinite(dtSeconds) || dtSeconds <= 0) return shown;
  return target + (shown - target) * Math.exp(-dtSeconds / tau);
}

/** SVG rotate() about the dial centre (100, 100) for an angle in degrees. */
export function angleTransform(deg) {
  return `rotate(${deg.toFixed(1)} 100 100)`;
}

const TILE_METRICS = ["head_forward", "head_tilt", "shoulder_level", "trunk_lean"];

/**
 * Body-part tile colours (frames 04 to 06, #98): the metric being cued takes the zone's colour
 * (amber while slipping, red while slouching); any other metric past its threshold is amber;
 * everything else is green. Returns { metric: "ok" | "warn" | "bad" } and the cued metric.
 */
export function tileStates(zone, cueMetric, metrics, cfg) {
  const out = {};
  for (const m of TILE_METRICS) {
    const v = metrics?.[`${m}_deg`];
    const past = Number.isFinite(v) && Math.abs(v) > cfg[`${m}_deg`];
    out[m] =
      m === cueMetric && (zone === "bad" || zone === "drift")
        ? zone === "bad"
          ? "bad"
          : "warn"
        : past
          ? "warn"
          : "ok";
  }
  return out;
}
