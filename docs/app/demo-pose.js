// Author: Ariel Anders
// Copyright (c) 2026 Ariel Anders
// SPDX-License-Identifier: MIT
//
// The scripted "Try the demo" poses: 20 s of good posture, then 20 s with the head
// forward, 20 s good, then 20 s with one shoulder lower, repeating (80 s loop), easing in and
// out of each fault over DEMO_EASE_S. Pure, so
// tests can replay exactly what the demo shows.

export const DEMO_CYCLE_S = 20;
/** Seconds the demo takes to move into (and back out of) a fault. Real people do not jump 0.116 of the frame in one frame. */
export const DEMO_EASE_S = 3;

/** The pose of one state at time `t`: "good", "slouch" (head forward) or "shoulder" (one shoulder lower). */
function poseOf(kind, t) {
  const slouch = kind === "slouch";
  const lm = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, visibility: 0.9 }));
  const wobble = Math.sin(t * 1.7) * 0.003;
  lm[11] = { x: 0.6 + wobble, y: 0.62, visibility: 0.99 }; // unmirrored: left shoulder on the right
  lm[12] = { x: 0.4 + wobble, y: 0.62, visibility: 0.99 };
  lm[23] = { x: 0.56, y: 0.92, visibility: 0.9 }; // #109: off the bottom edge, so the hip dots are never clipped
  lm[24] = { x: 0.44, y: 0.92, visibility: 0.9 };
  const ex = slouch ? -0.02 : 0;
  const ey = slouch ? 0.116 : 0;
  // Ear spacing and ear-to-shoulder gap follow the demo avatar's own proportions (head width
  // about 0.69 of shoulder width, ears about 0.70 shoulder widths above the shoulders).
  lm[7] = { x: 0.5 + 0.065 + ex + wobble, y: 0.435 + ey, visibility: 0.99 };
  lm[8] = { x: 0.5 - 0.065 + ex + wobble, y: 0.435 + ey, visibility: 0.99 };
  if (kind === "shoulder") {
    // One shoulder lower: the same fault the Learn the cues step shows. The head goes with the
    // shoulders, so only the shoulder cue fires.
    lm[11] = { ...lm[11], y: lm[11].y - 0.03 };
    lm[7] = { ...lm[7], y: lm[7].y - 0.012 };
  }
  return lm;
}

function kindOfCycle(cycle) {
  if (cycle < 0 || cycle % 2 === 0) return "good";
  return cycle % 4 === 3 ? "shoulder" : "slouch";
}

/**
 * Landmarks (33, MediaPipe order) for demo time `t` in seconds. Each cycle starts by easing
 * (smoothstep over DEMO_EASE_S) from the previous cycle's pose, so the avatar and the gauge needle
 * glide into and out of a fault instead of jumping. Every pose in between is a straight blend of
 * two real poses, so a fault is never smeared past its own pose: head-forward and shoulder
 * faults each scale only their own landmarks.
 */
export function demoPose(t) {
  const cycle = Math.floor(t / DEMO_CYCLE_S);
  const within = t - cycle * DEMO_CYCLE_S;
  const to = poseOf(kindOfCycle(cycle), t);
  const fromKind = kindOfCycle(cycle - 1);
  if (within >= DEMO_EASE_S || fromKind === kindOfCycle(cycle)) return to;
  const u = within / DEMO_EASE_S;
  const w = u * u * (3 - 2 * u);
  const from = poseOf(fromKind, t);
  return to.map((p, i) => ({
    x: from[i].x + (p.x - from[i].x) * w,
    y: from[i].y + (p.y - from[i].y) * w,
    visibility: p.visibility,
  }));
}

/**
 * The demo's pose for a Learn the cues step (#103): upright for the first step, then the
 * one fault the step teaches, so the skeleton and the tiles show what the card says.
 * metric: null | "head_forward" | "head_tilt" | "shoulder_level" | "trunk_lean".
 */
export function practicePose(metric, t) {
  const lm = poseOf("good", t); // always the good pose, never mid-ease
  if (metric === "head_forward") {
    for (const i of [7, 8]) lm[i] = { ...lm[i], x: lm[i].x - 0.02, y: lm[i].y + 0.116 };
  } else if (metric === "head_tilt") {
    lm[7] = { ...lm[7], y: lm[7].y - 0.019 };
    lm[8] = { ...lm[8], y: lm[8].y + 0.019 };
  } else if (metric === "shoulder_level") {
    lm[11] = { ...lm[11], y: lm[11].y - 0.03 };
    lm[7] = { ...lm[7], y: lm[7].y - 0.012 }; // the head goes with the shoulders: no tilt cue
  } else if (metric === "trunk_lean") {
    for (const i of [7, 8, 11, 12]) lm[i] = { ...lm[i], x: lm[i].x + 0.06 };
  }
  return lm;
}
