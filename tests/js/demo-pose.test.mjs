// Author: Ariel Anders
// Copyright (c) 2026 Ariel Anders
// SPDX-License-Identifier: MIT
//
// demo-pose.js: the scripted demo, replayed through the engine exactly as the app runs it.

import assert from "node:assert/strict";
import test from "node:test";
import { DEMO_CYCLE_S, DEMO_EASE_S, demoPose, practicePose } from "../../docs/app/demo-pose.js";
import { gaugeView, glideAngle, ZONE_COLOR, zoneOfAngle } from "../../docs/app/mini-view.js";
import { DEFAULTS, PostureEngine, State, scorePose } from "../../docs/app/posture-core.js";

test("the demo alternates good and slouch every cycle", () => {
  const head = (t) => demoPose(t)[7].y;
  assert.ok(head(DEMO_CYCLE_S + DEMO_EASE_S + 1) > head(1), "the head drops in the slouch half");
  assert.equal(head(2 * DEMO_CYCLE_S + DEMO_EASE_S + 1).toFixed(3), head(1).toFixed(3));
});

test("demo replay at 15 fps: gauge, label and skeleton agree through slouch onset and recovery", () => {
  const engine = new PostureEngine(DEFAULTS);
  engine.setDriftBaselineEnabled(false); // as app.js does for the synthetic demo
  const seen = new Set();
  let prev = null;
  let recoveries = 0;
  for (let t = 0; t < 4 * DEMO_CYCLE_S; t += 1 / 15) {
    const out = engine.process(demoPose(t), t);
    const view = gaugeView(
      out.transition.state,
      out.result.kind === "scored",
      out.hold?.kind === "slipping",
      out.needle,
    );
    const where = `t=${t.toFixed(2)} ${out.transition.state}`;
    if (view.zone !== "none" && view.angle !== null) assert.equal(zoneOfAngle(view.angle), view.zone, where);
    if (view.label === "slouching") assert.equal(view.color, ZONE_COLOR.bad, where);
    if (view.zone === "good") assert.equal(view.color, ZONE_COLOR.good, where);
    // Amber exactly while the fill runs toward the chime (frame 05), never amber without it.
    if (view.zone !== "bad") assert.equal(view.zone === "drift", out.hold?.kind === "slipping", where);
    if (prev === State.BAD_CHIME || prev === State.BAD_VOICE) {
      if (out.transition.state === State.GOOD) recoveries += 1;
    }
    prev = out.transition.state;
    seen.add(view.zone);
  }
  assert.ok(seen.has("good") && seen.has("drift") && seen.has("bad"), [...seen].join());
  assert.ok(recoveries >= 1, "the replay covers a recovery");
});

test("Learn the cues poses show only the taught fault, at any demo time (#103)", () => {
  const cfg = DEFAULTS.scoring;
  const keys = ["head_forward", "head_tilt", "shoulder_level", "trunk_lean"];
  for (const t of [1, DEMO_CYCLE_S + 5, 3 * DEMO_CYCLE_S + 2]) {
    for (const metric of [null, ...keys]) {
      const m = scorePose(practicePose(metric, t), cfg);
      for (const k of keys) {
        const v = Math.abs(m[`${k}_deg`]);
        if (k === metric) assert.ok(v > cfg[`${k}_deg`], `${metric} at ${t}: ${v}`);
        else assert.ok(v < cfg[`${k}_deg`] - cfg.hysteresis_margin_deg, `${metric} at ${t}: ${k} ${v}`);
      }
    }
  }
});

test("the demo eases into and out of every fault: no landmark jumps between 60 fps frames", () => {
  let worst = 0;
  let prev = demoPose(0);
  for (let t = 1 / 60; t < 4 * DEMO_CYCLE_S; t += 1 / 60) {
    const lm = demoPose(t);
    for (let i = 0; i < lm.length; i++)
      worst = Math.max(worst, Math.abs(lm[i].x - prev[i].x), Math.abs(lm[i].y - prev[i].y));
    prev = lm;
  }
  assert.ok(worst < 0.005, `largest per-frame landmark step ${worst}`); // was 0.116 at each cycle boundary
});

test("the ease reaches the full fault: a slouch is not weakened, only reached gradually", () => {
  const cfg = DEFAULTS.scoring;
  const mid = scorePose(demoPose(DEMO_CYCLE_S + DEMO_EASE_S / 2), cfg).head_forward_deg;
  const full = scorePose(demoPose(DEMO_CYCLE_S + DEMO_EASE_S + 1), cfg).head_forward_deg;
  const early = scorePose(demoPose(DEMO_CYCLE_S + 0.1), cfg).head_forward_deg;
  assert.ok(Math.abs(early) < Math.abs(mid) && Math.abs(mid) < Math.abs(full), `${early} ${mid} ${full}`);
  const afterEase = demoPose(DEMO_CYCLE_S + DEMO_EASE_S + 1)[7].y;
  assert.ok(Math.abs(afterEase - 0.551) < 0.002, `slouch head y ${afterEase}`); // 0.435 + 0.116, as before
});

test("an eased demo shows only the one fault it is easing: shoulder ease keeps head metrics in range", () => {
  const cfg = DEFAULTS.scoring;
  for (let t = 3 * DEMO_CYCLE_S; t < 3 * DEMO_CYCLE_S + DEMO_EASE_S + 0.5; t += 0.1) {
    const m = scorePose(demoPose(t), cfg);
    assert.ok(Math.abs(m.head_forward_deg) < cfg.head_forward_deg - cfg.hysteresis_margin_deg, `t=${t.toFixed(1)}`);
  }
});

test("the needle glides: a zone change sweeps instead of jumping, frame-rate independent", () => {
  assert.equal(glideAngle(null, 90, 0.016), 90, "first frame starts at the target");
  assert.equal(glideAngle(30, 150, 0), 30, "no time, no move");
  let a = 30;
  const steps = [];
  for (let i = 0; i < 60; i++) {
    const next = glideAngle(a, 150, 1 / 60);
    steps.push(next - a);
    assert.ok(next >= a && next <= 150, "monotone, never overshoots");
    a = next;
  }
  assert.ok(Math.max(...steps) < 8, `largest step ${Math.max(...steps)} deg per 60 fps frame (was 120)`);
  assert.ok(a > 140, "within 10 degrees of the target after a second");
  // The same elapsed time gives the same angle at 15 fps and at 60 fps.
  let b = 30;
  for (let i = 0; i < 15; i++) b = glideAngle(b, 150, 1 / 15);
  assert.ok(Math.abs(a - b) < 0.01, `${a} vs ${b}`);
});
