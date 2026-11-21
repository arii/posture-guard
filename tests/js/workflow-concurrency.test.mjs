import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const featurePush =
  "github.event_name == 'push' && startsWith(github.ref, 'refs/heads/') && github.ref != 'refs/heads/main'";

for (const name of ["live-site", "demo-video"]) {
  test(`${name}: cancel only feature pushes and isolate protected runs`, () => {
    const workflow = readFileSync(new URL(`../../.github/workflows/${name}.yml`, import.meta.url), "utf8");
    const block = workflow.split("\nconcurrency:\n")[1]?.split("\njobs:\n")[0];
    assert.equal(
      block,
      `  group: \u0024{{ github.workflow }}-\u0024{{ ${featurePush} && github.ref || github.run_id }}\n  cancel-in-progress: \u0024{{ ${featurePush} }}\n`,
    );
  });
}
