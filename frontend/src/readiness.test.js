import test from "node:test";
import assert from "node:assert/strict";
import { readinessState } from "./readiness.js";

const allReady = {
  network: { ok: true },
  text: { ok: true },
  video: { ok: true },
  audio: { ok: true },
};

test("readiness requires all four classifiers to be explicitly ready", () => {
  assert.equal(
    readinessState({ status: "ready", checks: allReady }),
    "ready",
  );

  for (const name of Object.keys(allReady)) {
    const checks = {
      ...allReady,
      [name]: { ok: false, error: "not ready" },
    };
    assert.equal(
      readinessState({ status: "ready", checks }),
      "not_ready",
      `${name} must be ready`,
    );
  }
});

test("missing or inconsistent readiness data never appears ready", () => {
  assert.equal(readinessState(null), "not_ready");
  assert.equal(
    readinessState({ status: "ready", checks: { network: { ok: true }, text: { ok: true } } }),
    "not_ready",
  );
  assert.equal(
    readinessState({ status: "not_ready", checks: allReady }),
    "not_ready",
  );
});
