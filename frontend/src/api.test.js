import test from "node:test";
import assert from "node:assert/strict";
import { api, ApiError } from "./api.js";

test("readiness request uses /ready and preserves its 503 response body", async () => {
  const originalFetch = globalThis.fetch;
  let requestUrl;
  globalThis.fetch = async (url) => {
    requestUrl = url;
    return {
      ok: false,
      status: 503,
      json: async () => ({
        status: "not_ready",
        checks: { network: { ok: true }, text: { ok: false, error: "model missing" } },
      }),
    };
  };

  try {
    const response = await api.readiness();
    assert.match(requestUrl, /\/ready$/);
    assert.equal(response.status, "not_ready");
    assert.equal(response.checks.text.ok, false);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("unreachable-backend hint uses the project-root startup command", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new Error("connection refused");
  };

  try {
    await assert.rejects(api.readiness(), (error) => {
      assert.ok(error instanceof ApiError);
      assert.equal(
        error.details.hint,
        "From the project root, run: uvicorn backend.platform_api.main:app --reload --port 8000",
      );
      return true;
    });
  } finally {
    globalThis.fetch = originalFetch;
  }
});
