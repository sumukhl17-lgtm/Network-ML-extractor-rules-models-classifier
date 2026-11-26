import test from "node:test";
import assert from "node:assert/strict";
import { isUnknown, pct, num, metricLabel, sortMetricNames, bestColumn, protocolName, formatTime } from "./format.js";

test("isUnknown absorbs the Unknown / unknown casing difference between branches", () => {
  assert.equal(isUnknown("Unknown"), true);
  assert.equal(isUnknown("unknown"), true);
  assert.equal(isUnknown("HTTPS"), false);
  assert.equal(isUnknown(undefined), false);
});

test("pct and num handle missing values", () => {
  assert.equal(pct(0.926), "93%");
  assert.equal(pct(0.9262, 1), "92.6%");
  assert.equal(pct(null), "n/a");
  assert.equal(num(undefined), "n/a");
  assert.equal(num(1234.5678), "1,234.57");
});

test("metric labels", () => {
  assert.equal(metricLabel("recall_Unknown"), "Recall, Unknown");
  assert.equal(metricLabel("f1_Known"), "F1, Known");
  assert.equal(metricLabel("f1_macro"), "F1 (macro average)");
  assert.equal(metricLabel("some_new_metric"), "some new metric");
});

test("Unknown recall sorts first, then per-class metrics, then the rest", () => {
  const order = sortMetricNames(["roc_auc", "accuracy", "f1_macro", "precision_Known", "recall_Unknown"]);
  assert.deepEqual(order, ["recall_Unknown", "precision_Known", "f1_macro", "accuracy", "roc_auc"]);
});

test("bestColumn", () => {
  assert.equal(bestColumn({ 1: 0.9, 2: 0.95 }), "2");
  assert.equal(bestColumn({ 1: 0.9, 2: 0.9 }), null); // a tie is not a winner
  assert.equal(bestColumn({ 1: 0.9, 2: null }), null); // nothing to compare against
});

test("protocolName and formatTime", () => {
  assert.equal(protocolName(6), "TCP");
  assert.equal(protocolName(17), "UDP");
  assert.equal(protocolName(99), "99");
  assert.equal(formatTime(null), "n/a");
  assert.equal(formatTime("garbage"), "garbage");
});
