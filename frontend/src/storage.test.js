import test, { beforeEach } from "node:test";
import assert from "node:assert/strict";
import { addHistory, loadHistory, clearHistory, MAX_ENTRIES, MAX_STORED_FLOWS, loadActor, saveActor } from "./storage.js";

function fakeStorage({ limit = Infinity } = {}) {
  const data = new Map();
  return {
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => {
      if (String(v).length > limit) throw new Error("QuotaExceededError");
      data.set(k, String(v));
    },
    removeItem: (k) => data.delete(k),
  };
}

beforeEach(() => {
  globalThis.localStorage = fakeStorage();
});

const textResult = (name) => ({ filename: name, category: "invoice", confidence: 0.9 });
const networkResult = (n) => ({
  filename: "f.csv",
  summary: { n_flows: n },
  results: Array.from({ length: n }, (_, i) => ({ index: i })),
});

test("history survives a reload (a new read sees what was saved)", () => {
  addHistory("text", textResult("a.txt"));
  addHistory("text", textResult("b.txt"));
  const entries = loadHistory();
  assert.deepEqual(entries.map((e) => e.filename), ["b.txt", "a.txt"]);
});

test("history is capped", () => {
  for (let i = 0; i < MAX_ENTRIES + 5; i++) addHistory("text", textResult(`f${i}.txt`));
  assert.equal(loadHistory().length, MAX_ENTRIES);
  assert.equal(loadHistory()[0].filename, `f${MAX_ENTRIES + 4}.txt`);
});

test("big network results are truncated and say so", () => {
  addHistory("network", networkResult(MAX_STORED_FLOWS + 50));
  const [entry] = loadHistory();
  assert.equal(entry.result.results.length, MAX_STORED_FLOWS);
  assert.equal(entry.truncated, true);
  assert.equal(entry.totalFlows, MAX_STORED_FLOWS + 50);
});

test("small network results are kept whole", () => {
  addHistory("network", networkResult(5));
  assert.equal(loadHistory()[0].truncated, false);
});

test("a full browser store drops the oldest entries instead of failing", () => {
  globalThis.localStorage = fakeStorage({ limit: 700 });
  for (let i = 0; i < 10; i++) addHistory("text", textResult(`f${i}.txt`));
  const entries = loadHistory();
  assert.ok(entries.length > 0 && entries.length < 10);
  assert.equal(entries[0].filename, "f9.txt"); // the newest is always kept
});

test("corrupt or missing storage does not throw", () => {
  globalThis.localStorage = fakeStorage();
  globalThis.localStorage.setItem("classification-console.history.v1", "{not json");
  assert.deepEqual(loadHistory(), []);
  delete globalThis.localStorage;
  assert.deepEqual(loadHistory(), []);
  assert.doesNotThrow(() => addHistory("text", textResult("x.txt")));
  assert.doesNotThrow(() => clearHistory());
});

test("clearHistory and actor name", () => {
  addHistory("text", textResult("a.txt"));
  clearHistory();
  assert.deepEqual(loadHistory(), []);
  assert.equal(loadActor(), "");
  saveActor("Ullas");
  assert.equal(loadActor(), "Ullas");
});
