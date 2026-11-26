// Small pure helpers (no React), so they can be tested with `npm test`.

export const isUnknown = (label) => String(label ?? "").toLowerCase() === "unknown";

export function pct(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  return Number(value).toLocaleString("en", { maximumFractionDigits: digits });
}

export function formatTime(iso) {
  if (!iso) return "n/a";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  return d.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
}

const PROTOCOLS = { 1: "ICMP", 6: "TCP", 17: "UDP" };
export const protocolName = (n) => (n === null || n === undefined ? "n/a" : PROTOCOLS[n] || String(n));

// Order and wording of the metrics shown when two model versions are compared.
const FIXED_LABELS = {
  f1_macro: "F1 (macro average)",
  accuracy: "Accuracy",
  cv_f1_macro_mean: "F1, cross-validated",
  roc_auc: "ROC-AUC",
  leave_one_capture_out_mean_f1: "F1, leave-one-capture-out",
};

export function metricLabel(name) {
  if (FIXED_LABELS[name]) return FIXED_LABELS[name];
  const m = /^(precision|recall|f1)_(.+)$/.exec(name);
  if (m) {
    const kind = { precision: "Precision", recall: "Recall", f1: "F1" }[m[1]];
    return `${kind}, ${m[2]}`;
  }
  return name.replace(/_/g, " ");
}

const RANK = (name) => {
  if (/^recall_unknown$/i.test(name)) return 0; // the number the project cares most about
  if (name === "f1_macro") return 2; // checked before the per-class pattern, which would also match it
  if (/^(precision|recall|f1)_/.test(name)) return 1;
  if (name === "accuracy") return 3;
  return 4;
};

export function sortMetricNames(names) {
  return [...names].sort((a, b) => RANK(a) - RANK(b) || a.localeCompare(b));
}

// metrics: {metricName: {columnId: value|null}} -> the column id with the highest value, or null if no clear best
export function bestColumn(valuesByColumn) {
  const entries = Object.entries(valuesByColumn).filter(([, v]) => typeof v === "number");
  if (entries.length < 2) return null;
  const max = Math.max(...entries.map(([, v]) => v));
  const winners = entries.filter(([, v]) => v === max);
  return winners.length === entries.length ? null : winners[0][0];
}

export const REASONS = {
  ok: "The model is confident about this category.",
  low_confidence: "The model was not confident enough, so the answer is unknown.",
  too_little_text: "There was too little readable text to classify (a scanned PDF has no text layer).",
};
