// Classification history, kept in this browser's localStorage so it survives a page refresh.
// Every access is wrapped: storage can be full, disabled, or hold corrupt data, and none of that may break the app.

const KEY = "classification-console.history.v1";
const NAME_KEY = "classification-console.actor";
export const MAX_ENTRIES = 30;
export const MAX_STORED_FLOWS = 200; // a 20,000-flow result would overflow the ~5 MB browser limit

function store() {
  try {
    return globalThis.localStorage ?? null;
  } catch {
    return null;
  }
}

export function loadHistory() {
  const s = store();
  if (!s) return [];
  try {
    const parsed = JSON.parse(s.getItem(KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function trim(kind, result) {
  if (kind !== "network" || !Array.isArray(result.results)) return { result, truncated: false };
  if (result.results.length <= MAX_STORED_FLOWS) return { result, truncated: false };
  return { result: { ...result, results: result.results.slice(0, MAX_STORED_FLOWS) }, truncated: true };
}

export function addHistory(kind, result, now = new Date()) {
  const { result: stored, truncated } = trim(kind, result);
  const entry = {
    id: `${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`,
    kind,
    filename: result.filename || "",
    at: now.toISOString(),
    truncated,
    totalFlows: kind === "network" ? result.summary?.n_flows ?? null : null,
    result: stored,
  };
  let entries = [entry, ...loadHistory()].slice(0, MAX_ENTRIES);
  const s = store();
  if (!s) return entries;
  // If the browser refuses (quota), drop the oldest entries until it fits instead of losing the new one.
  while (entries.length > 0) {
    try {
      s.setItem(KEY, JSON.stringify(entries));
      return entries;
    } catch {
      entries = entries.slice(0, -1);
    }
  }
  return entries;
}

export function clearHistory() {
  try {
    store()?.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

export function loadActor() {
  try {
    return store()?.getItem(NAME_KEY) || "";
  } catch {
    return "";
  }
}

export function saveActor(name) {
  try {
    store()?.setItem(NAME_KEY, name);
  } catch {
    /* ignore */
  }
}
