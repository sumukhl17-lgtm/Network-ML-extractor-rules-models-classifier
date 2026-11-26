// Every call to the FastAPI backend goes through here.
// The backend always answers errors as {"error": {"code", "message", "details"}} and we turn that into ApiError.

export const API_URL = import.meta.env?.VITE_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  constructor(status, code, message, details = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status; // 0 = the backend could not be reached at all
    this.code = code;
    this.details = details;
  }
}

async function request(path, options = {}, acceptedStatuses = []) {
  let response;
  try {
    response = await fetch(`${API_URL}${path}`, options);
  } catch {
    throw new ApiError(
      0,
      "BACKEND_UNREACHABLE",
      `Cannot reach the backend at ${API_URL}.`,
      { hint: "From the project root, run: uvicorn backend.platform_api.main:app --reload --port 8000" },
    );
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    /* not JSON */
  }
  if (!response.ok && !acceptedStatuses.includes(response.status)) {
    const e = body?.error;
    throw new ApiError(
      response.status,
      e?.code || "ERROR",
      e?.message || `The backend answered ${response.status}.`,
      e?.details || {},
    );
  }
  return body;
}

function upload(path, file) {
  const form = new FormData();
  form.append("file", file);
  return request(path, { method: "POST", body: form });
}

function postJson(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

const enc = encodeURIComponent;

export const api = {
  readiness: () => request("/ready", {}, [503]),
  classifyNetwork: (file) => upload("/classify/network", file),
  classifyText: (file) => upload("/classify/text", file),
  classifyVideo: (file) => upload("/classify/video", file),
  classifyAudio: (file) => upload("/classify/audio", file),
  registryModels: () => request("/registry/models"),
  compare: (name, versions) => request(`/registry/models/${enc(name)}/compare?versions=${enc(versions.join(","))}`),
  promote: (name, body) => postJson(`/registry/models/${enc(name)}/promote`, body),
  rollback: (name, body) => postJson(`/registry/models/${enc(name)}/rollback`, body),
  reload: (name) => request(`/registry/models/${enc(name)}/reload`, { method: "POST" }),
  activity: () => request("/registry/activity"),
  networkRuns: () => request("/registry/network/runs"),
};
