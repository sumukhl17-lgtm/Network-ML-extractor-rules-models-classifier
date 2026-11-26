const REQUIRED_CLASSIFIERS = ["network", "text", "video", "audio"];

export function readinessState(response) {
  const checks = response?.checks;
  return response?.status === "ready"
    && REQUIRED_CLASSIFIERS.every((name) => checks?.[name]?.ok === true)
    ? "ready"
    : "not_ready";
}
