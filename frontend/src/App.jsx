import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { readinessState } from "./readiness.js";
import { loadActor, loadHistory, saveActor } from "./storage.js";
import ClassifyPage from "./pages/ClassifyPage.jsx";
import HistoryPage from "./pages/HistoryPage.jsx";
import ModelsPage from "./pages/ModelsPage.jsx";
import ActivityPage from "./pages/ActivityPage.jsx";

const PAGES = [
  { id: "classify", label: "Classify" },
  { id: "history", label: "History" },
  { id: "models", label: "Models" },
  { id: "activity", label: "Activity" },
];

const fromHash = () => {
  const id = window.location.hash.replace("#", "");
  return PAGES.some((p) => p.id === id) ? id : "classify";
};

function useBackendStatus() {
  const [status, setStatus] = useState("checking");
  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .readiness()
        .then((response) => alive && setStatus(readinessState(response)))
        .catch(() => alive && setStatus("unavailable"));
    check();
    const timer = setInterval(check, 10000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);
  return status;
}

export default function App() {
  const [page, setPage] = useState(fromHash());
  const [history, setHistory] = useState(loadHistory());
  const [actor, setActor] = useState(loadActor());
  const [activityKey, setActivityKey] = useState(0);
  const backend = useBackendStatus();

  useEffect(() => {
    const onHash = () => setPage(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const refreshHistory = useCallback(() => setHistory(loadHistory()), []);

  return (
    <div className="shell">
      <aside className="rail">
        <h1 className="rail__title">Classification console</h1>
        <nav aria-label="Main">
          {PAGES.map((p) => (
            <a
              key={p.id}
              href={`#${p.id}`}
              className={`rail__link${page === p.id ? " is-on" : ""}`}
              aria-current={page === p.id ? "page" : undefined}
            >
              {p.label}
              {p.id === "history" && history.length > 0 && <span className="rail__count">{history.length}</span>}
            </a>
          ))}
        </nav>
        <div className="rail__foot">
          <label className="field">
            <span>Your name</span>
            <input
              value={actor}
              placeholder="Shown in the activity log"
              onChange={(e) => {
                setActor(e.target.value);
                saveActor(e.target.value);
              }}
            />
          </label>
          <p className={`status status--${backend}`} role="status">
            <span className="status__dot" />
            {backend === "checking" && "Checking backend readiness"}
            {backend === "unavailable" && "Backend unavailable"}
            {backend === "not_ready" && "Backend running, but one or more classifiers are not ready"}
            {backend === "ready" && "All four classifiers ready"}
          </p>
        </div>
      </aside>

      <main className="main">
        {page === "classify" && <ClassifyPage onSaved={refreshHistory} />}
        {page === "history" && <HistoryPage entries={history} onCleared={refreshHistory} />}
        {page === "models" && <ModelsPage actor={actor} onChanged={() => setActivityKey((k) => k + 1)} />}
        {page === "activity" && <ActivityPage refreshKey={activityKey} />}
      </main>
    </div>
  );
}
