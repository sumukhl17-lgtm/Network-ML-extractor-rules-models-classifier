import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { formatTime, num } from "../format.js";
import ErrorNotice from "../components/ErrorNotice.jsx";
import ConfirmDialog from "../components/ConfirmDialog.jsx";
import ComparePanel from "../components/ComparePanel.jsx";
import { Tag } from "../components/Chip.jsx";

function servingLine(model) {
  const s = model.serving || {};
  if (s.source === "registry") return `Serving registry version ${s.version}.`;
  if (s.source === "local") return `Serving a local file (${s.version}), not a registry version.`;
  if (s.source === "repo-file") return `Serving ${s.file} from the repository, not from the registry.`;
  if (s.source === "error") return `The live model could not be loaded: ${s.error}`;
  return "Live model unknown.";
}

function RunsSection() {
  const [runs, setRuns] = useState(null);
  const [error, setError] = useState(null);
  const [picked, setPicked] = useState([]);

  useEffect(() => {
    api.networkRuns().then(setRuns).catch(setError);
  }, []);

  if (error) return <ErrorNotice error={error} />;
  if (!runs) return <p className="muted">Loading training runs.</p>;
  if (!runs.found || runs.runs.length === 0) {
    return (
      <p className="muted">
        No runs found in the experiment <code>{runs.experiment}</code>. Run <code>train_compare.py</code> with{" "}
        <code>MLFLOW_TRACKING_URI</code> set and they will appear here.
      </p>
    );
  }

  const label = (r) => `${(r.run_name || "run").split("-")[0]} ${r.run_id.slice(0, 6)}`;
  const chosen = runs.runs.filter((r) => picked.includes(r.run_id));
  const names = [...new Set(chosen.flatMap((r) => Object.keys(r.metrics)))];
  const metrics = Object.fromEntries(
    names.map((n) => [n, Object.fromEntries(chosen.map((r) => [r.run_id, r.metrics[n] ?? null]))]),
  );
  const toggle = (id) => setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id]));

  return (
    <div>
      <h4>Training runs</h4>
      <p className="muted small">
        Unknown recall is the share of truly unknown flows the model caught. It is the number to watch.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>
                <span className="sr-only">Select</span>
              </th>
              <th>Run</th>
              <th>Started</th>
              <th className="num">Unknown recall</th>
              <th className="num">Unknown F1</th>
              <th className="num">ROC-AUC</th>
            </tr>
          </thead>
          <tbody>
            {runs.runs.map((r) => (
              <tr key={r.run_id}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Select ${label(r)}`}
                    checked={picked.includes(r.run_id)}
                    onChange={() => toggle(r.run_id)}
                  />
                </td>
                <td>{label(r)}</td>
                <td>{formatTime(r.started_at)}</td>
                <td className="num">{r.metrics.recall_Unknown === undefined ? "not logged" : num(r.metrics.recall_Unknown, 3)}</td>
                <td className="num">{r.metrics.f1_Unknown === undefined ? "not logged" : num(r.metrics.f1_Unknown, 3)}</td>
                <td className="num">{r.metrics.roc_auc === undefined ? "not logged" : num(r.metrics.roc_auc, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {chosen.length >= 2 && (
        <ComparePanel
          title="Run comparison"
          columns={chosen.map((r) => ({ id: r.run_id, label: label(r) }))}
          metrics={metrics}
        />
      )}
      {chosen.length < 2 && <p className="muted small">Tick two or more runs to compare precision, recall and F1.</p>}
    </div>
  );
}

function ModelCard({ model, onPromote, onRollback, onReload, busy }) {
  const [picked, setPicked] = useState([]);
  const [compare, setCompare] = useState(null);
  const [compareError, setCompareError] = useState(null);
  const s = model.serving || {};
  const mismatch = s.source === "registry" && model.champion && String(s.version) !== String(model.champion);

  const toggle = (v) => {
    setCompare(null);
    setPicked((p) => (p.includes(v) ? p.filter((x) => x !== v) : [...p, v]));
  };

  async function runCompare() {
    setCompareError(null);
    try {
      setCompare(await api.compare(model.name, picked));
    } catch (e) {
      setCompare(null);
      setCompareError(e);
    }
  }

  return (
    <section className="model" aria-label={`Model ${model.name}`}>
      <header className="model__head">
        <h3>{model.name}</h3>
        <Tag>{model.branch}</Tag>
        {model.champion && <Tag tone="good">champion: version {model.champion}</Tag>}
      </header>
      <p>{servingLine(model)}</p>
      {s.note && <p className="callout">{s.note}</p>}
      {model.branch === "text" && s.real_data_validated === false && (
        <p className="callout callout--unknown" role="status">
          <strong>DEMO MODEL</strong> – Not validated on real labelled documents.
          {s.evaluation_note ? ` ${s.evaluation_note}` : " Synthetic/demo evaluation must not be treated as real-world accuracy."}
        </p>
      )}
      {model.branch === "text" && s.source === "local" && (
        <p className="callout">
          The backend is using its local model file. Promoting a version here will not change what it serves until{" "}
          <code>MLFLOW_TRACKING_URI</code> is set for the backend and the model is reloaded.
        </p>
      )}
      {mismatch && (
        <p className="callout callout--unknown">
          The live model (version {s.version}) is not the champion (version {model.champion}). Use Reload live model.
        </p>
      )}

      {model.registered ? (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>
                    <span className="sr-only">Select</span>
                  </th>
                  <th>Version</th>
                  <th>Registered</th>
                  <th className="num">F1 (macro)</th>
                  <th className="num">Accuracy</th>
                  <th>Status</th>
                  <th>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {model.versions.map((v) => (
                  <tr key={v.version} className={v.is_champion ? "is-champion" : ""}>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Select version ${v.version}`}
                        checked={picked.includes(v.version)}
                        onChange={() => toggle(v.version)}
                      />
                    </td>
                    <td className="num">{v.version}</td>
                    <td>{formatTime(v.created_at)}</td>
                    <td className="num">{v.metrics.f1_macro === undefined ? "not logged" : num(v.metrics.f1_macro, 3)}</td>
                    <td className="num">{v.metrics.accuracy === undefined ? "not logged" : num(v.metrics.accuracy, 3)}</td>
                    <td>{v.is_champion ? <Tag tone="good">champion</Tag> : ""}</td>
                    <td>
                      {model.can_promote && !v.is_champion && (
                        <button
                          className="btn btn--secondary btn--small"
                          disabled={busy}
                          onClick={() => onPromote(model, v.version)}
                        >
                          Promote version {v.version}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="actions">
            <button className="btn btn--secondary" disabled={picked.length < 2 || busy} onClick={runCompare}>
              Compare selected versions
            </button>
            {model.can_promote && (
              <>
                <button className="btn btn--secondary" disabled={busy} onClick={() => onRollback(model)}>
                  Roll back
                </button>
                <button className="btn btn--secondary" disabled={busy} onClick={() => onReload(model)}>
                  Reload live model
                </button>
              </>
            )}
          </div>
          <ErrorNotice error={compareError} onDismiss={() => setCompareError(null)} />
          {compare && (
            <ComparePanel
              title="Version comparison"
              columns={compare.versions.map((v) => ({
                id: v,
                label: `Version ${v}${String(compare.champion) === String(v) ? " (champion)" : ""}`,
              }))}
              metrics={compare.metrics}
            />
          )}
        </>
      ) : (
        <div className="empty">
          <p>No versions are registered under this name yet.</p>
          {model.branch === "text" && (
            <p className="muted small">
              Register one with <code>python -m text.train --data-dir data/sample --register force</code>.
            </p>
          )}
        </div>
      )}

      {model.branch === "network" && <RunsSection />}
    </section>
  );
}

export default function ModelsPage({ actor, onChanged }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [banner, setBanner] = useState(null);
  const [dialog, setDialog] = useState(null); // {kind, model, version}
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setData(await api.registryModels());
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  function describe(result, verb, name) {
    let text = `Version ${result.champion} of ${name} is now the champion (${verb}).`;
    if (result.reload?.reloaded) {
      text += ` The live model was reloaded and is serving version ${result.reload.serving.version}.`;
    } else if (result.reload?.error) {
      text += ` Reloading the live model failed: ${result.reload.error}`;
    }
    return text;
  }

  async function confirm() {
    const { kind, model, version } = dialog;
    setBusy(true);
    try {
      const body = { confirm: true, actor: actor || "anonymous", reload: true };
      const result =
        kind === "promote"
          ? await api.promote(model.name, { ...body, version })
          : await api.rollback(model.name, body);
      await load(); // refresh the list first, so the message never appears next to stale data
      setBanner({ tone: "ok", text: describe(result, kind === "promote" ? "promoted" : "rolled back", model.name) });
      onChanged();
    } catch (e) {
      setBanner({ tone: "error", error: e });
    } finally {
      setBusy(false);
      setDialog(null);
    }
  }

  async function reload(model) {
    setBusy(true);
    try {
      const r = await api.reload(model.name);
      await load();
      setBanner({
        tone: "ok",
        text: r.reloaded ? `Reloaded ${model.name}: serving version ${r.serving.version}.` : r.reason,
      });
    } catch (e) {
      setBanner({ tone: "error", error: e });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h2>Models</h2>
      <p className="muted">
        Each retrain adds a version. The champion is the version the backend serves. Moving the champion label changes
        the live model without redeploying any code.
      </p>
      <ErrorNotice error={error} />
      {banner && banner.tone === "ok" && (
        <div className="notice notice--ok" role="status">
          <div className="notice__body">{banner.text}</div>
          <button className="btn btn--quiet" onClick={() => setBanner(null)}>
            Dismiss
          </button>
        </div>
      )}
      {banner && banner.tone === "error" && <ErrorNotice error={banner.error} onDismiss={() => setBanner(null)} />}

      {data && !data.available && (
        <div className="empty">
          <p>
            <strong>The model registry is not available.</strong> {data.reason}
          </p>
          <p className="muted small">
            Start it with <code>./mlflow/start_mlflow.sh</code>, then set <code>MLFLOW_TRACKING_URI</code> for the
            backend and restart it.
          </p>
        </div>
      )}
      {data?.available &&
        data.models.map((m) => (
          <ModelCard
            key={m.name}
            model={m}
            busy={busy}
            onPromote={(model, version) => setDialog({ kind: "promote", model, version })}
            onRollback={(model) => setDialog({ kind: "rollback", model })}
            onReload={reload}
          />
        ))}

      <ConfirmDialog
        open={!!dialog}
        busy={busy}
        title={dialog?.kind === "promote" ? `Promote version ${dialog?.version}?` : "Roll back to the previous champion?"}
        confirmLabel={dialog?.kind === "promote" ? "Promote" : "Roll back"}
        checkLabel="I understand this changes the model that answers requests."
        onConfirm={confirm}
        onCancel={() => !busy && setDialog(null)}
      >
        {dialog && (
          <>
            <p>
              {dialog.kind === "promote" ? (
                <>
                  <strong>{dialog.model.name}</strong> will use version {dialog.version} instead of version{" "}
                  {dialog.model.champion || "none"}.
                </>
              ) : (
                <>
                  <strong>{dialog.model.name}</strong> will go back to the champion it had before the last promotion
                  (version {dialog.model.champion} is current).
                </>
              )}
            </p>
            <p className="muted">
              The live model reloads right away. This is recorded as <strong>{actor || "anonymous"}</strong> in the
              activity log.
            </p>
          </>
        )}
      </ConfirmDialog>
    </div>
  );
}
