import { useMemo, useState } from "react";
import { LabelChip } from "./Chip.jsx";
import { isUnknown, num, pct, protocolName } from "../format.js";

const SHOW_LIMIT = 500;
const FILTERS = [
  { id: "all", label: "All flows" },
  { id: "unknown", label: "Unknown only" },
  { id: "advisory", label: "With a model opinion" },
];

function answeredBy(r) {
  if (r.source === "rule") return "Port rule";
  if (r.source === "default") return "No rule matched";
  if (r.source === "ml") return "Model";
  return r.source || "n/a";
}

function modelOpinion(r) {
  if (r.source === "default" && r.ml_label) {
    return `Model suggests ${r.ml_label} (${pct(r.ml_confidence)} sure)`;
  }
  if (r.source === "ml") return `Model answer (${pct(r.confidence)} sure)`;
  return "";
}

function LabelBar({ byLabel, total }) {
  const entries = Object.entries(byLabel).sort((a, b) => b[1] - a[1]);
  return (
    <div>
      <div className="bar" role="img" aria-label={entries.map(([l, n]) => `${l}: ${n}`).join(", ")}>
        {entries.map(([label, n]) => (
          <span
            key={label}
            className={`bar__part ${isUnknown(label) ? "bar__part--unknown" : "bar__part--known"}`}
            style={{ flexGrow: n }}
            title={`${label}: ${n}`}
          />
        ))}
      </div>
      <ul className="legend">
        {entries.map(([label, n]) => (
          <li key={label}>
            <span className={`swatch ${isUnknown(label) ? "swatch--unknown" : "swatch--known"}`} />
            <strong>{label}</strong> {n} of {total}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function NetworkResults({ data, truncatedFrom }) {
  const [filter, setFilter] = useState("all");
  const rows = useMemo(() => {
    const all = data.results || [];
    if (filter === "unknown") return all.filter((r) => r.is_unknown);
    if (filter === "advisory") return all.filter((r) => r.advisory);
    return all;
  }, [data, filter]);
  const s = data.summary;
  const shown = rows.slice(0, SHOW_LIMIT);
  const hasProtocol = (data.results || []).some((r) => r.protocol !== null && r.protocol !== undefined);
  const columns = hasProtocol ? 8 : 7;

  return (
    <section className="result" aria-label="Network classification result">
      <header className="result__head">
        <h3>{data.filename}</h3>
        <p className="muted small">
          Read as {data.input_kind === "pcap" ? "a packet capture" : "a table of flows"}; {s.n_flows} flows.
        </p>
      </header>

      <LabelBar byLabel={s.by_label} total={s.n_flows} />

      <div className="callout">
        <p>
          <strong>{s.by_source.rule || 0}</strong> flows matched a port rule (HTTP, HTTPS, DNS or SSH).{" "}
          <strong>{s.n_unknown}</strong> had no matching rule and are labelled Unknown.
        </p>
        {s.n_advisory > 0 && (
          <p className="muted">
            {data.mode === "ml_overrides_unmatched"
              ? "The server is set to let the model answer for unmatched flows. Treat those answers as hints."
              : "Unmatched flows also carry the model's opinion. It is shown as a hint, not a verdict: in our evaluation the model did not detect unfamiliar traffic that came from a different generator."}
          </p>
        )}
      </div>

      <div className="toolbar" role="group" aria-label="Filter flows">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            className={`btn btn--toggle${filter === f.id ? " is-on" : ""}`}
            aria-pressed={filter === f.id}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Flow</th>
              <th>Ports (source to destination)</th>
              {hasProtocol && <th>Protocol</th>}
              <th className="num">Packets</th>
              <th className="num">Bytes</th>
              <th>Label</th>
              <th>Answered by</th>
              <th>Model opinion</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.index}>
                <td className="num">{r.index + 1}</td>
                <td className="num">
                  {r.src_port} to {r.dst_port}
                </td>
                {hasProtocol && <td>{protocolName(r.protocol)}</td>}
                <td className="num">{num(r.packet_count, 0)}</td>
                <td className="num">{num(r.total_bytes, 0)}</td>
                <td>
                  <LabelChip label={r.label} advisory={r.source === "ml"} />
                </td>
                <td>{answeredBy(r)}</td>
                <td className={`wrap ${r.advisory ? "advisory-text" : "muted"}`}>{modelOpinion(r) || "none needed"}</td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr>
                <td colSpan={columns} className="muted empty-row">
                  No flows match this filter.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {rows.length > SHOW_LIMIT && (
        <p className="muted small">
          Showing the first {SHOW_LIMIT} of {rows.length} matching flows.
        </p>
      )}
      {truncatedFrom && (
        <p className="muted small">
          Saved history keeps only the first {data.results.length} of {truncatedFrom} flows.
        </p>
      )}
    </section>
  );
}
