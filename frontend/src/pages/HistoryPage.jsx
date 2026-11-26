import { useState } from "react";
import { clearHistory } from "../storage.js";
import { formatTime, num, pct } from "../format.js";
import NetworkResults from "../components/NetworkResults.jsx";
import TextResult from "../components/TextResult.jsx";
import VideoResult from "../components/VideoResult.jsx";
import AudioResult from "../components/AudioResult.jsx";
import { LabelChip } from "../components/Chip.jsx";

const KIND_LABELS = {
  network: "Network",
  text: "Document",
  video: "Video",
  audio: "Audio",
};

function summary(entry) {
  const result = entry.result || {};

  switch (entry.kind) {
    case "network": {
      const s = result.summary || {};
      return <span>{num(s.n_flows, 0)} flows, {num(s.n_unknown, 0)} unknown</span>;
    }
    case "text":
      return <LabelChip label={result.category || "Unknown"} />;
    case "video":
    case "audio":
      return (
        <span>
          <LabelChip label={result.category || "Unknown"} />
          {" "}{pct(result.confidence)}
        </span>
      );
    default:
      return <span>Classification result</span>;
  }
}

function HistoryResult({ entry }) {
  switch (entry.kind) {
    case "network":
      return (
        <NetworkResults
          data={entry.result}
          truncatedFrom={entry.truncated ? entry.totalFlows : null}
        />
      );
    case "text":
      return <TextResult data={entry.result} />;
    case "video":
      return <VideoResult data={entry.result} />;
    case "audio":
      return <AudioResult data={entry.result} />;
    default:
      return <p className="muted">This result type is no longer supported.</p>;
  }
}

export default function HistoryPage({ entries, onCleared }) {
  const [openId, setOpenId] = useState(null);

  return (
    <div className="page">
      <div className="page__head">
        <h2>History</h2>
        {entries.length > 0 && (
          <button
            className="btn btn--secondary"
            onClick={() => {
              if (window.confirm("Remove all saved results from this browser?")) {
                clearHistory();
                setOpenId(null);
                onCleared();
              }
            }}
          >
            Clear history
          </button>
        )}
      </div>

      <p className="muted">
        Results are saved in this browser only, so they stay after you refresh the page. They are not stored on the server.
      </p>

      {entries.length === 0 ? (
        <div className="empty">
          <p>Nothing here yet. Classify a file and its result will be saved.</p>
        </div>
      ) : (
        <ul className="history">
          {entries.map((e) => (
            <li key={e.id}>
              <button
                className="history__row"
                aria-expanded={openId === e.id}
                onClick={() => setOpenId(openId === e.id ? null : e.id)}
              >
                <span className="history__name">{e.filename || "(unnamed)"}</span>
                <span className="tag">{KIND_LABELS[e.kind] || e.kind || "Unknown"}</span>
                <span className="history__summary">{summary(e)}</span>
                <span className="muted small">{formatTime(e.at)}</span>
              </button>

              {openId === e.id && (
                <div className="history__detail">
                  <HistoryResult entry={e} />
                </div>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
