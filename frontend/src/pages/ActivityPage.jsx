import { useEffect, useState } from "react";
import { api } from "../api.js";
import { formatTime } from "../format.js";
import ErrorNotice from "../components/ErrorNotice.jsx";

export default function ActivityPage({ refreshKey }) {
  const [entries, setEntries] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    api
      .activity()
      .then((r) => !cancelled && (setEntries(r.entries), setError(null)))
      .catch((e) => !cancelled && setError(e));
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  return (
    <div className="page">
      <h2>Activity</h2>
      <p className="muted">Every promotion and rollback of a model version, newest first.</p>
      <ErrorNotice error={error} />
      {entries && entries.length === 0 && (
        <div className="empty">
          <p>No model changes recorded yet. Promote or roll back a version on the Models page and it will appear here.</p>
        </div>
      )}
      {entries && entries.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>When</th>
                <th>Who</th>
                <th>Action</th>
                <th>Model</th>
                <th>Changed from</th>
                <th>Changed to</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={`${e.timestamp}-${i}`}>
                  <td>{formatTime(e.timestamp)}</td>
                  <td>{e.actor}</td>
                  <td>{e.action === "rollback" ? "Rolled back" : "Promoted"}</td>
                  <td>{e.model}</td>
                  <td className="num">{e.from_version ? `version ${e.from_version}` : "no champion"}</td>
                  <td className="num">{`version ${e.to_version}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
