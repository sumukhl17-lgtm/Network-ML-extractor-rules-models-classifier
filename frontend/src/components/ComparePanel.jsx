import { bestColumn, metricLabel, num, sortMetricNames } from "../format.js";

// columns: [{id, label}]   metrics: {metricName: {columnId: number|null}}
export default function ComparePanel({ title, columns, metrics, note }) {
  const names = sortMetricNames(Object.keys(metrics));
  return (
    <section className="compare" aria-label={title}>
      <h4>{title}</h4>
      {note && <p className="muted small">{note}</p>}
      {names.length === 0 ? (
        <p className="muted">These versions have no logged metrics to compare.</p>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                {columns.map((c) => (
                  <th key={c.id} className="num">
                    {c.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {names.map((name) => {
                const best = bestColumn(metrics[name]);
                return (
                  <tr key={name}>
                    <td>{metricLabel(name)}</td>
                    {columns.map((c) => {
                      const v = metrics[name][c.id];
                      return (
                        <td key={c.id} className={`num${best === c.id ? " best" : ""}`}>
                          {v === null || v === undefined ? "not logged" : num(v, 3)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">The highest value in each row is in bold.</p>
    </section>
  );
}
