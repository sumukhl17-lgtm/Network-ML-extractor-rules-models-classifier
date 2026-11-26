import { LabelChip, Tag } from "./Chip.jsx";
import { REASONS, num, pct } from "../format.js";

export default function AudioResult({ data }) {
  const unknown = data.category === "unknown";
  const explanation = REASONS[data.reason] || data.reason;

  return (
    <section className="result" aria-label="Audio classification result">
      <header className="result__head">
        <h3>{data.filename}</h3>
        <p className="muted small">
          Audio classification, {num(data.duration_seconds, 1)} seconds.
        </p>
      </header>

      <div className="verdict">
        <LabelChip label={data.category} />
        <div className="meter" role="img" aria-label={`Confidence ${pct(data.confidence)}`}>
          <span className="meter__fill" style={{ width: pct(data.confidence) }} />
        </div>
        <span className="num">{pct(data.confidence)}</span>
      </div>

      <div className={`callout${unknown ? " callout--unknown" : ""}`}>
        <p>
          <strong>{unknown ? "Unknown." : "Classified."}</strong> {explanation}
          {data.best_guess && data.best_guess !== data.category
            ? ` Best guess: ${data.best_guess}.`
            : ""}
        </p>
      </div>

      {data.top_k?.length > 0 && (
        <div className="topk">
          <h4>Closest sound categories</h4>
          <ul>
            {data.top_k.map((item) => (
              <li key={item.category}>
                <span>{item.category}</span>
                <span className="meter meter--small" aria-hidden="true">
                  <span className="meter__fill" style={{ width: pct(item.confidence) }} />
                </span>
                <span className="num">{pct(item.confidence)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <dl className="facts">
        <div>
          <dt>Sound group</dt>
          <dd>{data.sound_group}</dd>
        </div>
        <div>
          <dt>Audio format</dt>
          <dd>{data.sample_rate} Hz, {data.channels} channel(s)</dd>
        </div>
        <div>
          <dt>Model version</dt>
          <dd>{data.model_version} <Tag>{data.source}</Tag></dd>
        </div>
      </dl>
    </section>
  );
}
