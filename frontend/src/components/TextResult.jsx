import { LabelChip, Tag } from "./Chip.jsx";
import { REASONS, num, pct } from "../format.js";

export default function TextResult({ data }) {
  const unknown = data.category === "unknown";
  let explanation = REASONS[data.reason] || data.reason;
  if (data.reason === "low_confidence" && data.best_guess) {
    explanation = `${explanation} Its best guess was ${data.best_guess} at ${pct(data.confidence)}.`;
  }
  return (
    <section className="result" aria-label="Text classification result">
      <header className="result__head">
        <h3>{data.filename}</h3>
        <p className="muted small">
          {String(data.file_type).toUpperCase()} document, {num(data.text_chars, 0)} characters of text read.
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
        </p>
      </div>

      {data.top_k?.length > 0 && (
        <div className="topk">
          <h4>Closest categories</h4>
          <ul>
            {data.top_k.map((c) => (
              <li key={c.category}>
                <span>{c.category}</span>
                <span className="meter meter--small" aria-hidden="true">
                  <span className="meter__fill" style={{ width: pct(c.confidence) }} />
                </span>
                <span className="num">{pct(c.confidence)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <dl className="facts">
        <div>
          <dt>Text extracted with</dt>
          <dd>{data.extractor}</dd>
        </div>
        <div>
          <dt>Model version</dt>
          <dd>
            {data.model_version} <Tag>{data.source}</Tag>
          </dd>
        </div>
      </dl>
    </section>
  );
}
