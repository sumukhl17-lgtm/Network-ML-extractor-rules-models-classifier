import { isUnknown } from "../format.js";

// Provenance is shown the same way everywhere:
//   solid chip  = the answer comes from a rule or from the text model (trust it)
//   dashed chip = the answer comes from, or is accompanied by, the network ML model (advisory only)
export function LabelChip({ label, advisory = false }) {
  const tone = isUnknown(label) ? "unknown" : "known";
  return (
    <span className={`chip chip--${tone}${advisory ? " chip--advisory" : ""}`}>
      {label}
      {advisory && <span className="chip__tag">advisory</span>}
    </span>
  );
}

export function Tag({ children, tone = "neutral" }) {
  return <span className={`tag tag--${tone}`}>{children}</span>;
}
