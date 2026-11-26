// Shows an ApiError the way the backend describes it: what went wrong, and how to fix it.
export default function ErrorNotice({ error, onDismiss }) {
  if (!error) return null;
  const d = error.details || {};
  return (
    <div className="notice notice--error" role="alert">
      <div className="notice__body">
        <strong>{error.message}</strong>
        {d.hint && <p>{d.hint}</p>}
        {Array.isArray(d.missing) && d.missing.length > 0 && (
          <p>
            Missing columns: <code>{d.missing.join(", ")}</code>
          </p>
        )}
        {d.row && d.column && (
          <p>
            Row {d.row}, column <code>{d.column}</code>.
          </p>
        )}
        {error.code && <p className="muted small">Error code: {error.code}</p>}
      </div>
      {onDismiss && (
        <button className="btn btn--quiet" onClick={onDismiss}>
          Dismiss
        </button>
      )}
    </div>
  );
}
