import { useEffect, useRef, useState } from "react";

// A confirmation step for actions that change which model is live.
// The confirm button stays disabled until the person ticks the box.
export default function ConfirmDialog({ open, title, children, confirmLabel, checkLabel, busy, onConfirm, onCancel }) {
  const ref = useRef(null);
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      setChecked(false);
      dialog.showModal();
    }
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog ref={ref} className="dialog" onCancel={onCancel} onClose={() => open && onCancel()}>
      <h3>{title}</h3>
      <div className="dialog__body">{children}</div>
      <label className="check">
        <input type="checkbox" checked={checked} onChange={(e) => setChecked(e.target.checked)} />
        <span>{checkLabel}</span>
      </label>
      <div className="dialog__actions">
        <button className="btn btn--secondary" onClick={onCancel} disabled={busy}>
          Cancel
        </button>
        <button className="btn btn--primary" onClick={onConfirm} disabled={!checked || busy}>
          {busy ? "Working" : confirmLabel}
        </button>
      </div>
    </dialog>
  );
}
