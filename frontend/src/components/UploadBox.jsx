import { useRef, useState } from "react";

function sizeText(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function UploadBox({ accept, hint, file, onFile, disabled }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);

  const take = (list) => {
    if (list && list.length > 0) onFile(list[0]);
  };

  return (
    <div
      className={`drop${over ? " drop--over" : ""}${disabled ? " drop--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) take(e.dataTransfer.files);
      }}
    >
      <input
        ref={input}
        type="file"
        accept={accept}
        hidden
        data-testid="file-input"
        onChange={(e) => take(e.target.files)}
      />
      {file ? (
        <p className="drop__file">
          <strong>{file.name}</strong>
          <span className="muted"> {sizeText(file.size)}</span>
        </p>
      ) : (
        <p className="drop__prompt">Drop a file here</p>
      )}
      <p className="muted small">{hint}</p>
      <button className="btn btn--secondary" onClick={() => input.current?.click()} disabled={disabled}>
        {file ? "Choose a different file" : "Choose a file"}
      </button>
    </div>
  );
}
