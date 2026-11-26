import { useState } from "react";
import { api } from "../api.js";
import { addHistory } from "../storage.js";
import UploadBox from "../components/UploadBox.jsx";
import ErrorNotice from "../components/ErrorNotice.jsx";
import NetworkResults from "../components/NetworkResults.jsx";
import TextResult from "../components/TextResult.jsx";
import VideoResult from "../components/VideoResult.jsx";
import AudioResult from "../components/AudioResult.jsx";

const MODES = {
  network: {
    label: "Network traffic",
    accept: ".pcap,.pcapng,.csv",
    hint: "A packet capture (.pcap, .pcapng) or a CSV of flows. IPFIX and NetFlow exports are not supported yet.",
    run: api.classifyNetwork,
  },
  text: {
    label: "Documents",
    accept: ".pdf,.docx,.txt",
    hint: "A PDF, Word (.docx) or plain text file.",
    run: api.classifyText,
  },
  video: {
    label: "Video",
    accept: ".mp4,.avi,.mov,.mkv,.webm",
    hint: "A video file (.mp4, .avi, .mov, .mkv or .webm), up to 200 MiB and 10 minutes.",
    run: api.classifyVideo,
  },
  audio: {
    label: "Audio",
    accept: ".wav,.mp3,.flac,.ogg,.m4a,.aac",
    hint: "An audio file (.wav, .mp3, .flac, .ogg, .m4a or .aac), up to 20 MiB and 120 seconds.",
    run: api.classifyAudio,
  },
};

export default function ClassifyPage({ onSaved }) {
  const [mode, setMode] = useState("network");
  const [state, setState] = useState({
    network: {},
    text: {},
    video: {},
    audio: {},
  });
  const [busy, setBusy] = useState(false);

  const cur = state[mode];
  const patch = (m, values) =>
    setState((s) => ({ ...s, [m]: { ...s[m], ...values } }));

  async function classify() {
    const m = mode;
    patch(m, { error: null });
    setBusy(true);

    try {
      const result = await MODES[m].run(cur.file);
      patch(m, { result });
      addHistory(m, result);
      onSaved?.();
    } catch (error) {
      patch(m, { error, result: null });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <h2>Classify</h2>

      <div className="tabs" role="tablist" aria-label="What to classify">
        {Object.entries(MODES).map(([id, item]) => (
          <button
            key={id}
            role="tab"
            aria-selected={mode === id}
            className={`tab${mode === id ? " is-on" : ""}`}
            onClick={() => setMode(id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <UploadBox
        accept={MODES[mode].accept}
        hint={MODES[mode].hint}
        file={cur.file}
        disabled={busy}
        onFile={(file) => patch(mode, { file, error: null, result: null })}
      />

      <div className="actions">
        <button
          className="btn btn--primary"
          disabled={!cur.file || busy}
          onClick={classify}
        >
          {busy ? "Classifying" : "Classify file"}
        </button>
      </div>

      <ErrorNotice
        error={cur.error}
        onDismiss={() => patch(mode, { error: null })}
      />

      {cur.result && mode === "network" && (
        <NetworkResults data={cur.result} />
      )}

      {cur.result && mode === "text" && (
        <TextResult data={cur.result} />
      )}

      {cur.result && mode === "video" && (
        <VideoResult data={cur.result} />
      )}

      {cur.result && mode === "audio" && (
        <AudioResult data={cur.result} />
      )}
    </div>
  );
}
