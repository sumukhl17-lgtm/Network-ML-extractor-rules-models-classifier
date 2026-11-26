import io
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

# make `import backend.text` work no matter where pytest is started from
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.text import config  # noqa: E402
from backend.text.data import generate_sample_dataset as gen  # noqa: E402

TEST_MODEL_DIR = Path(__file__).resolve().parents[1] / "models-sklearn161-retrained"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts with the retrained model and no cached model."""
    for var in ("TEXT_EXTRACTOR", "TIKA_URL", "TIKA_TIMEOUT", "TEXT_UNKNOWN_THRESHOLD", "TEXT_MODEL_SOURCE",
                "TEXT_MODEL_DIR", "TEXT_STRICT_VERSIONS", "MLFLOW_TRACKING_URI", "TEXT_MIN_CHARS", "TEXT_MAX_BYTES"):
        monkeypatch.delenv(var, raising=False)
    missing = [name for name in ("text_svm.joblib", "model_meta.json")
               if not (TEST_MODEL_DIR / name).is_file()]
    if missing:
        pytest.fail(
            f"Retrained text model directory {TEST_MODEL_DIR} is missing required artifact(s): "
            f"{', '.join(missing)}"
        )
    monkeypatch.setenv("TEXT_MODEL_DIR", str(TEST_MODEL_DIR))
    import backend.text.model as m
    m._cached = None
    yield
    m._cached = None


@pytest.fixture
def make_docx():
    def _make(text: str) -> bytes:
        from docx import Document
        d = Document()
        for line in text.split("\n"):
            d.add_paragraph(line)
        buf = io.BytesIO()
        d.save(buf)
        return buf.getvalue()
    return _make


@pytest.fixture
def make_pdf():
    reportlab = pytest.importorskip("reportlab")  # noqa: F841 - dev-only dependency

    def _make(text: str) -> bytes:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        y = 800
        for line in text.split("\n"):
            c.drawString(40, y, line[:110])
            y -= 14
        c.save()
        return buf.getvalue()
    return _make


@pytest.fixture(scope="session")
def fresh_docs():
    """Documents from a seed the shipped model never saw: {category: [text, ...]}."""
    rng = random.Random(424242)
    return {c: [gen.make_document(c, rng) for _ in range(5)] for c in gen.CATEGORIES}


@pytest.fixture
def tika_stub():
    """A fake Tika server: PUT /tika returns canned text. Records what it received."""
    state = {"body": b"Invoice number 42. Total amount due: Rs 5000. Payment terms net 30 days.", "status": 200, "seen": []}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # Tika answers GET /tika with a greeting; used for health checks
            self.send_response(state.get("health_status", 200))
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"This is Tika Server (stub). Please PUT")

        def do_PUT(self):
            n = int(self.headers.get("Content-Length", 0))
            state["seen"].append({"path": self.path, "accept": self.headers.get("Accept"), "bytes": self.rfile.read(n),
                                  "disposition": self.headers.get("Content-Disposition")})
            self.send_response(state["status"])
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            if state["status"] == 200:
                self.wfile.write(state["body"])

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()
