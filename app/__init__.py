"""Local web interface for classifying uploaded PCAP files."""

from flask import Flask, render_template_string, request

from classify_network import classify_network
from extract_flows import extract


app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Network flow classifier</title>
  <style>
    body { font: 16px system-ui, sans-serif; margin: 3rem auto; max-width: 960px; padding: 0 1rem; color: #172033; }
    h1 { margin-bottom: .4rem; }
    form { display: flex; flex-wrap: wrap; align-items: center; gap: 1rem; margin: 2rem 0; padding: 1.25rem; background: #f3f6fb; border-radius: 12px; }
    button { border: 0; border-radius: 8px; padding: .7rem 1.2rem; background: #155eef; color: white; font-weight: 650; cursor: pointer; }
    button:hover { background: #004eeb; }
    .error { color: #b42318; padding: .75rem; background: #fef3f2; border-radius: 8px; }
    .summary { color: #475467; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; }
    th, td { padding: .7rem; border-bottom: 1px solid #eaecf0; text-align: left; }
    th { background: #f9fafb; }
    @media (max-width: 640px) { body { margin-top: 1.5rem; } table { font-size: .85rem; } th, td { padding: .45rem .25rem; } }
  </style>
</head>
<body>
  <h1>Network flow classifier</h1>
  <p>Upload a PCAP capture to extract and classify its flows.</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="pcap" accept=".pcap" required>
    <button type="submit">Classify</button>
  </form>
  {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
  {% if results is not none %}
    <h2>Results</h2>
    <p class="summary">{{ filename }}: {{ results|length }} flow(s) classified.</p>
    {% if results %}
    <table>
      <thead><tr><th>#</th><th>Source port</th><th>Destination port</th><th>Label</th><th>Source</th><th>Confidence</th></tr></thead>
      <tbody>
      {% for result in results %}
        <tr>
          <td>{{ loop.index }}</td>
          <td>{{ result.src_port }}</td>
          <td>{{ result.dst_port }}</td>
          <td>{{ result.label }}</td>
          <td>{{ result.source }}</td>
          <td>{% if result.confidence is not none %}{{ "%.1f"|format(result.confidence * 100) }}%{% else %}—{% endif %}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(PAGE, error=None, results=None, filename=None)

    upload = request.files.get("pcap")
    if upload is None or not upload.filename:
        return (
            render_template_string(
                PAGE,
                error="Choose a .pcap file to classify.",
                results=None,
                filename=None,
            ),
            400,
        )
    if upload.filename.rsplit(".", 1)[-1].lower() != "pcap":
        return (
            render_template_string(
                PAGE,
                error="The uploaded file must have a .pcap extension.",
                results=None,
                filename=None,
            ),
            400,
        )

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pcap") as temporary_file:
        upload.save(temporary_file)
        temporary_file.flush()
        flows = extract(temporary_file.name)

    results = []
    for flow in flows.to_dict(orient="records"):
        prediction = classify_network(flow)
        results.append(
            {
                "src_port": flow["src_port"],
                "dst_port": flow["dst_port"],
                "label": prediction["label"],
                "source": prediction["source"],
                "confidence": prediction.get(
                    "confidence",
                    prediction.get("ml_confidence"),
                ),
            }
        )

    return render_template_string(
        PAGE,
        error=None,
        results=results,
        filename=upload.filename,
    )
