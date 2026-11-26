from io import BytesIO

import pandas as pd

import app


def test_homepage_shows_upload_and_classify_controls():
    response = app.app.test_client().get("/")

    assert response.status_code == 200
    assert b'method="post" enctype="multipart/form-data"' in response.data
    assert b'name="pcap" accept=".pcap"' in response.data
    assert b">Classify</button>" in response.data


def test_upload_classifies_each_extracted_flow(monkeypatch):
    monkeypatch.setattr(
        app,
        "extract",
        lambda path: pd.DataFrame(
            [
                {"src_port": 12345, "dst_port": 54321},
                {"src_port": 54321, "dst_port": 12345},
            ]
        ),
    )
    classified_flows = []

    def classify(flow):
        classified_flows.append(flow)
        return {"label": "Unknown", "source": "default", "ml_confidence": 0.8}

    monkeypatch.setattr(app, "classify_network", classify)
    response = app.app.test_client().post(
        "/",
        data={"pcap": (BytesIO(b"pcap fixture"), "sample.pcap")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert len(classified_flows) == 2
    assert b"sample.pcap: 2 flow(s) classified." in response.data
    assert b"Unknown" in response.data
    assert b"80.0%" in response.data


def test_upload_rejects_non_pcap_file():
    response = app.app.test_client().post(
        "/",
        data={"pcap": (BytesIO(b"not pcap"), "sample.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"must have a .pcap extension" in response.data
