import numpy as np

import classify_network


def make_flow(**overrides):
    flow = {
        "protocol": 6,
        "src_port": 12345,
        "dst_port": 54321,
        "packet_count": 12,
        "total_bytes": 1200,
        "avg_packet_size": 100.0,
        "duration": 40.0,
        "packet_rate": 0.3,
        "byte_rate": 30.0,
        "syn": 0,
        "fin": 0,
        "rst": 0,
        "psh": 6,
        "ack": 12,
    }
    flow.update(overrides)
    return flow


def test_rule_classification_skips_model(monkeypatch):
    def fail_if_called(row):
        assert row is not None
        raise AssertionError("model should not run when a rule matches")

    monkeypatch.setattr(classify_network._MODEL, "predict", fail_if_called)

    result = classify_network.classify_network(make_flow(dst_port=443))

    assert result == {"label": "HTTPS", "source": "rule"}


def install_fake_model(monkeypatch):
    observed = {}

    class FakeModel:
        classes_ = np.array([0, 1])

        def predict(self, row):
            observed["feature_names"] = row.columns.tolist()
            observed["values"] = row.iloc[0].tolist()
            return np.array([1])

        def predict_proba(self, row):
            assert row.columns.tolist() == classify_network._MODEL_FEATURES
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(classify_network, "_MODEL", FakeModel())
    return observed


def test_unmatched_flow_defaults_to_unknown_and_reports_ml_prediction(monkeypatch):
    observed = install_fake_model(monkeypatch)
    monkeypatch.setattr(classify_network, "ML_OVERRIDES_UNMATCHED", False)
    flow = make_flow()

    result = classify_network.classify_network(flow)

    assert result == {
        "label": "Unknown",
        "source": "default",
        "ml_label": "Unknown",
        "ml_confidence": 0.8,
    }
    assert observed["feature_names"] == classify_network._MODEL_FEATURES
    assert observed["values"] == [
        flow[feature] for feature in classify_network._MODEL_FEATURES
    ]


def test_ml_can_override_unmatched_flow_when_enabled(monkeypatch):
    install_fake_model(monkeypatch)
    monkeypatch.setattr(classify_network, "ML_OVERRIDES_UNMATCHED", True)

    result = classify_network.classify_network(make_flow())

    assert result == {
        "label": "Unknown",
        "source": "ml",
        "confidence": 0.8,
    }
