# network-ml: Known/Unknown traffic classifier (Person 1)

Rules label recognised protocols by port. Unmatched flows default to Unknown.
A Random Forest / XGBoost model trained on nDPI-labelled flow statistics gives an advisory opinion.

## Setup

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q

## The contract

    from classify_network import classify_network
    classify_network(flow)

`flow` is a dict with: protocol, src_port, dst_port, packet_count, total_bytes,
avg_packet_size, duration, packet_rate, byte_rate, syn, fin, rst, psh, ack.

Returns `{"label": "HTTPS", "source": "rule"}` for a rule match, otherwise
`{"label": "Unknown", "source": "default", "ml_label": ..., "ml_confidence": ...}`.
Set `ML_OVERRIDES_UNMATCHED = True` in classify_network.py to let the model decide instead.

## Pipeline

1. `run_capture.sh` (macOS, uses sudo tcpdump on en0) produces capture*.pcap
2. `extract_flows.py in.pcap out.csv`: NFStream flows; nDPI's application_name gives the label (guessed flows dropped)
3. `build_dataset.py --train ... --test ...`: split by capture file
4. `train_compare.py`: trains Random Forest and XGBoost, logs to MLflow (./mlruns).
   The shipped model is `python train_compare.py --exclude-features protocol,syn,fin,rst,psh,ack`

Captures and CSVs are not in the repo (they contain traffic data). Regenerate them with run_capture.sh.
Saved models depend on scikit-learn 1.6.1; on a new machine, retrain with train_compare.py.

## Findings

- Held-out capture, same odd-traffic generator: F1 about 0.97 for both models.
- A different (slower) generator: Unknown recall 0.0 for models trained on the first generator, and vice versa (generalization_matrix.py).
- Known-only novelty detection (novelty_test.py): Unknown recall about 0.
- The model is confidently wrong on the missed flows (confidence_check.py), so thresholds don't help.
- Rules-only (unmatched = Unknown) matches or beats rules + ML on these captures (baseline_compare.py).
  Reason: flow statistics can't see payload, and nDPI's Unknown depends on payload.
- The Unknown class is synthetic (random UDP on random ports), so scores are not real-world performance.
