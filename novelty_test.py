import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["packet_count", "total_bytes", "avg_packet_size",
            "duration", "packet_rate", "byte_rate"]
files = {n: pd.read_csv("flows%d.csv" % n) for n in range(1, 6)}

train = pd.concat([files[n] for n in (1, 3, 4)])
train_known = train[train["label"] == "Known"]

print("%-14s %-16s %13s %15s" % ("contamination", "test set", "Known recall", "Unknown recall"))
for c in (0.01, 0.05, 0.10):
    model = IsolationForest(n_estimators=300, contamination=c, random_state=42)
    model.fit(train_known[FEATURES])
    for name, n in (("flows2 (fast)", 2), ("flows5 (slow)", 5)):
        test = files[n]
        pred = model.predict(test[FEATURES])
        is_known = (test["label"] == "Known").values
        known_rec = (pred[is_known] == 1).mean()
        unknown_rec = (pred[~is_known] == -1).mean()
        print("%-14s %-16s %13.3f %15.3f" % (c, name, known_rec, unknown_rec))
