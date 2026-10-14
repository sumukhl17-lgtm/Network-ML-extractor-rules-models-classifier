import pandas as pd
from rules import classify_by_rule
from classify_network import classify_network

print("%-14s %-16s %13s %15s" % ("system", "test set", "Known recall", "Unknown recall"))
for n, name in ((2, "flows2 (fast)"), (5, "flows5 (slow)")):
    df = pd.read_csv("flows%d.csv" % n)
    rows = df.to_dict("records")
    truth_known = (df["label"] == "Known").values
    rules_only = pd.Series(["Known" if classify_by_rule(r) else "Unknown" for r in rows]).values
    hybrid = pd.Series(["Unknown" if classify_network(r)["label"] == "Unknown" else "Known" for r in rows]).values
    for sysname, pred in (("rules only", rules_only), ("rules + RF", hybrid)):
        k = (pred[truth_known] == "Known").mean()
        u = (pred[~truth_known] == "Unknown").mean()
        print("%-14s %-16s %13.3f %15.3f" % (sysname, name, k, u))
