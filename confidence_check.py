import pandas as pd
from classify_network import classify_network

df = pd.read_csv("flows5.csv")
out = []
for r in df.to_dict("records"):
    res = classify_network(r)
    if res["source"] == "ml" and res["label"] == "Known":
        out.append((r["label"], res["confidence"]))
res = pd.DataFrame(out, columns=["truth", "confidence"])
print(res.groupby("truth")["confidence"].describe().round(3))
