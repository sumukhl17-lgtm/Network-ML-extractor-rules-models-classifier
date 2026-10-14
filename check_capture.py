import sys
from nfstream import NFStreamer

for path in sys.argv[1:]:
    df = NFStreamer(source=path, statistical_analysis=True).to_pandas()
    usable = df[df["application_is_guessed"] == 0]
    unknown = usable["application_name"].str.startswith("Unknown").sum()
    odd = df["dst_ip"].str.startswith("203.0.113.").sum()
    print(f"{path}: total={len(df)} usable={len(usable)} "
          f"Unknown={unknown} Known={len(usable) - unknown} odd_flows_sent={odd}")
