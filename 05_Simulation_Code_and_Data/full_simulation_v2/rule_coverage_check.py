"""
rule_coverage_check.py - Quantifies the "No Rule Matched" rate (Table 2 rule-plane
coverage gap) from the existing N=1000 raw simulation output. Streams the 40 raw
batch parquet files in raw_data_v2/ (no re-simulation) and tabulates alert_label
counts per environment.

Referenced in manuscript Section 5.5 (Limitations): "No Rule Matched" accounts for
27.63% of controlled-environment readings and 28.38% of residential-environment
readings.
"""
import pandas as pd
import glob
import collections

files = sorted(glob.glob("raw_data_v2/*.parquet"))

counts = collections.defaultdict(lambda: collections.Counter())
totals = collections.Counter()

for f in files:
    env = "Controlled" if f.split("\\")[-1].split("/")[-1].startswith("Controlled") else "Residential"
    df = pd.read_parquet(f, columns=["environment", "alert_label"])
    totals[env] += len(df)
    vc = df["alert_label"].value_counts()
    for label, c in vc.items():
        counts[env][label] += c

for env in ["Controlled", "Residential"]:
    print(f"--- {env} (total rows {totals[env]}) ---")
    for label, c in counts[env].most_common():
        print(f"  {label}: {c} ({100 * c / totals[env]:.4f}%)")
