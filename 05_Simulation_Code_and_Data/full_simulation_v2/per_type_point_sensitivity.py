"""
per_type_point_sensitivity.py - Per-anomaly-type point-level (per-second)
sensitivity, decomposed by environment and anomaly_type.

Table 6 reports only the AGGREGATE point-level sensitivity across all three
anomaly types pooled together. This script answers a question the Week5
re-review panel raised (Devil's Advocate C1/C2, corroborated by direct code
inspection of sim_engine.py): does the aggregate 26-28% figure reflect a
roughly uniform miss rate across bradycardia/tachycardia/fever, or is it
dominated by one anomaly type whose rule coverage has a structural gap?

Streams the existing 40 raw batch parquet files (no re-simulation).
"""
import pandas as pd
import numpy as np
import glob
import collections

files = sorted(glob.glob("raw_data_v2/*.parquet"))

counts = collections.defaultdict(lambda: {"TP": 0, "FN": 0})

for f in files:
    env = "Controlled" if f.split("\\")[-1].split("/")[-1].startswith("Controlled") else "Residential"
    df = pd.read_parquet(
        f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type"]
    )
    anom = df[df["ground_truth_anomaly"]]
    for a_type, sub in anom.groupby("anomaly_type"):
        tp = int(sub["alert_triggered"].sum())
        fn = len(sub) - tp
        counts[(env, a_type)]["TP"] += tp
        counts[(env, a_type)]["FN"] += fn

print(f"{'Environment':<12} {'Type':<12} {'TP':>10} {'FN':>10} {'Total':>10} {'Sensitivity':>12}")
for env in ["Controlled", "Residential"]:
    env_tp = env_fn = 0
    for a_type in ["bradycardia", "tachycardia", "fever"]:
        c = counts[(env, a_type)]
        tp, fn = c["TP"], c["FN"]
        total = tp + fn
        sens = 100 * tp / total if total else float("nan")
        print(f"{env:<12} {a_type:<12} {tp:>10} {fn:>10} {total:>10} {sens:>11.2f}%")
        env_tp += tp
        env_fn += fn
    env_total = env_tp + env_fn
    env_sens = 100 * env_tp / env_total if env_total else float("nan")
    print(f"{env:<12} {'AGGREGATE':<12} {env_tp:>10} {env_fn:>10} {env_total:>10} {env_sens:>11.2f}%")
    print()
