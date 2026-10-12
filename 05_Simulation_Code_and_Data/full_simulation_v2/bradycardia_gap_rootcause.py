"""
bradycardia_gap_rootcause.py - For bradycardia episodes that received ZERO
alerts during their entire window, what rule state (if any) was the system
actually in during those seconds? Decomposes the undetected-episode seconds
by alert_label to determine whether the gap is a "No Rule Matched" coverage
hole, a competing rule firing instead, or something else.

Requested by Week5 (2026-08-26) re-review, Methodology W4.
"""
import glob
import numpy as np
import pandas as pd
import collections

files = sorted(glob.glob("raw_data_v2/*.parquet"))

label_counts = collections.Counter()
total_seconds = 0
n_undetected_episodes = 0
n_total_bradycardia_episodes = 0

for f in files:
    env = "Controlled" if f.split("\\")[-1].split("/")[-1].startswith("Controlled") else "Residential"
    df = pd.read_parquet(
        f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type", "alert_label"]
    )
    df = df.sort_values(["run_id", "time"])
    is_anom = df["ground_truth_anomaly"].to_numpy()
    run_id = df["run_id"].to_numpy()
    run_change = np.empty(len(run_id), dtype=bool)
    run_change[0] = True
    run_change[1:] = run_id[1:] != run_id[:-1]
    prev_anom = np.empty(len(is_anom), dtype=bool)
    prev_anom[0] = False
    prev_anom[1:] = is_anom[:-1]
    episode_start = is_anom & (~prev_anom | run_change)
    episode_id = np.cumsum(episode_start) * is_anom
    df["_episode_id"] = episode_id

    anom_rows = df[df["ground_truth_anomaly"] & (df["anomaly_type"] == "bradycardia")]
    grouped = anom_rows.groupby(["run_id", "_episode_id"])
    for (rid, eid), sub in grouped:
        n_total_bradycardia_episodes += 1
        if not sub["alert_triggered"].any():
            n_undetected_episodes += 1
            counts = sub["alert_label"].value_counts()
            for label, cnt in counts.items():
                label_counts[label] += cnt
            total_seconds += len(sub)

print(f"Total bradycardia episodes: {n_total_bradycardia_episodes}")
print(f"Fully-undetected bradycardia episodes (zero alerts in entire window): {n_undetected_episodes} "
      f"({100*n_undetected_episodes/n_total_bradycardia_episodes:.2f}%)")
print(f"\nAlert-label distribution across all {total_seconds:,} seconds of those undetected episodes:")
for label, cnt in label_counts.most_common():
    print(f"  {label:<30} {cnt:>10,}  ({100*cnt/total_seconds:.2f}%)")
