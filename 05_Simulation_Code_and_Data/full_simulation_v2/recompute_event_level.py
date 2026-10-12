import glob
import os
import collections
import numpy as np
import pandas as pd

SRC_DIR = "raw_data_v2_corrected"

files = sorted(glob.glob(f"{SRC_DIR}/*.parquet"))
episode_totals = collections.Counter()
episode_detected = collections.Counter()

for f in files:
    env = "Controlled" if os.path.basename(f).startswith("Controlled") else "Residential"
    df = pd.read_parquet(f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type"])
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
    anom_rows = df[df["ground_truth_anomaly"]]
    grouped = anom_rows.groupby("_episode_id").agg(
        anomaly_type=("anomaly_type", "first"), alerted=("alert_triggered", "any")
    )
    for a_type, sub in grouped.groupby("anomaly_type"):
        episode_totals[(env, a_type)] += len(sub)
        episode_detected[(env, a_type)] += int(sub["alerted"].sum())

for env in ["Controlled", "Residential"]:
    tot = sum(v for (e, t), v in episode_totals.items() if e == env)
    det = sum(v for (e, t), v in episode_detected.items() if e == env)
    print(f"--- {env}: {det}/{tot} = {100*det/tot:.2f}% ---")
    for t in ["bradycardia", "tachycardia", "fever"]:
        to = episode_totals[(env, t)]
        de = episode_detected[(env, t)]
        print(f"  {t}: {de}/{to} = {100*de/to:.2f}%")
