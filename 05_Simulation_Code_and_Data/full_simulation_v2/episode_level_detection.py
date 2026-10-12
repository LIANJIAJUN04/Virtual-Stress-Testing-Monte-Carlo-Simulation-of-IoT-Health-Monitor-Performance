"""
episode_level_detection.py - Formal, reproducible computation of event-level
("episode-level") anomaly detection rate, alongside the point-level (per-second)
sensitivity already reported in Table 6.

Point-level sensitivity (Table 6) scores every ground-truth-anomaly SECOND
independently, including the ramp-up portion of an anomaly's injection window
before the physiological value has drifted far enough to cross a Table 2
threshold (bradycardia ramps over the first half of its 5-minute window; fever
over the first two-thirds of its 30-minute window). Event-level detection rate
instead asks, per injected anomaly EPISODE: did the system alert at any point
before the episode's window closed, regardless of when.

Streams the existing 40 raw batch parquet files (no re-simulation), vectorised
per run (contiguous-episode grouping via a cumulative-sum edge trick rather
than a Python row loop, since each arm is ~43.2M rows).
"""
import pandas as pd
import numpy as np
import glob
import collections

files = sorted(glob.glob("raw_data_v2/*.parquet"))

episode_totals = collections.Counter()
episode_detected = collections.Counter()

for f in files:
    env = "Controlled" if f.split("\\")[-1].split("/")[-1].startswith("Controlled") else "Residential"
    df = pd.read_parquet(
        f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type"]
    )
    df = df.sort_values(["run_id", "time"])
    is_anom = df["ground_truth_anomaly"].to_numpy()
    run_id = df["run_id"].to_numpy()
    # new episode starts where is_anom flips 0->1, OR run_id changes while anom
    run_change = np.empty(len(run_id), dtype=bool)
    run_change[0] = True
    run_change[1:] = run_id[1:] != run_id[:-1]
    prev_anom = np.empty(len(is_anom), dtype=bool)
    prev_anom[0] = False
    prev_anom[1:] = is_anom[:-1]
    episode_start = is_anom & (~prev_anom | run_change)
    episode_id = np.cumsum(episode_start) * is_anom  # 0 for non-anomaly rows
    df["_episode_id"] = episode_id

    anom_rows = df[df["ground_truth_anomaly"]]
    grouped = anom_rows.groupby("_episode_id").agg(
        anomaly_type=("anomaly_type", "first"),
        alerted=("alert_triggered", "any"),
    )
    for anomaly_type, sub in grouped.groupby("anomaly_type"):
        episode_totals[(env, anomaly_type)] += len(sub)
        episode_detected[(env, anomaly_type)] += int(sub["alerted"].sum())

for env in ["Controlled", "Residential"]:
    env_total = sum(v for (e, t), v in episode_totals.items() if e == env)
    env_detected = sum(v for (e, t), v in episode_detected.items() if e == env)
    print(f"--- {env}: {env_detected}/{env_total} = {100*env_detected/env_total:.2f}% ---")
    for t in ["bradycardia", "tachycardia", "fever"]:
        tot = episode_totals[(env, t)]
        det = episode_detected[(env, t)]
        rate = 100 * det / tot if tot else float("nan")
        print(f"  {t}: {det}/{tot} = {rate:.2f}%")
