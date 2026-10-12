"""
recompute_mttd.py - Recompute per-episode Mean-Time-To-Detect against the
corrected alert stream (raw_data_v2_corrected/), for episodes that were ever
detected. Vectorised (no per-row Python loop over 86.4M rows).

Detection time = first time within [episode_start, episode_end) where
alert_triggered is True, minus episode_start. Episodes never detected are
excluded from MTTD (consistent with the original live-simulation logic,
which also only recorded a detection time when one occurred).
"""
import glob
import os
import numpy as np
import pandas as pd

SRC_DIR = "raw_data_v2_corrected"
OUT_CSV = "mttd_corrected.csv"


def process_file(path):
    df = pd.read_parquet(
        path, columns=["run_id", "environment", "time", "alert_triggered",
                        "ground_truth_anomaly", "anomaly_type"]
    )
    df = df.sort_values(["run_id", "time"]).reset_index(drop=True)
    env = df["environment"].iloc[0]

    is_anom = df["ground_truth_anomaly"].to_numpy()
    run_id = df["run_id"].to_numpy()
    time_arr = df["time"].to_numpy()
    alert = df["alert_triggered"].to_numpy()
    a_type = df["anomaly_type"].to_numpy()

    run_change = np.empty(len(run_id), dtype=bool)
    run_change[0] = True
    run_change[1:] = run_id[1:] != run_id[:-1]
    prev_anom = np.empty(len(is_anom), dtype=bool)
    prev_anom[0] = False
    prev_anom[1:] = is_anom[:-1]
    episode_start = is_anom & (~prev_anom | run_change)
    episode_id = np.cumsum(episode_start) * is_anom

    rows = []
    d = pd.DataFrame({
        "episode_id": episode_id, "time": time_arr, "alert": alert,
        "is_anom": is_anom, "anomaly_type": a_type, "run_id": run_id,
    })
    anom_rows = d[d["is_anom"]]
    for eid, sub in anom_rows.groupby("episode_id"):
        sub_alert = sub[sub["alert"]]
        if len(sub_alert) == 0:
            continue
        start_t = sub["time"].iloc[0]
        detect_t = sub_alert["time"].iloc[0]
        rows.append({
            "run_id": int(sub["run_id"].iloc[0]),
            "environment": env,
            "anomaly_type": sub["anomaly_type"].iloc[0],
            "detect_time": detect_t - start_t,
        })
    return rows


def main():
    files = sorted(glob.glob(f"{SRC_DIR}/*.parquet"))
    all_rows = []
    for i, f in enumerate(files, 1):
        rows = process_file(f)
        all_rows.extend(rows)
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {len(rows)} detected episodes")
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out)} detected-episode MTTD rows to {OUT_CSV}")
    for env in ["Controlled", "Residential"]:
        for a_type in ["bradycardia", "tachycardia", "fever"]:
            sub = out[(out["environment"] == env) & (out["anomaly_type"] == a_type)]
            if len(sub):
                print(f"  {env:<12} {a_type:<12} mean={sub['detect_time'].mean():8.2f}s "
                      f"std={sub['detect_time'].std():8.2f}s n={len(sub)}")


if __name__ == "__main__":
    main()
