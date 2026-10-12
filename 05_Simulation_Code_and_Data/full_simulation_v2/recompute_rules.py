"""
recompute_rules.py - Post-processing re-evaluation of Table 2 alert
labels/triggers against the CORRECTED rule set (config.py, fixed 2026-08-26,
Week5 re-review R1/R2/R6), using the already-generated raw sensor readings.
Does NOT re-run the physiological / OU-process simulation.

Why forward-fill is required (not a naive per-row rule lookup):
The live simulation's Processor holds hr_buffer/temp_buffer across packet-
loss windows (a lost packet does not clear the buffer - see sim_engine.py
Processor.add_reading, which only updates on a non-None sample). The stored
hr_measured/temp_measured columns, by contrast, are sample()'s RAW per-step
return value, which IS None for the sensor's entire sampling-interval window
whenever that window's packet is lost (see run_single_simulation). So the
buffer value the ORIGINAL evaluate() call saw at any given second is
reconstructed by forward-filling hr_measured/temp_measured within each run
(sorted by time) - a value only stays "missing" for rule-evaluation purposes
before that sensor's first-ever successful sample in the run, exactly
matching the original Processor's buffer=None-at-start behaviour. This
reproduces the original live evaluation exactly; it is not an approximation.

Only alert_triggered / alert_label / rule_id are recomputed. hr_true,
temp_true, hr_measured, temp_measured, ground_truth_anomaly, anomaly_type,
data_missing, processing_latency are carried through unchanged (data_missing
is deliberately NOT touched - see note below).

Output: raw_data_v2_corrected/<same filename>.parquet (original raw_data_v2/
is left untouched as the pre-fix audit trail).
"""
import glob
import os
import numpy as np
import pandas as pd

from config import DECISION_RULES_BOTH_PRESENT, INSUFFICIENT_DATA_HR_MISSING, \
    INSUFFICIENT_DATA_TEMP_MISSING, INSUFFICIENT_DATA_BOTH_MISSING

SRC_DIR = "raw_data_v2"
DST_DIR = "raw_data_v2_corrected"
os.makedirs(DST_DIR, exist_ok=True)


def evaluate_vectorized(hr, temp):
    """hr, temp: numpy float arrays (NaN = missing/buffer-empty).
    Returns (alert_triggered, alert_label, rule_id) arrays, replicating
    Processor.evaluate()'s first-match-wins semantics exactly."""
    n = len(hr)
    alert = np.zeros(n, dtype=bool)
    label = np.full(n, "", dtype=object)
    rule_id = np.full(n, -99, dtype=int)

    both_present = ~np.isnan(hr) & ~np.isnan(temp)
    hr_only_missing = np.isnan(hr) & ~np.isnan(temp)
    temp_only_missing = ~np.isnan(hr) & np.isnan(temp)
    both_missing = np.isnan(hr) & np.isnan(temp)

    # both-sensor rules, first match wins -> iterate in REVERSE and overwrite,
    # so earlier (higher-priority) rules end up as the final assignment.
    unmatched = both_present.copy()
    for hr_lo, hr_hi, t_lo, t_hi, lbl, rid in reversed(DECISION_RULES_BOTH_PRESENT):
        hr_ok = np.ones(n, dtype=bool) if hr_lo is None and hr_hi is None else \
            ((hr >= hr_lo) if hr_lo is not None else True) & \
            ((hr <= hr_hi) if hr_hi is not None else True)
        t_ok = np.ones(n, dtype=bool) if t_lo is None and t_hi is None else \
            ((temp >= t_lo) if t_lo is not None else True) & \
            ((temp <= t_hi) if t_hi is not None else True)
        matched = both_present & hr_ok & t_ok
        label[matched] = lbl
        rule_id[matched] = rid
        alert[matched] = (rid != 0)
        unmatched &= ~matched
    # both-present but no rule matched
    label[unmatched] = "No Rule Matched"
    rule_id[unmatched] = -1
    alert[unmatched] = False

    label[hr_only_missing] = INSUFFICIENT_DATA_HR_MISSING
    rule_id[hr_only_missing] = -1
    label[temp_only_missing] = INSUFFICIENT_DATA_TEMP_MISSING
    rule_id[temp_only_missing] = -1
    label[both_missing] = INSUFFICIENT_DATA_BOTH_MISSING
    rule_id[both_missing] = -1

    return alert, label, rule_id


def process_file(path):
    df = pd.read_parquet(path)
    df = df.sort_values(["run_id", "time"]).reset_index(drop=True)

    # Reconstruct the live Processor's held buffer: forward-fill within each
    # run (a packet-loss window's raw None does not clear the buffer).
    hr_buf = df.groupby("run_id")["hr_measured"].ffill().to_numpy(dtype=float)
    temp_buf = df.groupby("run_id")["temp_measured"].ffill().to_numpy(dtype=float)

    alert, label, rule_id = evaluate_vectorized(hr_buf, temp_buf)

    df["alert_triggered"] = alert
    df["alert_label"] = label
    df["rule_id"] = rule_id
    # data_missing is deliberately left untouched: it reports the raw
    # per-step sensor sample (Table 5's data-completeness metric), not the
    # buffer state, and that definition did not change.

    out_path = os.path.join(DST_DIR, os.path.basename(path))
    df.to_parquet(out_path, index=False)
    return len(df), int(alert.sum())


def main():
    files = sorted(glob.glob(f"{SRC_DIR}/*.parquet"))
    total_rows = 0
    total_alerts = 0
    for i, f in enumerate(files, 1):
        n, a = process_file(f)
        total_rows += n
        total_alerts += a
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: {n:,} rows, {a:,} alerts ({100*a/n:.3f}%)")
    print(f"\nDone. Total rows: {total_rows:,}  Total alerts: {total_alerts:,} ({100*total_alerts/total_rows:.3f}%)")

    # Copy the *_mttd.csv files through unchanged reference - MTTD needs
    # recomputation too (detection timing depends on the corrected alert
    # stream), handled by a separate script (recompute_mttd.py).


if __name__ == "__main__":
    main()
