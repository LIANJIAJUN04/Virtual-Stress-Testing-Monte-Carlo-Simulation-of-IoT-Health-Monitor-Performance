"""
analyze_results.py - Aggregate all batch outputs into the paper's KPI tables.

Reads every {Controlled,Residential}_batch_*.{parquet,csv} + *_mttd.csv file
from CONFIG['OUTPUT_DIR'], then prints/saves:
  - Table 5 style descriptive stats (alert rate, data completeness, avg HR/Temp)
  - Table 6 style clinical confusion matrix (sensitivity/specificity/precision/
    F1), computed against the simulation's own ground-truth anomaly labels -
    the same dataset now backs both the descriptive and clinical numbers,
    unlike the old two-separate-simulations setup.
  - Table 7 style Mean Time To Detect (MTTD) per anomaly type
  - Table 8 style alert distribution by rule
  - Controlled vs Residential significance tests (per-run alert rate)

Usage: python analyze_results.py
"""

import glob
import json

import numpy as np
import pandas as pd

from config import CONFIG

try:
    from scipy import stats as scipy_stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def welch_ttest(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if HAVE_SCIPY:
        t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t_stat), float(p_value)

    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 0 else 0.0
    # Normal approximation to the p-value (reasonable for n>=30 per group;
    # SciPy is not installed in this environment). Label as approximate.
    from math import erf, sqrt
    p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
    return float(t_stat), float(p_value)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def load_all():
    output_dir = CONFIG["OUTPUT_DIR"]
    files = sorted(glob.glob(f"{output_dir}/*_batch_*.parquet") + glob.glob(f"{output_dir}/*_batch_*.csv"))
    if not files:
        print(f"ERROR: no batch files found in {output_dir}/. Run run_batch.py first.")
        raise SystemExit(1)

    dfs = []
    for f in files:
        df = pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
        dfs.append(df)
        print(f"  loaded {f}: {len(df):,} rows")
    full_df = pd.concat(dfs, ignore_index=True)

    mttd_files = sorted(glob.glob(f"{output_dir}/*_mttd.csv"))
    mttd_dfs = [pd.read_csv(f) for f in mttd_files if pd.read_csv(f).shape[0] > 0] if mttd_files else []
    mttd_df = pd.concat(mttd_dfs, ignore_index=True) if mttd_dfs else pd.DataFrame(
        columns=["run_id", "environment", "anomaly_type", "detect_time"])

    return full_df, mttd_df


def main():
    print("=" * 70)
    print("ANALYZING FULL SIMULATION RESULTS")
    print("=" * 70)
    full_df, mttd_df = load_all()

    print(f"\nTotal rows: {len(full_df):,}")
    print(f"Total runs: {full_df.groupby('environment')['run_id'].nunique().to_dict()}")

    summary = {"environments": {}}

    # ---- Table 5: descriptive stats ----
    print("\n" + "=" * 70)
    print("TABLE 5 STYLE - DESCRIPTIVE STATS")
    print("=" * 70)
    for env in sorted(full_df["environment"].unique()):
        edf = full_df[full_df["environment"] == env]
        alert_rate = edf["alert_triggered"].mean()
        data_missing = edf["data_missing"].mean()
        hr_mean = edf["hr_measured"].mean()
        temp_mean = edf["temp_measured"].mean()
        print(f"\n{env}: alert_rate={alert_rate*100:.2f}%  data_missing={data_missing*100:.2f}%  "
              f"avg_hr={hr_mean:.2f}  avg_temp={temp_mean:.2f}")
        summary["environments"][env] = {
            "runs": int(edf["run_id"].nunique()),
            "alert_rate": float(alert_rate),
            "data_missing": float(data_missing),
            "avg_heart_rate": float(hr_mean),
            "avg_temperature": float(temp_mean),
        }

    # ---- Table 6: clinical confusion matrix vs ground-truth anomaly ----
    print("\n" + "=" * 70)
    print("TABLE 6 STYLE - CLINICAL PERFORMANCE (vs ground_truth_anomaly)")
    print("=" * 70)
    for env in sorted(full_df["environment"].unique()):
        edf = full_df[full_df["environment"] == env]
        gt = edf["ground_truth_anomaly"].astype(bool)
        pred = edf["alert_triggered"].astype(bool)

        tp = int((gt & pred).sum())
        fp = int((~gt & pred).sum())
        tn = int((~gt & ~pred).sum())
        fn = int((gt & ~pred).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        f1 = (2 * precision * sensitivity / (precision + sensitivity)
              if (precision + sensitivity) else float("nan"))

        print(f"\n{env}: TP={tp} FP={fp} TN={tn} FN={fn}")
        print(f"  Sensitivity={sensitivity:.4f}  Specificity={specificity:.4f}  "
              f"Precision={precision:.4f}  Accuracy={accuracy:.4f}  F1={f1:.4f}")

        summary["environments"][env].update({
            "confusion_matrix": {"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": precision, "accuracy": accuracy, "f1_score": f1,
        })

    # ---- Table 7: MTTD by anomaly type ----
    print("\n" + "=" * 70)
    print("TABLE 7 STYLE - MEAN TIME TO DETECT (MTTD)")
    print("=" * 70)
    mttd_summary = []
    if len(mttd_df):
        for (env, a_type), grp in mttd_df.groupby(["environment", "anomaly_type"]):
            row = {"environment": env, "anomaly_type": a_type,
                   "mean_mttd": float(grp["detect_time"].mean()),
                   "std_mttd": float(grp["detect_time"].std()),
                   "count": int(len(grp))}
            mttd_summary.append(row)
            print(f"  {env:12s} {a_type:12s} mean={row['mean_mttd']:.2f}s "
                  f"std={row['std_mttd']:.2f}s n={row['count']}")
    else:
        print("  No detected anomalies recorded.")
    summary["mttd"] = mttd_summary

    # ---- Table 8: alert distribution by rule ----
    print("\n" + "=" * 70)
    print("TABLE 8 STYLE - ALERT DISTRIBUTION BY RULE")
    print("=" * 70)
    rule_summary = []
    for env in sorted(full_df["environment"].unique()):
        edf = full_df[(full_df["environment"] == env) & (full_df["alert_triggered"])]
        counts = edf["alert_label"].value_counts(normalize=True) * 100
        print(f"\n{env}:")
        for label, pct in counts.items():
            print(f"  {label}: {pct:.1f}%")
            rule_summary.append({"environment": env, "rule": label, "pct": float(pct)})
    summary["alert_distribution"] = rule_summary

    # ---- Significance testing: per-run alert rate, Controlled vs Residential ----
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE (per-run alert rate, Controlled vs Residential)")
    print("=" * 70)
    if not HAVE_SCIPY:
        print("  NOTE: scipy not installed in this environment - p-value uses a")
        print("  normal approximation. Install scipy for the exact Welch's t-test.")

    per_run_alert = full_df.groupby(["environment", "run_id"])["alert_triggered"].mean()
    if "Controlled" in full_df["environment"].unique() and "Residential" in full_df["environment"].unique():
        c = per_run_alert.loc["Controlled"].values
        r = per_run_alert.loc["Residential"].values
        t_stat, p_value = welch_ttest(c, r)
        d = cohens_d(c, r)
        print(f"  Controlled: {c.mean()*100:.2f}% +/- {c.std()*100:.2f}%")
        print(f"  Residential: {r.mean()*100:.2f}% +/- {r.std()*100:.2f}%")
        print(f"  t={t_stat:.4f}  p={p_value:.6f}  Cohen's d={d:.4f}")
        summary["significance"] = {"t_stat": t_stat, "p_value": p_value, "cohens_d": d}

    with open("simulation_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nSaved: simulation_summary_v2.json")


if __name__ == "__main__":
    main()
