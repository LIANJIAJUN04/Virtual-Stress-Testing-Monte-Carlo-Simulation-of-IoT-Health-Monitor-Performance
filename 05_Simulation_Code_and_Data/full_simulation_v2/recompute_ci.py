import glob
import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

SRC_DIR = "raw_data_v2_corrected"
N_BOOT = 2000
RNG = np.random.default_rng(20260826)


def welch_ttest(a, b):
    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_value)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def main():
    files = sorted(glob.glob(f"{SRC_DIR}/*.parquet"))
    per_run_confusion = {}
    per_run_type_counts = {}  # (env, run_id, anomaly_type) -> {TP,FN}

    for f in files:
        env = "Controlled" if os.path.basename(f).startswith("Controlled") else "Residential"
        df = pd.read_parquet(f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type"])
        df = df.sort_values(["run_id", "time"])
        gt = df["ground_truth_anomaly"].to_numpy()
        pred = df["alert_triggered"].to_numpy()
        run_id = df["run_id"].to_numpy()
        a_type = df["anomaly_type"].to_numpy()

        for rid, idx in pd.Series(np.arange(len(df))).groupby(run_id):
            i = idx.to_numpy()
            g, p = gt[i], pred[i]
            per_run_confusion[(env, rid)] = {
                "TP": int((g & p).sum()), "FP": int((~g & p).sum()),
                "TN": int((~g & ~p).sum()), "FN": int((g & ~p).sum()),
            }
            at = a_type[i]
            for t in ["bradycardia", "tachycardia", "fever"]:
                mask = g & (at == t)
                tp = int((mask & p).sum())
                fn = int((mask & ~p).sum())
                per_run_type_counts[(env, rid, t)] = {"TP": tp, "FN": fn}

    all_runs = {env: sorted({rid for (e, rid) in per_run_confusion if e == env}) for env in ["Controlled", "Residential"]}

    print("=" * 80)
    print("TABLE 6 - cluster-bootstrap 95% CI around pooled estimate (corrected data)")
    print("=" * 80)
    for env in ["Controlled", "Residential"]:
        run_ids = np.array(all_runs[env])
        tp = fp = tn = fn = 0
        for rid in run_ids:
            c = per_run_confusion[(env, rid)]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        point_sens = tp / (tp + fn)
        point_spec = tn / (tn + fp)
        point_prec = tp / (tp + fp)

        boot_sens, boot_spec, boot_prec = [], [], []
        for _ in range(N_BOOT):
            sample = RNG.choice(run_ids, size=len(run_ids), replace=True)
            btp = bfp = btn = bfn = 0
            for rid in sample:
                c = per_run_confusion[(env, rid)]
                btp += c["TP"]; bfp += c["FP"]; btn += c["TN"]; bfn += c["FN"]
            boot_sens.append(btp / (btp + bfn) if (btp + bfn) else np.nan)
            boot_spec.append(btn / (btn + bfp) if (btn + bfp) else np.nan)
            boot_prec.append(btp / (btp + bfp) if (btp + bfp) else np.nan)
        for name, point, boot in [("sensitivity", point_sens, boot_sens),
                                    ("specificity", point_spec, boot_spec),
                                    ("precision", point_prec, boot_prec)]:
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            print(f"  {env:<12} {name:<12} point={100*point:6.2f}%  95% CI=[{100*lo:6.2f}%, {100*hi:6.2f}%]")
        print()

    print("=" * 80)
    print("Per-anomaly-type point-level sensitivity (corrected data)")
    print("=" * 80)
    for env in ["Controlled", "Residential"]:
        run_ids = np.array(all_runs[env])
        env_tp = env_fn = 0
        for t in ["bradycardia", "tachycardia", "fever"]:
            tp = fn = 0
            for rid in run_ids:
                c = per_run_type_counts[(env, rid, t)]
                tp += c["TP"]; fn += c["FN"]
            total = tp + fn
            sens = 100 * tp / total if total else float("nan")
            print(f"  {env:<12} {t:<12} TP={tp:>9} FN={fn:>9} Total={total:>9} Sensitivity={sens:6.2f}%")
            env_tp += tp; env_fn += fn
        env_total = env_tp + env_fn
        print(f"  {env:<12} {'AGGREGATE':<12} TP={env_tp:>9} FN={env_fn:>9} Total={env_total:>9} Sensitivity={100*env_tp/env_total:6.2f}%")
        print()


if __name__ == "__main__":
    main()
