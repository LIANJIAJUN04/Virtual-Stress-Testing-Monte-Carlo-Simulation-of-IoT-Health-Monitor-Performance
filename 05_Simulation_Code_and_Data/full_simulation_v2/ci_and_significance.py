"""
ci_and_significance.py - Cluster-bootstrap 95% confidence intervals for the
ALREADY-REPORTED Table 6 / Table 9 pooled point estimates, and significance
testing for Table 7 MTTD environmental comparisons.

Requested by the Week5 (2026-08-26) re-review panel (Journal-Fit W3,
Methodology W2/W3, Devil's Advocate M2): the paper's Methods section frames
CI estimation as a benefit of the Monte Carlo design, but Results never
delivered it, and MTTD environmental differences were asserted directionally
without a significance test despite SDs comparable to the differences
themselves.

Cluster unit = RUN (500 per environment), not individual seconds, since
seconds within one anomaly episode are highly autocorrelated (Methodology W2).
Uses a cluster bootstrap (resample runs with replacement, recompute the
POOLED confusion-matrix-derived metric on each resample) rather than the
mean-of-per-run-ratios approach, specifically so the CI is a direct
uncertainty band around the same pooled point estimate already reported in
Table 6/Table 9 (avoids introducing a second, differently-valued "per-run
mean" statistic that would need its own justification - mean-of-ratios and
ratio-of-sums diverge substantially here whenever per-run alert volume is
uneven, which it is).

Streams the existing 40 raw batch parquet files + MTTD csvs (no re-simulation).
"""
import glob
import numpy as np
import pandas as pd

from analyze_results import welch_ttest, cohens_d

N_BOOT = 2000
RNG = np.random.default_rng(20260826)


def build_per_run_tables():
    files = sorted(glob.glob("raw_data_v2/*.parquet"))
    per_run_confusion = {}  # (env, run_id) -> dict(TP,FP,TN,FN)
    per_run_episodes = {}   # (env, run_id, anomaly_type) -> dict(n_episodes, n_detected)

    for f in files:
        env = "Controlled" if f.split("\\")[-1].split("/")[-1].startswith("Controlled") else "Residential"
        df = pd.read_parquet(
            f, columns=["run_id", "time", "alert_triggered", "ground_truth_anomaly", "anomaly_type"]
        )
        df = df.sort_values(["run_id", "time"])
        gt = df["ground_truth_anomaly"].to_numpy()
        pred = df["alert_triggered"].to_numpy()
        run_id = df["run_id"].to_numpy()

        for rid, idx in pd.Series(np.arange(len(df))).groupby(run_id):
            i = idx.to_numpy()
            g, p = gt[i], pred[i]
            per_run_confusion[(env, rid)] = {
                "TP": int((g & p).sum()), "FP": int((~g & p).sum()),
                "TN": int((~g & ~p).sum()), "FN": int((g & ~p).sum()),
            }

        is_anom = gt
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
        grouped = anom_rows.groupby(["run_id", "_episode_id"]).agg(
            anomaly_type=("anomaly_type", "first"), alerted=("alert_triggered", "any")
        )
        for (rid, a_type), sub in grouped.reset_index().groupby(["run_id", "anomaly_type"]):
            per_run_episodes[(env, rid, a_type)] = {
                "n_episodes": len(sub), "n_detected": int(sub["alerted"].sum())
            }

    return per_run_confusion, per_run_episodes


def cluster_bootstrap_confusion(per_run_confusion, env, run_ids):
    n = len(run_ids)
    boot_sens, boot_spec, boot_prec = [], [], []
    for _ in range(N_BOOT):
        sample = RNG.choice(run_ids, size=n, replace=True)
        tp = fp = tn = fn = 0
        for rid in sample:
            c = per_run_confusion[(env, rid)]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        boot_sens.append(tp / (tp + fn) if (tp + fn) else np.nan)
        boot_spec.append(tn / (tn + fp) if (tn + fp) else np.nan)
        boot_prec.append(tp / (tp + fp) if (tp + fp) else np.nan)
    return np.array(boot_sens), np.array(boot_spec), np.array(boot_prec)


def cluster_bootstrap_episode_rate(per_run_episodes, env, a_type, run_ids):
    n = len(run_ids)
    boot_rate = []
    for _ in range(N_BOOT):
        sample = RNG.choice(run_ids, size=n, replace=True)
        tot = det = 0
        for rid in sample:
            e = per_run_episodes.get((env, rid, a_type))
            if e is None:
                continue
            tot += e["n_episodes"]; det += e["n_detected"]
        boot_rate.append(det / tot if tot else np.nan)
    return np.array(boot_rate)


def pct_ci(boot, point):
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def main():
    per_run_confusion, per_run_episodes = build_per_run_tables()
    all_runs = {env: sorted({rid for (e, rid) in per_run_confusion if e == env}) for env in ["Controlled", "Residential"]}

    print("=" * 82)
    print("TABLE 6 headline metrics - cluster-bootstrap 95% CI around the POOLED estimate")
    print("(resampling 500 runs with replacement, 2000 replicates)")
    print("=" * 82)
    for env in ["Controlled", "Residential"]:
        run_ids = np.array(all_runs[env])
        tp = fp = tn = fn = 0
        for rid in run_ids:
            c = per_run_confusion[(env, rid)]
            tp += c["TP"]; fp += c["FP"]; tn += c["TN"]; fn += c["FN"]
        point_sens = tp / (tp + fn)
        point_spec = tn / (tn + fp)
        point_prec = tp / (tp + fp)
        boot_sens, boot_spec, boot_prec = cluster_bootstrap_confusion(per_run_confusion, env, run_ids)
        for name, point, boot in [("sensitivity", point_sens, boot_sens),
                                    ("specificity", point_spec, boot_spec),
                                    ("precision", point_prec, boot_prec)]:
            p, lo, hi = pct_ci(boot, point)
            print(f"  {env:<12} {name:<12} point={100*p:6.2f}%  95% CI=[{100*lo:6.2f}%, {100*hi:6.2f}%]")
        print()

    print("=" * 82)
    print("TABLE 9 event-level detection rate - cluster-bootstrap 95% CI around pooled estimate")
    print("=" * 82)
    for env in ["Controlled", "Residential"]:
        run_ids = np.array(all_runs[env])
        for a_type in ["bradycardia", "tachycardia", "fever"]:
            tot = det = 0
            for rid in run_ids:
                e = per_run_episodes.get((env, rid, a_type))
                if e is None:
                    continue
                tot += e["n_episodes"]; det += e["n_detected"]
            point = det / tot if tot else float("nan")
            boot = cluster_bootstrap_episode_rate(per_run_episodes, env, a_type, run_ids)
            p, lo, hi = pct_ci(boot, point)
            print(f"  {env:<12} {a_type:<12} point={100*p:6.2f}%  95% CI=[{100*lo:6.2f}%, {100*hi:6.2f}%]")
        print()

    print("=" * 82)
    print("TABLE 7 MTTD - Welch's t-test, Controlled vs Residential, per anomaly type")
    print("=" * 82)
    mttd_files = sorted(glob.glob("raw_data_v2/*_mttd.csv"))
    mttd_dfs = [pd.read_csv(fp) for fp in mttd_files]
    mttd_df = pd.concat([d for d in mttd_dfs if len(d)], ignore_index=True)
    for a_type in ["bradycardia", "tachycardia", "fever"]:
        c = mttd_df[(mttd_df["environment"] == "Controlled") & (mttd_df["anomaly_type"] == a_type)]["detect_time"].values
        r = mttd_df[(mttd_df["environment"] == "Residential") & (mttd_df["anomaly_type"] == a_type)]["detect_time"].values
        t_stat, p_value = welch_ttest(c, r)
        d = cohens_d(c, r)
        print(f"  {a_type:<12} Controlled mean={c.mean():7.2f}s (n={len(c)})  Residential mean={r.mean():7.2f}s (n={len(r)})  "
              f"t={t_stat:6.3f}  p={p_value:.4f}  Cohen's d={d:6.3f}")


if __name__ == "__main__":
    main()
