"""
build_figures.py - Generate publication figures (Figure 1-4) for the paper,
directly from the full_simulation_v2 raw per-run output. Streams each batch
file, computes per-run summary stats (not loading all 86.4M rows into memory
at once), then plots.

Run: python build_figures.py
Output: paper_figures_v2/{png,pdf,svg}/figure{1,2,3,4}_*.{png,pdf,svg}
        paper_figures_v2/per_run_summary.csv (reusable aggregate, kept small)
"""
import glob
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

matplotlib.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Colorblind-safe categorical pair (Wong 2011 palette), fixed order.
COLOR_CONTROLLED = "#0072B2"
COLOR_RESIDENTIAL = "#E69F00"
ENV_COLORS = {"Controlled": COLOR_CONTROLLED, "Residential": COLOR_RESIDENTIAL}
ENV_ORDER = ["Controlled", "Residential"]

HERE = Path(__file__).parent
DATA_DIR = HERE / "raw_data_v2"
OUT_DIR = HERE / "paper_figures_v2"
for sub in ("png", "pdf", "svg"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

SAMPLE_PER_ENV = 20000  # for distribution plot, random sample not full 43.2M rows/env


def save_all(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / ext / f"{name}.{ext}")
    plt.close(fig)


def stream_aggregate():
    """One pass over every batch file: per-run summary rows + a random sample
    of measured HR/Temp per environment for the distribution figure."""
    per_run_rows = []
    samples = {"Controlled": [], "Residential": []}
    rng = np.random.default_rng(42)

    files = sorted(glob.glob(str(DATA_DIR / "*_batch_*.parquet")))
    print(f"Found {len(files)} batch files")
    for i, f in enumerate(files):
        df = pd.read_parquet(f, columns=[
            "run_id", "environment", "hr_measured", "temp_measured",
            "alert_triggered", "data_missing", "processing_latency",
        ])
        env = df["environment"].iloc[0]

        g = df.groupby("run_id").agg(
            alert_rate=("alert_triggered", "mean"),
            data_completeness=("data_missing", lambda s: 1 - s.mean()),
            avg_hr=("hr_measured", "mean"),
            avg_temp=("temp_measured", "mean"),
            avg_latency=("processing_latency", "mean"),
        ).reset_index()
        g["environment"] = env
        per_run_rows.append(g)

        # reservoir-ish sample: just take a random slice of valid measured values this batch
        valid = df.dropna(subset=["hr_measured", "temp_measured"])
        take = min(SAMPLE_PER_ENV // 20, len(valid))  # ~20 batches per env
        if take > 0:
            idx = rng.choice(len(valid), size=take, replace=False)
            samples[env].append(valid.iloc[idx][["hr_measured", "temp_measured"]])

        print(f"  [{i+1}/{len(files)}] {f.split(chr(92))[-1]}: {len(df):,} rows, env={env}")

    per_run = pd.concat(per_run_rows, ignore_index=True)
    sample_df = pd.concat(
        [pd.concat(v, ignore_index=True).assign(environment=k) for k, v in samples.items()],
        ignore_index=True,
    )
    per_run.to_csv(OUT_DIR / "per_run_summary.csv", index=False)
    sample_df.to_csv(OUT_DIR / "measured_sample.csv", index=False)
    print(f"\nPer-run summary: {len(per_run)} rows (runs). Sample: {len(sample_df)} rows.")
    return per_run, sample_df


def figure1_alert_rate(per_run):
    """Bar chart with error bars: mean alert rate per environment, +/- 1 SD across runs."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    means, sds = [], []
    for env in ENV_ORDER:
        vals = per_run.loc[per_run.environment == env, "alert_rate"] * 100
        means.append(vals.mean())
        sds.append(vals.std())
    x = np.arange(len(ENV_ORDER))
    bars = ax.bar(x, means, yerr=sds, capsize=4, width=0.55,
                   color=[ENV_COLORS[e] for e in ENV_ORDER],
                   edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_ORDER)
    ax.set_ylabel("Alert Rate (%)")
    ax.set_title("Figure 1. Mean Alert Rate by Environment (±1 SD across runs)")
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.2f}%", (xi, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    fig.tight_layout()
    save_all(fig, "figure1_alert_comparison")


def figure2_data_completeness(per_run):
    """Bar chart: mean data completeness (%) per environment, +/- 1 SD."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    means, sds = [], []
    for env in ENV_ORDER:
        vals = per_run.loc[per_run.environment == env, "data_completeness"] * 100
        means.append(vals.mean())
        sds.append(vals.std())
    x = np.arange(len(ENV_ORDER))
    ax.bar(x, means, yerr=sds, capsize=4, width=0.55,
           color=[ENV_COLORS[e] for e in ENV_ORDER], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(ENV_ORDER)
    ax.set_ylabel("Data Completeness (%)")
    ax.set_ylim(90, 100)
    ax.set_title("Figure 2. Sensor Data Completeness by Environment")
    for xi, m in zip(x, means):
        ax.annotate(f"{m:.2f}%", (xi, m), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)
    fig.tight_layout()
    save_all(fig, "figure2_data_completeness")


def figure3_distributions(sample_df):
    """Overlapping density plots of measured HR and Temperature by environment."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for env in ENV_ORDER:
        hr = sample_df.loc[sample_df.environment == env, "hr_measured"].dropna()
        axes[0].hist(hr, bins=60, density=True, alpha=0.5, label=env, color=ENV_COLORS[env])
    axes[0].set_xlabel("Measured Heart Rate (BPM)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("(a) Heart Rate Distribution")
    axes[0].legend()

    for env in ENV_ORDER:
        temp = sample_df.loc[sample_df.environment == env, "temp_measured"].dropna()
        axes[1].hist(temp, bins=60, density=True, alpha=0.5, label=env, color=ENV_COLORS[env])
    axes[1].set_xlabel("Measured Temperature (°C)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) Temperature Distribution")
    axes[1].legend()

    fig.suptitle("Figure 3. Overlapping Physiological Signal Distributions by Environment")
    fig.tight_layout()
    save_all(fig, "figure3_distributions")


def figure4_significance(per_run):
    """Box/strip plot of per-run alert rate by environment with the Welch's t-test annotated."""
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    data = [per_run.loc[per_run.environment == env, "alert_rate"].values * 100 for env in ENV_ORDER]
    bp = ax.boxplot(data, tick_labels=ENV_ORDER, patch_artist=True, widths=0.5,
                     showfliers=False)
    for patch, env in zip(bp["boxes"], ENV_ORDER):
        patch.set_facecolor(ENV_COLORS[env])
        patch.set_alpha(0.6)
    rng = np.random.default_rng(0)
    for i, env in enumerate(ENV_ORDER):
        vals = per_run.loc[per_run.environment == env, "alert_rate"].values * 100
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, s=6, color=ENV_COLORS[env],
                   alpha=0.35, linewidths=0)

    t_stat, p_value = scipy_stats.ttest_ind(data[0], data[1], equal_var=False)
    pooled_std = np.sqrt(((len(data[0]) - 1) * np.var(data[0], ddof=1) +
                           (len(data[1]) - 1) * np.var(data[1], ddof=1)) /
                          (len(data[0]) + len(data[1]) - 2))
    cohens_d = (np.mean(data[0]) - np.mean(data[1])) / pooled_std

    ax.set_ylabel("Per-Run Alert Rate (%)")
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    ax.set_title(f"Figure 4. Per-Run Alert Rate Distribution\n"
                 f"Welch's t = {t_stat:.2f}, {p_text}, Cohen's d = {cohens_d:.2f}")
    fig.tight_layout()
    save_all(fig, "figure4_statistical_significance")
    return t_stat, p_value, cohens_d


def main():
    summary_path = OUT_DIR / "per_run_summary.csv"
    sample_path = OUT_DIR / "measured_sample.csv"
    if summary_path.exists() and sample_path.exists():
        print("Reusing existing per_run_summary.csv / measured_sample.csv")
        per_run = pd.read_csv(summary_path)
        sample_df = pd.read_csv(sample_path)
    else:
        per_run, sample_df = stream_aggregate()

    figure1_alert_rate(per_run)
    figure2_data_completeness(per_run)
    figure3_distributions(sample_df)
    t_stat, p_value, cohens_d = figure4_significance(per_run)
    print(f"\nFigure 4 stats: t={t_stat:.4f} p={p_value:.6f} d={cohens_d:.4f}")
    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
