"""
run_batch.py - Run one resumable batch of full-length simulations.

Usage:
    python run_batch.py <environment> <batch_id>
    python run_batch.py Controlled 0
    python run_batch.py Residential 3

Each batch writes one file to CONFIG['OUTPUT_DIR']. Run all batches for both
environments (see the printed "Valid batch IDs" range) to get the full
N=1000 (500 Controlled + 500 Residential) dataset, then run
analyze_results.py to aggregate.

Batches are seeded deterministically from (environment, run_id) so re-running
a batch reproduces the same numbers.
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd

from config import CONFIG
from sim_engine import run_single_simulation


def seed_for(env_name: str, run_id: int) -> int:
    # Deterministic, distinct seed per (environment, run_id) pair.
    base = 0 if env_name == "Controlled" else 1_000_000
    return base + run_id


def run_batch(env_name: str, batch_id: int):
    runs_per_env = CONFIG["TOTAL_RUNS"] // len(CONFIG["ENVIRONMENTS"])
    batch_size = CONFIG["RUNS_PER_BATCH"]
    start_run = batch_id * batch_size
    end_run = min((batch_id + 1) * batch_size, runs_per_env)

    print("=" * 70)
    print(f"BATCH: {env_name.upper()} - batch {batch_id} (runs {start_run}-{end_run - 1})")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 70)

    all_dfs = []
    all_mttd = []
    t0 = time.time()

    for run_id in range(start_run, end_run):
        df_run = run_single_simulation(env_name, run_id, seed_for(env_name, run_id))
        all_dfs.append(df_run)
        all_mttd.extend(df_run.attrs.get("mttd_records", []))

        if (run_id - start_run + 1) % 5 == 0 or run_id == end_run - 1:
            elapsed = time.time() - t0
            done = run_id - start_run + 1
            avg = elapsed / done
            remaining = avg * (end_run - start_run - done)
            print(f"  run {run_id + 1}/{end_run}  elapsed={elapsed:.1f}s  est.remaining={remaining:.1f}s")

    combined = pd.concat(all_dfs, ignore_index=True)
    mttd_df = pd.DataFrame(all_mttd)

    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    base = f"{output_dir}/{env_name}_batch_{batch_id}"
    try:
        import pyarrow  # noqa: F401
        combined.to_parquet(f"{base}.parquet", engine="pyarrow", compression="snappy")
        out_path = f"{base}.parquet"
    except ImportError:
        combined.to_csv(f"{base}.csv", index=False)
        out_path = f"{base}.csv"

    mttd_df.to_csv(f"{base}_mttd.csv", index=False)

    elapsed_total = time.time() - t0
    print(f"\nSaved: {out_path}  ({len(combined):,} rows)")
    print(f"Saved: {base}_mttd.csv  ({len(mttd_df)} detected anomalies)")
    print(f"Batch duration: {elapsed_total:.1f}s ({elapsed_total / max(end_run - start_run, 1):.2f}s/run)")


def main():
    if len(sys.argv) != 3:
        runs_per_env = CONFIG["TOTAL_RUNS"] // len(CONFIG["ENVIRONMENTS"])
        batch_size = CONFIG["RUNS_PER_BATCH"]
        max_batches = (runs_per_env + batch_size - 1) // batch_size
        print("Usage: python run_batch.py <environment> <batch_id>")
        print(f"Valid environments: {CONFIG['ENVIRONMENTS']}")
        print(f"Valid batch IDs: 0 to {max_batches - 1} (runs_per_env={runs_per_env}, batch_size={batch_size})")
        sys.exit(1)

    env_name, batch_id = sys.argv[1], int(sys.argv[2])
    if env_name not in CONFIG["ENVIRONMENTS"]:
        print(f"ERROR: invalid environment '{env_name}'. Valid: {CONFIG['ENVIRONMENTS']}")
        sys.exit(1)

    run_batch(env_name, batch_id)


if __name__ == "__main__":
    main()
