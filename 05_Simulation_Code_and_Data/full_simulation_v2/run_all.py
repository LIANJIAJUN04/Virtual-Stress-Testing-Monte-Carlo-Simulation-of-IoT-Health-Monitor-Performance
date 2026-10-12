"""
run_all.py - Run every batch for every environment sequentially, in one process.

Equivalent to calling `python run_batch.py <env> <batch_id>` for every valid
(env, batch_id) pair, but avoids per-process Python startup overhead and
gives one continuous log for the full N=1000 run.

Usage: python run_all.py
"""

import time

from config import CONFIG
from run_batch import run_batch


def main():
    runs_per_env = CONFIG["TOTAL_RUNS"] // len(CONFIG["ENVIRONMENTS"])
    batch_size = CONFIG["RUNS_PER_BATCH"]
    max_batches = (runs_per_env + batch_size - 1) // batch_size

    print(f"Full run: {CONFIG['TOTAL_RUNS']} total runs, "
          f"{runs_per_env} per environment, {max_batches} batches/env, "
          f"{len(CONFIG['ENVIRONMENTS'])} environments "
          f"= {max_batches * len(CONFIG['ENVIRONMENTS'])} batches total")

    t_start = time.time()
    for env_name in CONFIG["ENVIRONMENTS"]:
        for batch_id in range(max_batches):
            run_batch(env_name, batch_id)

    print("\n" + "=" * 70)
    print(f"ALL BATCHES COMPLETE. Total elapsed: {(time.time() - t_start) / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
