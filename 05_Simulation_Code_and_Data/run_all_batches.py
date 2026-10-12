"""
run_all_batches.py - Run ALL batches
Usage: python run_all_batches.py
"""
import subprocess
import time
import os
from datetime import datetime
from full_spec_config import CONFIG

def run_single_batch(env, batch_id):
"""Run a single batch"""
print(f"  Starting: {env} batch {batch_id}")
cmd = ['python', 'run_batch.py', env, str(batch_id)]

try:
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode == 0:
print(f"  ✓ Completed: {env} batch {batch_id}")
return True, ""
else:
error_msg = f"Failed: {result.returncode}"
if result.stderr:
error_msg += f"\nError: {result.stderr[:100]}"
return False, error_msg
except Exception as e:
return False, f"Error: {str(e)}"

def main():
print(f"\n{'='*70}")
print("SIMULATION CONTROLLER")
print(f"{'='*70}")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runs: {CONFIG['TOTAL_RUNS']}")
print(f"Runs per batch: {CONFIG['RUNS_PER_BATCH']}")
print(f"Output dir: {CONFIG['OUTPUT_DIR']}")
print(f"{'='*70}\n")

os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

runs_per_env = CONFIG['TOTAL_RUNS'] // len(CONFIG['ENVIRONMENTS'])
batches_per_env = (runs_per_env + CONFIG['RUNS_PER_BATCH'] - 1) // CONFIG['RUNS_PER_BATCH']
total_batches = batches_per_env * len(CONFIG['ENVIRONMENTS'])

print(f"Batches per env: {batches_per_env}")
print(f"Total batches: {total_batches}")
print(f"\n{'='*70}")

completed = 0
failed = []
overall_start = time.time()

for env in CONFIG['ENVIRONMENTS']:
print(f"\n{'='*50}")
print(f"PROCESSING: {env.upper()}")
print(f"{'='*50}")

for batch_id in range(batches_per_env):
batch_start = time.time()
print(f"\n[Batch {completed+1}/{total_batches}]")

success, error = run_single_batch(env, batch_id)
batch_time = time.time() - batch_start

if success:
completed += 1
print(f"  Time: {batch_time:.2f}s")

if completed > 0:
avg_time = (time.time() - overall_start) / completed
remaining = avg_time * (total_batches - completed)
print(f"  Est. remaining: {remaining/60:.1f} min")
else:
failed.append((env, batch_id, error))
print(f"  ✗ FAILED: {error}")

total_time = time.time() - overall_start

print(f"\n{'='*70}")
print("SIMULATION COMPLETE")
print(f"{'='*70}")
print(f"Total time: {total_time/60:.2f} minutes")
print(f"Completed: {completed}/{total_batches}")
print(f"Failed: {len(failed)}")

if os.path.exists(CONFIG['OUTPUT_DIR']):
files = os.listdir(CONFIG['OUTPUT_DIR'])
print(f"Files created: {len(files)}")

print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")

if __name__ == "__main__":
main()
