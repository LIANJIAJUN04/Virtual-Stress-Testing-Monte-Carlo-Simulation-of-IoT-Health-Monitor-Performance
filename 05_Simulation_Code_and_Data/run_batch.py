"""
run_batch.py - Run a single batch of simulations
Usage: python run_batch.py <environment> <batch_id>
"""
import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from full_spec_config import CONFIG

def run_single_simulation(run_id, env_name):
"""Run one simulation - simplified for testing"""
sim_hours = CONFIG['SIMULATION_HOURS']
dt = CONFIG['TIME_STEP_SECONDS']
total_steps = int((sim_hours * 3600) / dt)

records = []

for t in np.arange(0, total_steps * dt, dt):
# Test data - replace with your actual simulation
hr_true = 72.0 + np.random.normal(0, 5)
temp_true = 36.8 + np.random.normal(0, 0.5)

if env_name == 'Residential':
noise_scale = 1.5
else:
noise_scale = 1.0

hr_measured = hr_true + np.random.normal(0, 2 * noise_scale)
temp_measured = temp_true + np.random.normal(0, 0.3 * noise_scale)
alert_triggered = (hr_measured > 100) or (hr_measured < 55) or (temp_measured > 38)

# Record every 10 time steps
if t % (10 * dt) == 0:
records.append({
'run_id': run_id,
'environment': env_name,
'time': t,
'hr_true': hr_true,
'temp_true': temp_true,
'hr_measured': hr_measured,
'temp_measured': temp_measured,
'alert_triggered': alert_triggered
})

return pd.DataFrame(records)

def create_batch_simulation(env_name, batch_id):
"""Create and run a batch simulation"""
print(f"\n{'='*60}")
print(f"BATCH: {env_name.upper()} - BATCH {batch_id}")
print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*60}")

runs_per_env = CONFIG['TOTAL_RUNS'] // len(CONFIG['ENVIRONMENTS'])
batch_size = CONFIG['RUNS_PER_BATCH']

start_run = batch_id * batch_size
end_run = min((batch_id + 1) * batch_size, runs_per_env)

print(f"Running runs {start_run} to {end_run-1}")

all_results = []
start_time = time.time()

for run_id in range(start_run, end_run):
if run_id % 2 == 0:
elapsed = time.time() - start_time
print(f"  Run {run_id+1}/{end_run} - {elapsed:.1f}s")

df_run = run_single_simulation(run_id, env_name)
all_results.append(df_run)

if all_results:
combined_df = pd.concat(all_results, ignore_index=True)
output_dir = CONFIG['OUTPUT_DIR']
os.makedirs(output_dir, exist_ok=True)

output_file = f"{output_dir}/{env_name}_batch_{batch_id}.parquet"
combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')

print(f"\n✓ Saved: {output_file}")
print(f"  Size: {os.path.getsize(output_file)/1024/1024:.2f} MB")
print(f"  Rows: {len(combined_df):,}")

return combined_df

return pd.DataFrame()

def main():
if len(sys.argv) != 3:
print("\nERROR: Missing arguments!")
print("Usage: python run_batch.py <environment> <batch_id>")
print("Example: python run_batch.py Controlled 0")
print("\nValid environments:", CONFIG['ENVIRONMENTS'])
sys.exit(1)

env_name = sys.argv[1]
batch_id = int(sys.argv[2])

if env_name not in CONFIG['ENVIRONMENTS']:
print(f"ERROR: Invalid environment '{env_name}'")
print(f"Valid: {CONFIG['ENVIRONMENTS']}")
sys.exit(1)

result_df = create_batch_simulation(env_name, batch_id)
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"BATCH COMPLETED")
print(f"Environment: {env_name}")
print(f"Batch ID: {batch_id}")
print(f"Time: {elapsed:.2f} seconds")
print(f"End: {datetime.now().strftime('%H:%M:%S')}")
print(f"{'='*60}")

if __name__ == "__main__":
start_time = time.time()
main()
