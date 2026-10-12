import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime
from full_spec_config import CONFIG

def run_single_simulation_real(run_id, env_name):
    """Run one 6-hour simulation"""
    sim_hours = CONFIG['SIMULATION_HOURS']
    dt = CONFIG['TIME_STEP_SECONDS']
    total_steps = int((sim_hours * 3600) / dt)
    
    records = []
    
    for t in np.arange(0, total_steps * dt, dt):
        # Generate realistic physiological data
        hr_true = 72.0 + np.random.normal(0, 5)
        temp_true = 36.8 + np.random.normal(0, 0.5)
        
        # Environment-specific noise
        if env_name == 'Residential':
            hr_noise = np.random.normal(0, 3.0)
            temp_noise = np.random.normal(0, 0.45)
            packet_loss = np.random.random() < 0.05
        else:
            hr_noise = np.random.normal(0, 1.5)
            temp_noise = np.random.normal(0, 0.2)
            packet_loss = np.random.random() < 0.02
        
        # Apply sensor noise and packet loss
        if packet_loss:
            hr_measured = np.nan
            temp_measured = np.nan
        else:
            hr_measured = hr_true + hr_noise
            temp_measured = temp_true + temp_noise
        
        # Simple alert logic
        alert = False
        if not np.isnan(hr_measured) and not np.isnan(temp_measured):
            if hr_measured > 100 or hr_measured < 55:
                alert = True
            if temp_measured > 38 or temp_measured < 35:
                alert = True
        
        # Record data (sampled to reduce size)
        if t % (5 * dt) == 0:  # Record every 10 seconds
            records.append({
                'run_id': run_id,
                'environment': env_name,
                'time': t,
                'hr_true': hr_true,
                'temp_true': temp_true,
                'hr_measured': hr_measured,
                'temp_measured': temp_measured,
                'alert_triggered': alert,
                'data_missing': np.isnan(hr_measured) or np.isnan(temp_measured)
            })
    
    return pd.DataFrame(records)

def create_batch_simulation(env_name, batch_id):
    """Run a batch of simulations"""
    print(f"\n{'='*60}")
    print(f"FULL 1000-RUN SIMULATION")
    print(f"Batch: {env_name.upper()} - ID: {batch_id}")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    runs_per_env = CONFIG['TOTAL_RUNS'] // len(CONFIG['ENVIRONMENTS'])
    batch_size = CONFIG['RUNS_PER_BATCH']
    
    start_run = batch_id * batch_size
    end_run = min((batch_id + 1) * batch_size, runs_per_env)
    
    print(f"Running runs {start_run} to {end_run-1}")
    print(f"Batch size: {end_run - start_run} runs")
    print(f"Each run: {CONFIG['SIMULATION_HOURS']} hours")
    print(f"Time step: {CONFIG['TIME_STEP_SECONDS']} seconds")
    
    all_results = []
    batch_start = time.time()
    
    # Run each simulation in the batch
    for run_id in range(start_run, end_run):
        run_start = time.time()
        
        # Progress update
        if (run_id - start_run) % 10 == 0:
            elapsed = time.time() - batch_start
            completed = run_id - start_run
            if completed > 0:
                avg_time = elapsed / completed
                remaining = avg_time * (end_run - run_id)
                print(f"  Run {run_id+1}/{end_run} - {elapsed:.1f}s - {remaining/60:.1f} min left")
            else:
                print(f"  Run {run_id+1}/{end_run} - Starting...")
        
        # Run simulation
        df_run = run_single_simulation_real(run_id, env_name)
        all_results.append(df_run)
        
        run_time = time.time() - run_start
        if run_time > 1.0:
            print(f"    Run {run_id} took {run_time:.1f} seconds")
    
    # Combine and save results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Create output directory
        output_dir = CONFIG['OUTPUT_DIR']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to efficient parquet format
        output_file = f"{output_dir}/{env_name}_batch_{batch_id}.parquet"
        combined_df.to_parquet(output_file, engine='pyarrow', compression='snappy')
        
        # Calculate statistics
        batch_time = time.time() - batch_start
        avg_run_time = batch_time / (end_run - start_run)
        
        print(f"\n{'='*60}")
        print(f"✓ BATCH COMPLETED SUCCESSFULLY")
        print(f"  Output file: {output_file}")
        print(f"  File size: {os.path.getsize(output_file)/1024/1024:.2f} MB")
        print(f"  Total rows: {len(combined_df):,}")
        print(f"  Batch duration: {batch_time:.2f} seconds ({batch_time/60:.2f} minutes)")
        print(f"  Average per run: {avg_run_time:.2f} seconds")
        print(f"  End time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        return combined_df
    
    return pd.DataFrame()

def main():
    """Main function - check arguments and run batch"""
    # Check command line arguments
    if len(sys.argv) != 3:
        print("\n" + "="*60)
        print("ERROR: Incorrect number of arguments!")
        print("="*60)
        print("Usage: python run_batch_integrated.py <environment> <batch_id>")
        print("\nExamples:")
        print("  python run_batch_integrated.py Controlled 0")
        print("  python run_batch_integrated.py Residential 3")
        print("\nValid environments:", CONFIG['ENVIRONMENTS'])
        
        # Calculate valid batch IDs
        runs_per_env = CONFIG['TOTAL_RUNS'] // len(CONFIG['ENVIRONMENTS'])
        batch_size = CONFIG['RUNS_PER_BATCH']
        max_batches = (runs_per_env + batch_size - 1) // batch_size
        
        print(f"Valid batch IDs: 0 to {max_batches-1}")
        print(f"Total runs per environment: {runs_per_env}")
        print(f"Runs per batch: {batch_size}")
        print("="*60)
        sys.exit(1)
    
    # Get arguments
    env_name = sys.argv[1]
    batch_id = int(sys.argv[2])
    
    # Validate environment
    if env_name not in CONFIG['ENVIRONMENTS']:
        print(f"\nERROR: Invalid environment '{env_name}'")
        print(f"Valid options: {CONFIG['ENVIRONMENTS']}")
        sys.exit(1)
    
    # Validate batch ID
    runs_per_env = CONFIG['TOTAL_RUNS'] // len(CONFIG['ENVIRONMENTS'])
    batch_size = CONFIG['RUNS_PER_BATCH']
    max_batches = (runs_per_env + batch_size - 1) // batch_size
    
    if batch_id < 0 or batch_id >= max_batches:
        print(f"\nERROR: Invalid batch_id {batch_id}")
        print(f"Valid range: 0 to {max_batches-1}")
        sys.exit(1)
    
    # Run the batch
    create_batch_simulation(env_name, batch_id)

if __name__ == "__main__":
    main()

