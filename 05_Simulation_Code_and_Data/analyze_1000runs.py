import pandas as pd
import numpy as np
import glob
import os
from scipy import stats

print("="*70)
print("ANALYZING 1000-RUN SIMULATION RESULTS")
print("="*70)

# Load all parquet files
files = glob.glob('full_spec_results/*.parquet')
print(f"Found {len(files)} result files")

if len(files) == 0:
    print("ERROR: No parquet files found in full_spec_results/")
    print("Run some batches first with: python3 run_batch_integrated.py Controlled 0")
    exit(1)

all_dfs = []
for f in files:
    df = pd.read_parquet(f)
    all_dfs.append(df)
    print(f"  Loaded {f}: {len(df):,} rows")

full_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal data loaded:")
print(f"  Total rows: {len(full_df):,}")
print(f"  Total runs: {full_df['run_id'].nunique()}")
print(f"  Environments: {full_df['environment'].unique().tolist()}")

# Calculate metrics by environment
print("\n" + "="*70)
print("PERFORMANCE METRICS BY ENVIRONMENT")
print("="*70)

metrics_data = []
for env in sorted(full_df['environment'].unique()):
    env_df = full_df[full_df['environment'] == env]
    
    alert_rate = env_df['alert_triggered'].mean() * 100
    data_missing = env_df['data_missing'].mean() * 100
    
    hr_measured = env_df['hr_measured'].dropna()
    temp_measured = env_df['temp_measured'].dropna()
    
    print(f"\n{env.upper()} Environment:")
    print(f"  Alert rate: {alert_rate:.2f}%")
    print(f"  Data missing rate: {data_missing:.2f}%")
    print(f"  Data completeness: {100 - data_missing:.1f}%")
    
    if len(hr_measured) > 0:
        print(f"  Heart rate range: {hr_measured.min():.1f} - {hr_measured.max():.1f} BPM")
        print(f"  Avg heart rate: {hr_measured.mean():.1f} BPM")
    
    if len(temp_measured) > 0:
        print(f"  Temperature range: {temp_measured.min():.1f} - {temp_measured.max():.1f} °C")
        print(f"  Avg temperature: {temp_measured.mean():.1f} °C")
    
    print(f"  Runs completed: {env_df['run_id'].nunique()}")
    print(f"  Data points: {len(env_df):,}")
    
    metrics_data.append({
        'Environment': env,
        'Alert_Rate': alert_rate,
        'Data_Missing': data_missing,
        'Runs': env_df['run_id'].nunique(),
        'Data_Points': len(env_df)
    })

# Statistical comparison
print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

# Get alert rates per run for each environment
alert_by_run = full_df.groupby(['environment', 'run_id'])['alert_triggered'].mean().reset_index()

for env in ['Controlled', 'Residential']:
    if env in alert_by_run['environment'].unique():
        env_alerts = alert_by_run[alert_by_run['environment'] == env]['alert_triggered']
        print(f"{env} alert rate: {env_alerts.mean()*100:.2f}% ± {env_alerts.std()*100:.2f}%")

# Compare if both environments exist
envs_in_data = alert_by_run['environment'].unique()
if 'Controlled' in envs_in_data and 'Residential' in envs_in_data:
    controlled_alerts = alert_by_run[alert_by_run['environment'] == 'Controlled']['alert_triggered']
    residential_alerts = alert_by_run[alert_by_run['environment'] == 'Residential']['alert_triggered']
    
    if len(controlled_alerts) > 1 and len(residential_alerts) > 1:
        t_stat, p_value = stats.ttest_ind(controlled_alerts, residential_alerts)
        
        print(f"\nAlert Rate Comparison (t-test):")
        print(f"  Controlled: {controlled_alerts.mean()*100:.2f}%")
        print(f"  Residential: {residential_alerts.mean()*100:.2f}%")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        
        if p_value < 0.001:
            significance = "*** (p < 0.001)"
        elif p_value < 0.01:
            significance = "** (p < 0.01)"
        elif p_value < 0.05:
            significance = "* (p < 0.05)"
        else:
            significance = "ns (not significant)"
        
        print(f"  Significance: {significance}")
else:
    print(f"\nNote: Only found data for environments: {envs_in_data}")

# Save summary
import json
summary = {
    'total_simulations': int(full_df['run_id'].nunique()),
    'total_data_points': int(len(full_df)),
    'environments': {}
}

for env in full_df['environment'].unique():
    env_df = full_df[full_df['environment'] == env]
    hr_mean = float(env_df['hr_measured'].mean()) if not env_df['hr_measured'].isna().all() else None
    temp_mean = float(env_df['temp_measured'].mean()) if not env_df['temp_measured'].isna().all() else None
    
    summary['environments'][env] = {
        'runs': int(env_df['run_id'].nunique()),
        'alert_rate': float(env_df['alert_triggered'].mean()),
        'data_missing': float(env_df['data_missing'].mean()),
        'avg_heart_rate': hr_mean,
        'avg_temperature': temp_mean
    }

with open('simulation_summary_1000runs.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*70)
print("SUMMARY SAVED TO: simulation_summary_1000runs.json")
print("="*70)

# Quick stats for paper
print("\nFOR YOUR PAPER ABSTRACT:")
print("-"*40)
if 'Controlled' in summary['environments'] and 'Residential' in summary['environments']:
    c_alert = summary['environments']['Controlled']['alert_rate'] * 100
    r_alert = summary['environments']['Residential']['alert_rate'] * 100
    c_missing = summary['environments']['Controlled']['data_missing'] * 100
    r_missing = summary['environments']['Residential']['data_missing'] * 100
    
    print(f"Alert rates: Controlled={c_alert:.1f}%, Residential={r_alert:.1f}%")
    print(f"Data completeness: Controlled={100-c_missing:.1f}%, Residential={100-r_missing:.1f}%")
    print(f"Relative difference: {(r_alert-c_alert):.1f}% increase in alert rate")
    
    if 'p_value' in locals():
        if p_value < 0.05:
            print(f"Statistical significance: Significant (p={p_value:.4f})")
        else:
            print(f"Statistical significance: Not significant (p={p_value:.4f})")
print("-"*40)

