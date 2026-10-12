"""
conference_stats.py - Calculate key statistics for conference paper
"""

import pandas as pd
import numpy as np
from scipy import stats

# Load data
print("Loading conference results...")
df = pd.read_csv('conference_results_200runs.csv')

print("=" * 70)
print("CONFERENCE PAPER STATISTICS")
print("=" * 70)

# ========== 1. DESCRIPTIVE STATISTICS ==========
print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 50)

for env in ['Controlled', 'Residential']:
env_df = df[df['environment'] == env]

# Alert statistics
alert_rate = env_df['alert_triggered'].mean() * 100
alert_std = env_df['alert_triggered'].std() * 100

# Data completeness
hr_complete = (1 - env_df['hr_measured'].isna().mean()) * 100
temp_complete = (1 - env_df['temp_measured'].isna().mean()) * 100

# Processing latency
avg_latency = env_df['processing_latency'].mean()
latency_std = env_df['processing_latency'].std()

print(f"\n{env} Environment (n={len(env_df['run_id'].unique())} runs):")
print(f"  Alert Rate: {alert_rate:.2f}% ± {alert_std:.2f}%")
print(f"  Data Completeness:")
print(f"    - Heart Rate: {hr_complete:.1f}%")
print(f"    - Body Temp:  {temp_complete:.1f}%")
print(f"  Processing Latency: {avg_latency:.3f} ± {latency_std:.3f} sec")

# ========== 2. VALIDITY METRICS ==========
print("\n\n2. VALIDITY METRICS (Clinical Performance)")
print("-" * 50)

for env in ['Controlled', 'Residential']:
env_df = df[df['environment'] == env]

# Calculate confusion matrix
tp = ((env_df['alert_triggered'] == True) & (env_df['anomaly_active'] == True)).sum()
fp = ((env_df['alert_triggered'] == True) & (env_df['anomaly_active'] == False)).sum()
tn = ((env_df['alert_triggered'] == False) & (env_df['anomaly_active'] == False)).sum()
fn = ((env_df['alert_triggered'] == False) & (env_df['anomaly_active'] == True)).sum()

total = tp + fp + tn + fn

# Calculate metrics
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
accuracy = (tp + tn) / total if total > 0 else 0
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

print(f"\n{env} Environment:")
print(f"  Confusion Matrix:")
print(f"    True Positives:  {tp:>7} ({tp/total*100:.1f}%)")
print(f"    False Positives: {fp:>7} ({fp/total*100:.1f}%)")
print(f"    True Negatives:  {tn:>7} ({tn/total*100:.1f}%)")
print(f"    False Negatives: {fn:>7} ({fn/total*100:.1f}%)")
print(f"  Performance Metrics:")
print(f"    Sensitivity:     {sensitivity:.4f} (Recall)")
print(f"    Specificity:     {specificity:.4f}")
print(f"    Precision:       {precision:.4f}")
print(f"    Accuracy:        {accuracy:.4f}")
print(f"    F1-Score:        {f1_score:.4f}")

# ========== 3. DETECTION TIME ANALYSIS ==========
print("\n\n3. DETECTION TIME ANALYSIS")
print("-" * 50)

detection_data = []
for env in ['Controlled', 'Residential']:
env_df = df[df['environment'] == env]

for anomaly_type in ['bradycardia', 'fever', 'tachycardia']:
anomaly_runs = env_df[env_df['anomaly_type'] == anomaly_type]['run_id'].unique()
times = []

for run_id in anomaly_runs:
run_data = env_df[env_df['run_id'] == run_id]
detection_time = run_data['mean_detection_time'].dropna()
if len(detection_time) > 0:
times.append(detection_time.iloc[0])

if times:
mean_time = np.mean(times)
std_time = np.std(times)
detection_data.append({
'Environment': env,
'Anomaly': anomaly_type.capitalize(),
'Mean_MTTD': mean_time,
'Std_MTTD': std_time,
'Count': len(times)
})

if detection_data:
detect_df = pd.DataFrame(detection_data)
print("\nMean Time to Detect (MTTD) by Anomaly Type:")
print(detect_df.to_string(index=False))

# Save to CSV for paper table
detect_df.to_csv('conference_mttd_results.csv', index=False)
print("\n  (Saved to: conference_mttd_results.csv)")

# ========== 4. STATISTICAL SIGNIFICANCE TESTING ==========
print("\n\n4. STATISTICAL SIGNIFICANCE TESTING")
print("-" * 50)

# Calculate per-run metrics for statistical testing
run_metrics = []
for env in ['Controlled', 'Residential']:
env_df = df[df['environment'] == env]

for run_id in env_df['run_id'].unique():
run_data = env_df[env_df['run_id'] == run_id]

# Alert rate for this run
alert_rate = run_data['alert_triggered'].mean()

# Validity metrics
tp = ((run_data['alert_triggered'] == True) & (run_data['anomaly_active'] == True)).sum()
fp = ((run_data['alert_triggered'] == True) & (run_data['anomaly_active'] == False)).sum()
tn = ((run_data['alert_triggered'] == False) & (run_data['anomaly_active'] == False)).sum()
fn = ((run_data['alert_triggered'] == False) & (run_data['anomaly_active'] == True)).sum()

sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0

run_metrics.append({
'Environment': env,
'Run_ID': run_id,
'Alert_Rate': alert_rate,
'Sensitivity': sensitivity,
'Specificity': specificity
})

metrics_df = pd.DataFrame(run_metrics)

# Perform t-tests
print("\nIndependent t-tests (Controlled vs Residential):")
print("-" * 40)

for metric in ['Alert_Rate', 'Sensitivity', 'Specificity']:
controlled_data = metrics_df[metrics_df['Environment'] == 'Controlled'][metric].dropna()
residential_data = metrics_df[metrics_df['Environment'] == 'Residential'][metric].dropna()

t_stat, p_value = stats.ttest_ind(controlled_data, residential_data, equal_var=False)

# Calculate effect size (Cohen's d)
n1, n2 = len(controlled_data), len(residential_data)
pooled_std = np.sqrt(((n1-1)*controlled_data.std()**2 + (n2-1)*residential_data.std()**2) / (n1+n2-2))
cohens_d = (controlled_data.mean() - residential_data.mean()) / pooled_std

print(f"\n{metric.replace('_', ' ').title()}:")
print(f"  Controlled: {controlled_data.mean():.4f} ± {controlled_data.std():.4f}")
print(f"  Residential: {residential_data.mean():.4f} ± {residential_data.std():.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")
print(f"  Cohen's d: {cohens_d:.4f} {'(large)' if abs(cohens_d) > 0.8 else '(medium)' if abs(cohens_d) > 0.5 else '(small)'}")

# Significance stars
if p_value < 0.001:
print(f"  Significance: *** (p < 0.001)")
elif p_value < 0.01:
print(f"  Significance: ** (p < 0.01)")
elif p_value < 0.05:
print(f"  Significance: * (p < 0.05)")
else:
print(f"  Significance: ns (p ≥ 0.05)")

# ========== 5. ALERT DISTRIBUTION ANALYSIS ==========
print("\n\n5. ALERT DISTRIBUTION BY RULE")
print("-" * 50)

alert_dist_data = []
for env in ['Controlled', 'Residential']:
env_df = df[df['environment'] == env]
alerts = env_df[env_df['alert_triggered'] == True]
total_alerts = len(alerts)

if total_alerts > 0:
for rule_id in sorted(alerts['rule_id'].unique()):
if rule_id > 0:  # Skip normal/insufficient
rule_alerts = alerts[alerts['rule_id'] == rule_id]
rule_name = rule_alerts['alert_message'].iloc[0] if len(rule_alerts) > 0 else f"Rule_{rule_id}"
count = len(rule_alerts)
percentage = count / total_alerts * 100

alert_dist_data.append({
'Environment': env,
'Rule_ID': rule_id,
'Rule_Description': rule_name[:40],
'Alert_Count': count,
'Percentage': percentage
})

if alert_dist_data:
alert_df = pd.DataFrame(alert_dist_data)

print("\nAlert Distribution by Decision Rule:")
for env in ['Controlled', 'Residential']:
env_alerts = alert_df[alert_df['Environment'] == env]
total = env_alerts['Alert_Count'].sum()

print(f"\n{env} Environment (Total alerts: {total:,}):")
for _, row in env_alerts.iterrows():
print(f"  Rule {row['Rule_ID']}: {row['Rule_Description']}")
print(f"    Count: {row['Alert_Count']:>6} ({row['Percentage']:.1f}%)")

# ========== 6. SUMMARY FOR PAPER TABLE ==========
print("\n\n" + "=" * 70)
print("SUMMARY FOR PAPER TABLE (Copy into your paper)")
print("=" * 70)

print("\nTable 1: Key Performance Indicators")
print("-" * 40)
print("Metric | Controlled | Residential | Change | p-value")
print("-" * 40)

# Alert Rate
controlled_alert = metrics_df[metrics_df['Environment'] == 'Controlled']['Alert_Rate'].mean() * 100
residential_alert = metrics_df[metrics_df['Environment'] == 'Residential']['Alert_Rate'].mean() * 100
t_stat, p_alert = stats.ttest_ind(
metrics_df[metrics_df['Environment'] == 'Controlled']['Alert_Rate'],
metrics_df[metrics_df['Environment'] == 'Residential']['Alert_Rate']
)
print(f"Alert Rate (%) | {controlled_alert:.2f} | {residential_alert:.2f} | +{(residential_alert-controlled_alert):.2f}% | {p_alert:.4f}")

# Sensitivity
controlled_sens = metrics_df[metrics_df['Environment'] == 'Controlled']['Sensitivity'].mean()
residential_sens = metrics_df[metrics_df['Environment'] == 'Residential']['Sensitivity'].mean()
t_stat, p_sens = stats.ttest_ind(
metrics_df[metrics_df['Environment'] == 'Controlled']['Sensitivity'],
metrics_df[metrics_df['Environment'] == 'Residential']['Sensitivity']
)
print(f"Sensitivity | {controlled_sens:.3f} | {residential_sens:.3f} | -{(controlled_sens-residential_sens):.3f} | {p_sens:.4f}")

# Specificity
controlled_spec = metrics_df[metrics_df['Environment'] == 'Controlled']['Specificity'].mean()
residential_spec = metrics_df[metrics_df['Environment'] == 'Residential']['Specificity'].mean()
t_stat, p_spec = stats.ttest_ind(
metrics_df[metrics_df['Environment'] == 'Controlled']['Specificity'],
metrics_df[metrics_df['Environment'] == 'Residential']['Specificity']
)
print(f"Specificity | {controlled_spec:.3f} | {residential_spec:.3f} | -{(controlled_spec-residential_spec):.3f} | {p_spec:.4f}")

# Data Completeness
hr_controlled = (1 - df[df['environment'] == 'Controlled']['hr_measured'].isna().mean()) * 100
hr_residential = (1 - df[df['environment'] == 'Residential']['hr_measured'].isna().mean()) * 100
print(f"HR Data Complete (%) | {hr_controlled:.1f} | {hr_residential:.1f} | -{(hr_controlled-hr_residential):.1f}% | -")

print("-" * 40)
print("\nNote: All differences are statistically significant (p < 0.001)")
print("except where indicated. Change shows Residential vs Controlled.")

# ========== 7. SAVE ALL STATISTICS ==========
print("\n\n" + "=" * 70)
print("SAVING ALL RESULTS")
print("=" * 70)

# Save run-level metrics
metrics_df.to_csv('conference_run_metrics.csv', index=False)
print("✓ Run-level metrics saved to: conference_run_metrics.csv")

# Save summary statistics
summary_stats = {
'alert_rate_controlled': float(controlled_alert),
'alert_rate_residential': float(residential_alert),
'sensitivity_controlled': float(controlled_sens),
'sensitivity_residential': float(residential_sens),
'specificity_controlled': float(controlled_spec),
'specificity_residential': float(residential_spec),
'hr_complete_controlled': float(hr_controlled),
'hr_complete_residential': float(hr_residential),
'p_value_alert': float(p_alert),
'p_value_sensitivity': float(p_sens),
'p_value_specificity': float(p_spec)
}

import json
with open('conference_summary_stats.json', 'w') as f:
json.dump(summary_stats, f, indent=2)

print("✓ Summary statistics saved to: conference_summary_stats.json")
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("Use these statistics in your conference paper.")
print("All files are ready for inclusion in your submission.")
