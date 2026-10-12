"""
conference_sim.py - Optimized for 200 runs for conference paper
Author: Your Name
Date: June 2024
Purpose: Monte Carlo simulation for IoT health monitoring reliability
"""

import numpy as np
import pandas as pd
import time
import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ========== OPTIMIZED CONFIGURATION FOR CONFERENCE ==========
np.random.seed(42)  # Reproducible results
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TIME OPTIMIZATIONS
SIMULATION_DT: float = 2.0  # 2-second steps (instead of 1) = 2x speedup
SIMULATION_TOTAL_TIME: int = 21600  # 6 hours (not 24) = 4x speedup
MONTE_CARLO_RUNS: int = 100  # Runs per environment (200 total)

# Physiological parameters (adult)
HEART_RATE_MIN: int = 40
HEART_RATE_MAX: int = 180
HEART_RATE_RESTING: float = 72.0
HEART_RATE_VARIABILITY: float = 0.08  # Reduced for faster convergence

BODY_TEMP_MIN: float = 35.0
BODY_TEMP_MAX: float = 41.0
BODY_TEMP_BASELINE: float = 36.8
BODY_TEMP_CIRCADIAN_AMPLITUDE: float = 0.4  # Reduced amplitude

# ========== SENSOR MODELS ==========
@dataclass
class SensorSpec:
name: str
sampling_interval: float
accuracy_offset: float
precision_std: float
packet_loss_prob: float

SENSOR_SPECS: Dict[str, SensorSpec] = {
"MAX30100": SensorSpec("MAX30100", 2.0, 0.0, 1.5, 0.02),
"DS18B20": SensorSpec("DS18B20", 4.0, -0.5, 0.2, 0.01),
"DHT11": SensorSpec("DHT11", 4.0, 0.0, 2.0, 0.03),
}

# ========== DECISION RULES (Optimized - 6 key rules) ==========
DECISION_RULES = [
# Rule 1: Hypothermia + Bradycardia
(None, 55, None, 36.0, "Hypothermia, Bradycardia", 1),
# Rule 2: High Fever + Tachycardia
(100, None, 38.0, None, "High Fever, Dehydration", 2),
# Rule 3: Mild Fever
(70, 100, 37.5, 38.5, "Fever, Overheating", 3),
# Rule 4: Anxiety/Stress (high HR, normal temp)
(90, None, 36.5, 37.5, "Anxiety, Stress", 4),
# Rule 5: Bradycardia only
(None, 60, 36.5, 37.5, "Potential Heart Issue", 5),
# Rule 6: Normal
(60, 100, 36.0, 37.5, "Normal", 0)
]

# ========== ENVIRONMENTS ==========
@dataclass
class Environment:
name: str
sensor_noise_scale: float = 1.0
packet_loss_scale: float = 1.0
network_latency_mean: float = 0.05
network_latency_std: float = 0.02

CONTROLLED_ENV = Environment("Controlled", 1.0, 1.0, 0.05, 0.02)
RESIDENTIAL_ENV = Environment("Residential", 1.5, 2.0, 0.15, 0.10)

# ========== PATIENT MODEL (Optimized) ==========
class Patient:
def __init__(self):
self.time = 0.0
self.heart_rate = HEART_RATE_RESTING
self.body_temp = BODY_TEMP_BASELINE
self.anomalies = []
self.anomaly_active = False
self.current_anomaly = None

def generate_anomalies(self):
"""Inject 0-1 anomaly during 6-hour simulation (conference scale)"""
num_anomalies = np.random.poisson(lam=0.5)  # λ=0.5 for 6 hours
num_anomalies = min(max(num_anomalies, 0), 1)  # 0 or 1 only

if num_anomalies > 0:
start = np.random.uniform(0.3, 0.7) * SIMULATION_TOTAL_TIME
duration = np.random.uniform(900, 3600)  # 15 min to 1 hour

anomaly_type = np.random.choice(['bradycardia', 'fever', 'tachycardia'])

if anomaly_type == 'bradycardia':
target_hr = np.random.uniform(45, 55)
self.anomalies.append((start, start + duration, 'bradycardia', 
{'target_hr': target_hr}))
elif anomaly_type == 'fever':
target_temp = np.random.uniform(38.5, 39.5)
self.anomalies.append((start, start + duration, 'fever', 
{'target_temp': target_temp}))
else:  # tachycardia
target_hr = np.random.uniform(110, 130)
self.anomalies.append((start, start + duration, 'tachycardia', 
{'target_hr': target_hr}))

def update(self, dt: float) -> Tuple[float, float, str]:
"""Update physiology and return (HR, Temp, Anomaly_Type)"""
self.time += dt

# Base heart rate: simplified random walk
hr_change = np.random.normal(0, 0.3)
self.heart_rate += hr_change

# Base temperature: simplified circadian
circadian = BODY_TEMP_CIRCADIAN_AMPLITUDE * np.sin(2 * np.pi * self.time / 21600)
temp_change = np.random.normal(0, 0.01)
self.body_temp = BODY_TEMP_BASELINE + circadian + temp_change

# Check for active anomaly
anomaly_type = 'none'
self.anomaly_active = False

for start, end, a_type, params in self.anomalies:
if start <= self.time < end:
self.anomaly_active = True
anomaly_type = a_type
progress = (self.time - start) / (end - start)

if a_type == 'bradycardia':
target = params['target_hr']
self.heart_rate = self.heart_rate * 0.8 + target * 0.2
elif a_type == 'fever':
target = params['target_temp']
self.body_temp = self.body_temp * 0.8 + target * 0.2
elif a_type == 'tachycardia':
target = params['target_hr']
self.heart_rate = self.heart_rate * 0.8 + target * 0.2

# Apply bounds
self.heart_rate = np.clip(self.heart_rate, HEART_RATE_MIN, HEART_RATE_MAX)
self.body_temp = np.clip(self.body_temp, BODY_TEMP_MIN, BODY_TEMP_MAX)

return self.heart_rate, self.body_temp, anomaly_type

# ========== SENSOR MODEL (Optimized) ==========
class Sensor:
def __init__(self, spec: SensorSpec, env: Environment):
self.spec = spec
self.env = env
self.last_sample_time = -spec.sampling_interval  # Force immediate sample
self.last_reading = None

def sample(self, true_value: float, current_time: float) -> Optional[float]:
"""Optimized sampling with vectorized noise"""
if current_time - self.last_sample_time < self.spec.sampling_interval:
return self.last_reading

self.last_sample_time = current_time

# Packet loss
loss_prob = self.spec.packet_loss_prob * self.env.packet_loss_scale
if np.random.random() < loss_prob:
self.last_reading = None
return None

# Add noise
noise_std = self.spec.precision_std * self.env.sensor_noise_scale
noisy_value = true_value + self.spec.accuracy_offset + np.random.normal(0, noise_std)

self.last_reading = noisy_value
return noisy_value

# ========== PROCESSOR (Decision Logic) ==========
class Processor:
def __init__(self, env: Environment):
self.env = env
self.sensor_buffer = {name: [] for name in SENSOR_SPECS.keys()}
self.buffer_capacity = 5  # Reduced buffer for speed

def add_reading(self, sensor_name: str, value: Optional[float]):
if value is not None:
self.sensor_buffer[sensor_name].append(value)
if len(self.sensor_buffer[sensor_name]) > self.buffer_capacity:
self.sensor_buffer[sensor_name].pop(0)

def evaluate(self) -> Tuple[bool, str, int, float]:
"""Return (alert_triggered, message, rule_id, latency)"""
hr = self.sensor_buffer["MAX30100"][-1] if self.sensor_buffer["MAX30100"] else None
temp = self.sensor_buffer["DS18B20"][-1] if self.sensor_buffer["DS18B20"] else None

if hr is None or temp is None:
return False, "Insufficient Data", -1, 0.0

# Apply rules
for hr_low, hr_high, temp_low, temp_high, message, rule_id in DECISION_RULES:
hr_check = (hr_low is None or hr >= hr_low) and (hr_high is None or hr <= hr_high)
temp_check = (temp_low is None or temp >= temp_low) and (temp_high is None or temp <= temp_high)

if hr_check and temp_check:
alert_triggered = rule_id != 0
latency = np.random.normal(self.env.network_latency_mean, 
self.env.network_latency_std)
latency = max(latency, 0.02)
return alert_triggered, message, rule_id, latency

return False, "No Rule Matched", -1, 0.0

# ========== SINGLE SIMULATION RUN ==========
def run_single_simulation(env: Environment, run_id: int) -> pd.DataFrame:
"""Run one 6-hour simulation"""
records = []

# Initialize
patient = Patient()
patient.generate_anomalies()

sensors = {name: Sensor(spec, env) for name, spec in SENSOR_SPECS.items()}
processor = Processor(env)

# Tracking for KPIs
detection_start_time = None
detection_times = []

# Time loop
for t in np.arange(0, SIMULATION_TOTAL_TIME, SIMULATION_DT):
# 1. Update patient
hr_true, temp_true, anomaly_type = patient.update(SIMULATION_DT)

# 2. Sensor readings
hr_measured = sensors["MAX30100"].sample(hr_true, t)
temp_measured = sensors["DS18B20"].sample(temp_true, t)
env_temp_measured = sensors["DHT11"].sample(25.0 + 3.0 * np.sin(2*np.pi*t/21600), t)

# 3. Feed processor
processor.add_reading("MAX30100", hr_measured)
processor.add_reading("DS18B20", temp_measured)
processor.add_reading("DHT11", env_temp_measured)

# 4. Evaluate
alert, message, rule_id, latency = processor.evaluate()

# 5. Track detection time for anomalies
if patient.anomaly_active and detection_start_time is None:
detection_start_time = t

if alert and patient.anomaly_active and detection_start_time:
detection_time = t - detection_start_time
detection_times.append(detection_time)
detection_start_time = None

# 6. Record
records.append({
'run_id': run_id,
'environment': env.name,
'time': t,
'hr_true': hr_true,
'temp_true': temp_true,
'hr_measured': hr_measured if hr_measured is not None else np.nan,
'temp_measured': temp_measured if temp_measured is not None else np.nan,
'env_temp': env_temp_measured if env_temp_measured is not None else np.nan,
'alert_triggered': alert,
'alert_message': message,
'rule_id': rule_id,
'processing_latency': latency,
'anomaly_active': patient.anomaly_active,
'anomaly_type': anomaly_type,
'data_missing': (hr_measured is None) or (temp_measured is None)
})

# Calculate mean detection time for this run
mean_detect = np.mean(detection_times) if detection_times else np.nan

# Add detection time to all records
for record in records:
record['mean_detection_time'] = mean_detect

return pd.DataFrame(records)

# ========== MAIN SIMULATION LOOP ==========
def main():
"""Run 200 simulations (100 Controlled + 100 Residential)"""
start_time = time.time()
all_results = []

print("=" * 70)
print("CONFERENCE SIMULATION: 200 RUNS (100 Controlled + 100 Residential)")
print("=" * 70)

environments = [("Controlled", CONTROLLED_ENV), ("Residential", RESIDENTIAL_ENV)]

for env_name, env in environments:
print(f"\nStarting {env_name} environment simulations...")

for run_id in range(MONTE_CARLO_RUNS):
# Progress indicator
if (run_id + 1) % 10 == 0:
elapsed = time.time() - start_time
est_total = elapsed / (run_id + 1) * MONTE_CARLO_RUNS
remaining = est_total - elapsed
print(f"  Run {run_id + 1}/{MONTE_CARLO_RUNS} - "
f"Elapsed: {elapsed/60:.1f} min - "
f"Remaining: {remaining/60:.1f} min")

# Run simulation
df_run = run_single_simulation(env, run_id)
all_results.append(df_run)

# Combine results
print("\nCombining results...")
final_df = pd.concat(all_results, ignore_index=True)

# Save results
output_file = "conference_results_200runs.csv"
final_df.to_csv(output_file, index=False)

# Save metadata
metadata = {
'simulation_time': time.time() - start_time,
'total_rows': len(final_df),
'runs_per_env': MONTE_CARLO_RUNS,
'simulation_hours': SIMULATION_TOTAL_TIME / 3600,
'time_step': SIMULATION_DT,
'environments': [env.name for env in [CONTROLLED_ENV, RESIDENTIAL_ENV]],
'sensor_specs': {k: v.__dict__ for k, v in SENSOR_SPECS.items()}
}

with open('conference_metadata.json', 'w') as f:
json.dump(metadata, f, indent=2)

# Print summary
elapsed_total = time.time() - start_time
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"Total time: {elapsed_total/60:.2f} minutes")
print(f"Results saved to: {output_file}")
print(f"Total data points: {len(final_df):,}")
print(f"File size: ~{len(final_df) * 100 / 1_000_000:.1f} MB")

# Quick statistics
print("\nQuick Statistics:")
for env in ['Controlled', 'Residential']:
env_data = final_df[final_df['environment'] == env]
alert_rate = env_data['alert_triggered'].mean() * 100
data_complete = (1 - env_data['data_missing'].mean()) * 100
avg_latency = env_data['processing_latency'].mean()

print(f"\n{env}:")
print(f"  Alert rate: {alert_rate:.2f}% of time steps")
print(f"  Data completeness: {data_complete:.1f}%")
print(f"  Avg latency: {avg_latency:.3f} sec")

return final_df

if __name__ == "__main__":
# Run the simulation
df = main()

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Generate figures: python conference_plots.py")
print("2. Calculate statistics: python conference_stats.py")
print("3. Write paper using IEEE conference template")
print("=" * 70)
