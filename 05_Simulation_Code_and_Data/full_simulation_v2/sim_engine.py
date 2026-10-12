"""
sim_engine.py - Core discrete-event simulation for one 24-hour run.

Implements the mechanisms described in the manuscript's Methodology section:
  - Heart rate as a true Ornstein-Uhlenbeck process (mean-reverting to 72 BPM)
  - Circadian body-temperature model + additive noise
  - Poisson-distributed anomaly injection (bradycardia / tachycardia / fever)
  - Three sensor models (MAX30100, DS18B20, DHT11) with per-sensor sampling
    interval, accuracy offset, noise, and packet loss
  - The 8-condition rule table from Table 2 (see config.py for the exact
    rules and the documented "Undefined = missing reading" interpretation)

This supersedes conference_sim.py / run_batch.py / run_batch_integrated.py,
which had (a) corrupted indentation making them non-executable, and (b) logic
that did not match the manuscript text (plain random walk instead of OU,
only 6 of the 8 Table 2 rules, no shared ground-truth anomaly labels between
the "descriptive" and "clinical" result sets). See README.md in this folder.
"""

import numpy as np
import pandas as pd

from config import (
    CONFIG,
    DECISION_RULES_BOTH_PRESENT,
    INSUFFICIENT_DATA_HR_MISSING,
    INSUFFICIENT_DATA_TEMP_MISSING,
    INSUFFICIENT_DATA_BOTH_MISSING,
)


class Patient:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.heart_rate = CONFIG["HEART_RATE_BASELINE"]
        self.body_temp = CONFIG["BODY_TEMP_BASELINE"]
        self.time = 0.0
        self.anomalies = []  # list of (start, end, type, params)
        self._generate_anomalies()

    def _generate_anomalies(self):
        total_seconds = CONFIG["SIMULATION_HOURS"] * 3600
        lam = CONFIG["ANOMALY_LAMBDA_PER_24H"] * (total_seconds / 86400.0)
        num_anomalies = self.rng.poisson(lam=lam)

        anomaly_types = list(CONFIG["ANOMALY_TYPES"].keys())
        placed = []  # (start, end) already used, to avoid overlap
        attempts = 0
        while len(placed) < num_anomalies and attempts < num_anomalies * 20:
            attempts += 1
            a_type = anomaly_types[self.rng.integers(0, len(anomaly_types))]
            spec = CONFIG["ANOMALY_TYPES"][a_type]
            duration = spec["duration_seconds"]
            start = self.rng.uniform(0, max(total_seconds - duration, 1))
            end = start + duration

            if any(not (end < s or start > e) for s, e in placed):
                continue  # overlaps an existing anomaly, retry

            placed.append((start, end))
            lo, hi = spec["target_range"]
            target = self.rng.uniform(lo, hi)
            if a_type in ("bradycardia", "tachycardia"):
                self.anomalies.append((start, end, a_type, {"target_hr": target}))
            else:
                self.anomalies.append((start, end, a_type, {"target_temp": target}))

    def _active_anomaly(self):
        for start, end, a_type, params in self.anomalies:
            if start <= self.time < end:
                progress = (self.time - start) / (end - start)
                return a_type, params, progress
        return None, None, None

    def update(self, dt: float):
        self.time += dt

        # --- Heart rate: true Ornstein-Uhlenbeck process ---
        theta = CONFIG["HR_OU_THETA"]
        sigma = CONFIG["HR_OU_SIGMA"]
        mu = CONFIG["HEART_RATE_BASELINE"]
        dW = self.rng.normal(0, np.sqrt(dt))
        self.heart_rate += theta * (mu - self.heart_rate) * dt + sigma * dW

        # --- Body temperature: circadian cycle + additive noise ---
        amp = CONFIG["BODY_TEMP_CIRCADIAN_AMPLITUDE"]
        circadian = amp * np.sin(2 * np.pi * self.time / 86400.0)
        noise = self.rng.normal(0, CONFIG["BODY_TEMP_NOISE_STD"])
        self.body_temp = CONFIG["BODY_TEMP_BASELINE"] + circadian + noise

        # --- Apply active anomaly, if any ---
        anomaly_type, params, progress = self._active_anomaly()
        if anomaly_type == "bradycardia":
            target = params["target_hr"]
            self.heart_rate = self.heart_rate + (target - self.heart_rate) * min(progress * 2, 1.0)
        elif anomaly_type == "tachycardia":
            # step onset: jump most of the way there immediately, then hold
            target = params["target_hr"]
            self.heart_rate = self.heart_rate * 0.3 + target * 0.7
        elif anomaly_type == "fever":
            target = params["target_temp"]
            self.body_temp = self.body_temp + (target - self.body_temp) * min(progress * 1.5, 1.0)

        self.heart_rate = float(np.clip(self.heart_rate, CONFIG["HEART_RATE_MIN"], CONFIG["HEART_RATE_MAX"]))
        self.body_temp = float(np.clip(self.body_temp, CONFIG["BODY_TEMP_MIN"], CONFIG["BODY_TEMP_MAX"]))

        return self.heart_rate, self.body_temp, (anomaly_type or "none")


class Sensor:
    def __init__(self, name: str, env_params: dict, rng: np.random.Generator):
        spec = CONFIG["SENSOR_SPECS"][name]
        self.sampling_interval = spec["sampling_interval"]
        self.accuracy_offset = spec["accuracy_offset"]
        self.precision_std = spec["precision_std"]
        self.packet_loss_prob = spec["packet_loss_prob"] * env_params["packet_loss_scale"]
        self.noise_scale = env_params["sensor_noise_scale"]
        self.rng = rng
        self._last_sample_time = -self.sampling_interval
        self._last_reading = None

    def sample(self, true_value: float, current_time: float):
        if current_time - self._last_sample_time < self.sampling_interval:
            return self._last_reading
        self._last_sample_time = current_time

        if self.rng.random() < self.packet_loss_prob:
            self._last_reading = None
            return None

        noise = self.rng.normal(0, self.precision_std * self.noise_scale)
        self._last_reading = true_value + self.accuracy_offset + noise
        return self._last_reading


class Processor:
    """Applies the Table 2 8-rule decision table to the latest sensor readings."""

    def __init__(self, env_params: dict, rng: np.random.Generator):
        self.env_params = env_params
        self.rng = rng
        self.hr_buffer = None
        self.temp_buffer = None

    def add_reading(self, sensor_name: str, value):
        if sensor_name == "MAX30100" and value is not None:
            self.hr_buffer = value
        elif sensor_name == "DS18B20" and value is not None:
            self.temp_buffer = value

    def evaluate(self):
        hr, temp = self.hr_buffer, self.temp_buffer
        latency = max(self.rng.normal(self.env_params["network_latency_mean"],
                                       self.env_params["network_latency_std"]), 0.01)

        if hr is not None and temp is not None:
            for hr_lo, hr_hi, t_lo, t_hi, label, rule_id in DECISION_RULES_BOTH_PRESENT:
                hr_ok = (hr_lo is None or hr >= hr_lo) and (hr_hi is None or hr <= hr_hi)
                t_ok = (t_lo is None or temp >= t_lo) and (t_hi is None or temp <= t_hi)
                if hr_ok and t_ok:
                    return rule_id != 0, label, rule_id, latency
            return False, "No Rule Matched", -1, latency

        if hr is None and temp is not None:
            return False, INSUFFICIENT_DATA_HR_MISSING, -1, latency

        if temp is None and hr is not None:
            return False, INSUFFICIENT_DATA_TEMP_MISSING, -1, latency

        return False, INSUFFICIENT_DATA_BOTH_MISSING, -1, 0.0


def run_single_simulation(env_name: str, run_id: int, seed: int) -> pd.DataFrame:
    """Run one full-length (SIMULATION_HOURS, dt=TIME_STEP_SECONDS) simulation."""
    rng = np.random.default_rng(seed)
    env_params = CONFIG["ENVIRONMENT_PARAMS"][env_name]
    dt = CONFIG["TIME_STEP_SECONDS"]
    total_steps = int((CONFIG["SIMULATION_HOURS"] * 3600) / dt)

    patient = Patient(rng)
    sensors = {name: Sensor(name, env_params, rng) for name in CONFIG["SENSOR_SPECS"]}
    processor = Processor(env_params, rng)

    records = []
    detection_start = None
    # per-anomaly-instance detection tracking: (start,end,type) -> first alert time relative to start
    anomaly_windows = [(s, e, t) for s, e, t, _ in patient.anomalies]
    detected = {w: None for w in anomaly_windows}

    for step in range(total_steps):
        t = step * dt
        hr_true, temp_true, anomaly_type = patient.update(dt)

        hr_measured = sensors["MAX30100"].sample(hr_true, t)
        temp_measured = sensors["DS18B20"].sample(temp_true, t)
        env_temp_measured = sensors["DHT11"].sample(
            25.0 + 3.0 * np.sin(2 * np.pi * t / 86400.0), t
        )

        processor.add_reading("MAX30100", hr_measured)
        processor.add_reading("DS18B20", temp_measured)
        processor.add_reading("DHT11", env_temp_measured)

        alert, label, rule_id, latency = processor.evaluate()

        # Ground-truth-anomaly detection timing
        for w in anomaly_windows:
            s, e, _ = w
            if s <= t < e and alert and detected[w] is None:
                detected[w] = t - s

        data_missing = (hr_measured is None) or (temp_measured is None)

        records.append({
            "run_id": run_id,
            "environment": env_name,
            "time": t,
            "hr_true": hr_true,
            "temp_true": temp_true,
            "hr_measured": hr_measured if hr_measured is not None else np.nan,
            "temp_measured": temp_measured if temp_measured is not None else np.nan,
            "alert_triggered": alert,
            "alert_label": label,
            "rule_id": rule_id,
            "processing_latency": latency,
            "ground_truth_anomaly": anomaly_type != "none",
            "anomaly_type": anomaly_type,
            "data_missing": data_missing,
        })

    df = pd.DataFrame(records)
    # attach per-run MTTD info as a separate small frame the batch runner collects
    df.attrs["mttd_records"] = [
        {"run_id": run_id, "environment": env_name, "anomaly_type": w[2], "detect_time": v}
        for w, v in detected.items() if v is not None
    ]
    return df
