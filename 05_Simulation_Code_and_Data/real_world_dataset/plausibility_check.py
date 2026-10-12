"""
plausibility_check.py - Compares real-world sensor readings (medisync_collected_readings.csv,
the user's own ESP32+MQTT+InfluxDB dataset, Zenodo DOI 10.5281/zenodo.20793920) against the
heart-rate/temperature parameters used in full_simulation_v2/config.py.

Scope: plausibility/external-validity support for the simulation's parameter choices ONLY.
The real dataset's status/prediction/alert columns are the field device's own threshold logic,
not clinician ground truth, so this script does NOT compute sensitivity/specificity/precision
against them - see 08_Weekly_Reports/Week2_2026-08-10.md for why.

Run: python plausibility_check.py
"""
import csv
import json
import math
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
CSV_PATH = HERE / "medisync_collected_readings.csv"
CONFIG_PATH = HERE.parent / "full_simulation_v2" / "config.py"
SUMMARY_PATH = HERE.parent / "full_simulation_v2" / "simulation_summary_v2.json"

# Simulated targets, derived analytically from config.py (not re-run here):
#   HR: Ornstein-Uhlenbeck stationary std = sigma / sqrt(2*theta)
HR_OU_THETA = 0.005
HR_OU_SIGMA = 0.30
HR_BASELINE = 72.0
SIM_HR_STATIONARY_STD = HR_OU_SIGMA / math.sqrt(2 * HR_OU_THETA)

#   Temp: measured baseline = true baseline + DS18B20 accuracy_offset;
#   std approximated from circadian sine amplitude (std of a sine = amplitude/sqrt(2))
#   plus small additive per-step noise (BODY_TEMP_NOISE_STD), combined in quadrature.
BODY_TEMP_BASELINE = 36.8
DS18B20_OFFSET = -0.5
CIRCADIAN_AMPLITUDE = 0.5
BODY_TEMP_NOISE_STD = 0.03
SIM_TEMP_MEASURED_MEAN = BODY_TEMP_BASELINE + DS18B20_OFFSET
SIM_TEMP_STD = math.sqrt((CIRCADIAN_AMPLITUDE / math.sqrt(2)) ** 2 + BODY_TEMP_NOISE_STD ** 2)


def load_real_data():
    bpm, temp_all, temp_plausible, spo2 = [], [], [], []
    by_subject = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["anon_subject_id"]
            by_subject.setdefault(sid, {"bpm": [], "temp": [], "spo2": []})
            try:
                b = float(row["bpm"])
                if b > 0:  # 0 = sensor dropout, not a physiological reading
                    bpm.append(b)
                    by_subject[sid]["bpm"].append(b)
            except (ValueError, KeyError):
                pass
            try:
                t = float(row["temperature"])
                temp_all.append(t)
                by_subject[sid]["temp"].append(t)
                if 34.0 <= t <= 42.0:  # exclude off-body/ambient artifact readings
                    temp_plausible.append(t)
            except (ValueError, KeyError):
                pass
            try:
                spo2.append(float(row["spo2"]))
            except (ValueError, KeyError):
                pass
    return bpm, temp_all, temp_plausible, spo2, by_subject


def main():
    bpm, temp_all, temp_plausible, spo2, by_subject = load_real_data()

    result = {
        "source": "medisync_collected_readings.csv (Zenodo DOI 10.5281/zenodo.20793920)",
        "n_rows_total": sum(len(v["temp"]) for v in by_subject.values()),
        "heart_rate": {
            "simulated_baseline_bpm": HR_BASELINE,
            "simulated_stationary_std_bpm": round(SIM_HR_STATIONARY_STD, 2),
            "real_n_nonzero": len(bpm),
            "real_mean_bpm": round(st.mean(bpm), 2),
            "real_std_bpm": round(st.pstdev(bpm), 2),
            "real_min_bpm": min(bpm),
            "real_max_bpm": max(bpm),
            "per_subject_mean_bpm": {
                sid: round(st.mean(v["bpm"]), 1)
                for sid, v in sorted(by_subject.items())
                if v["bpm"]
            },
        },
        "temperature": {
            "simulated_measured_mean_c": round(SIM_TEMP_MEASURED_MEAN, 2),
            "simulated_std_c_approx": round(SIM_TEMP_STD, 2),
            "real_n_all": len(temp_all),
            "real_n_plausible_34_42c": len(temp_plausible),
            "real_plausible_fraction": round(len(temp_plausible) / len(temp_all), 3),
            "real_mean_c_plausible_only": round(st.mean(temp_plausible), 2),
            "real_std_c_plausible_only": round(st.pstdev(temp_plausible), 2),
        },
        "spo2_not_modeled_in_simulation": {
            "note": "The manuscript's 8 decision rules (Table 2) use only heart rate and "
                    "temperature; SpO2 is not part of the rule system despite the MAX30100 "
                    "sensor collecting it, so no simulated SpO2 exists to compare against.",
            "real_n": len(spo2),
            "real_mean_pct": round(st.mean(spo2), 2),
            "real_std_pct": round(st.pstdev(spo2), 2),
            "real_min_pct": min(spo2),
            "real_max_pct": max(spo2),
        },
    }

    out_path = HERE / "plausibility_check_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
