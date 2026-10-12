"""
config.py - Simulation parameters for the N=1000, 24-hour, dt=1s Monte Carlo run.

Values are taken directly from the manuscript (Methodology section, Table 1,
Table 2, Table 3) wherever the manuscript specifies a number. Where the
manuscript describes a mechanism but does not pin down a numeric parameter
(OU reversion rate/volatility, tachycardia anomaly duration, additive
temperature-noise magnitude), a value is chosen here and marked ASSUMPTION so
it can be reviewed/adjusted before the run counts as final.
"""

CONFIG = {
    # ---- Monte Carlo scale (Methodology: "N = 1,000 independent Monte Carlo
    # runs", "simulated 24-hour periods", "fixed time-step clock (dt = 1.0 s)") ----
    "TOTAL_RUNS": 1000,
    "ENVIRONMENTS": ["Controlled", "Residential"],
    "SIMULATION_HOURS": 24.0,
    "TIME_STEP_SECONDS": 1.0,
    "RUNS_PER_BATCH": 25,  # batch size for resumable batch runs

    "OUTPUT_DIR": "raw_data_v2",

    # ---- Physiological baseline (Methodology paragraph 3) ----
    "HEART_RATE_BASELINE": 72.0,
    "HEART_RATE_MIN": 40.0,
    "HEART_RATE_MAX": 180.0,
    # ASSUMPTION: manuscript describes the HR process qualitatively as an
    # Ornstein-Uhlenbeck process pulled back toward 72 BPM but does not give
    # numeric reversion speed (theta) or volatility (sigma). Chosen so the
    # process has a stationary std-dev of ~3 BPM and a relaxation timescale
    # of a few minutes, consistent with normal resting heart-rate variability.
    "HR_OU_THETA": 0.005,   # per-second mean-reversion rate
    "HR_OU_SIGMA": 0.30,    # per-second volatility

    "BODY_TEMP_BASELINE": 36.8,
    "BODY_TEMP_MIN": 35.0,
    "BODY_TEMP_MAX": 41.0,
    "BODY_TEMP_CIRCADIAN_AMPLITUDE": 0.5,  # manuscript: oscillates 36.3-37.3 degC
    # ASSUMPTION: "additive random noise... tiny, unpredictable jitters" -
    # magnitude not specified. Chosen small relative to the 0.5 degC circadian
    # swing.
    "BODY_TEMP_NOISE_STD": 0.03,

    # ---- Anomaly injection (Methodology: "Poisson distribution technique...
    # lambda ~ 1 event per 24 hours"; Table 1 for target ranges/durations) ----
    "ANOMALY_LAMBDA_PER_24H": 1.0,
    "ANOMALY_TYPES": {
        "bradycardia": {
            "target_range": (45.0, 55.0),
            "duration_seconds": 5 * 60,       # Table 1: "5 minutes (Linear Decrease)"
            "pattern": "linear",
        },
        "tachycardia": {
            "target_range": (110.0, 130.0),
            # ASSUMPTION: Table 1 gives only "Rapid Onset (Step Increase)" for
            # tachycardia with no explicit sustain duration. Using the same
            # 5-minute sustain window as bradycardia (both are acute HR events).
            "duration_seconds": 5 * 60,
            "pattern": "step",
        },
        "fever": {
            "target_range": (38.5, 40.0),
            "duration_seconds": 30 * 60,      # Table 1: "30 minutes (Gradual Increase)"
            "pattern": "gradual",
        },
    },

    # ---- Sensor specs (Methodology + conference_metadata.json, unchanged) ----
    "SENSOR_SPECS": {
        "MAX30100": {"sampling_interval": 2.0, "accuracy_offset": 0.0, "precision_std": 1.5, "packet_loss_prob": 0.02},
        "DS18B20":  {"sampling_interval": 4.0, "accuracy_offset": -0.5, "precision_std": 0.2, "packet_loss_prob": 0.01},
        "DHT11":    {"sampling_interval": 4.0, "accuracy_offset": 0.0, "precision_std": 2.0, "packet_loss_prob": 0.03},
    },

    # ---- Environments (Table 3) ----
    "ENVIRONMENT_PARAMS": {
        "Controlled": {
            "sensor_noise_scale": 1.0,
            "packet_loss_scale": 1.0,
            "network_latency_mean": 0.05,
            "network_latency_std": 0.02,
        },
        "Residential": {
            "sensor_noise_scale": 1.5,   # Table 3: "+50% for motion artifacts"
            "packet_loss_scale": 2.0,    # Table 3: packet loss 0.01 -> 0.02
            "network_latency_mean": 0.15,
            "network_latency_std": 0.10,
        },
    },

    "PROCESSING_LATENCY_MEAN": 0.050,  # Methodology: "~50 ms (+-10 ms)"
    "PROCESSING_LATENCY_STD": 0.010,

    "ALERT_TRANSMISSION_RELIABILITY": 0.99,
    "ALERT_TRANSMISSION_DELAY_MEAN": 2.0,
}

# ---- Decision rules (Table 2) ----
# CORRECTED 2026-08-26 (Week5 re-review, R1/R2/R6): the previous version of
# this table diverged from the original FYP prototype's actual decision logic
# (original prototype source code, Appendix C). The original
# code's 8 conditions are ALL two-sensor (HR AND Temp) conditions, evaluated
# in if/elif order (first match wins). The prior version of this file had
# reassigned 3 of those 8 conditions -- "Potential Issue with the Heart's
# Electrical System", "Hypothyroidism", "Shock / Sepsis" -- to single-sensor
# "missing reading" rules instead, using the same numeric bound on whichever
# one sensor was present. That is not what the original prototype does: it
# silently dropped those 3 two-sensor conditions from ever being evaluated
# when both sensors ARE present (their HR+Temp combinations fell through to
# "No Rule Matched" instead), while attaching their labels to an unrelated
# missing-data scenario the original prototype's code does not distinguish
# at all. This was discovered via direct comparison against Appendix C,
# not the simulation output, and is a data-integrity fix, not a redesign:
# it restores the SAME 8 conditions and thresholds the original prototype
# specifies, in original order. See 08_Weekly_Reports/Week5_2026-08-26.md.
#
# Each rule: (hr_low, hr_high, temp_low, temp_high, label, rule_id).
# None means "not constrained by this reading". Rules are evaluated in
# order; first match wins (matches the original prototype's if/elif chain).
DECISION_RULES_BOTH_PRESENT = [
    (None, 55, None, 36.0, "Hypothermia / Bradycardia", 1),
    (100, None, 38.0, None, "High Fever / Dehydration", 2),
    (70, 90, 37.5, 38.5, "Fever / Overheating", 3),
    (90, None, 36.5, 37.5, "Anxiety / Stress / Physical Exertion", 4),
    (None, 60, 36.5, 37.5, "Potential Issue with Heart's Electrical System", 5),
    (60, 80, None, 36.0, "Hypothyroidism / Exposure to Cold Temperature", 6),
    (100, None, None, 36.0, "Shock / Sepsis / Circulatory Problems", 7),
    (60, 100, 36.0, 37.5, "Normal", 0),
]

# When exactly one of HR/Temp is missing (lost packet / stale buffer), the
# original prototype's code has no defined behaviour for this case (it was
# not designed to handle a missing reading at all). Rather than reuse one of
# the 8 rule labels above for an unrelated scenario, missing-sensor seconds
# are labelled distinctly and never trigger an alert on their own.
INSUFFICIENT_DATA_HR_MISSING = "Insufficient Data (HR missing)"
INSUFFICIENT_DATA_TEMP_MISSING = "Insufficient Data (Temp missing)"
INSUFFICIENT_DATA_BOTH_MISSING = "Insufficient Data (HR and Temp missing)"
