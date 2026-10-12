# Virtual Stress Testing: Monte Carlo Simulation of IoT Health Monitor Performance

**First Author:** Dr. Subarmaniam A/L Kannan  
**Second Author:** Lian Jia Jun (243UT246W5)  
**Institution:** Faculty of Information Science and Technology (FIST), Multimedia University (MMU), Melaka Campus

---

## What is this project about?

Wearable and IoT health monitors (for example, a Raspberry Pi device with heart-rate, SpO₂ and temperature sensors) often work well in a prototype demo. Few are tested to find out **how reliable their alerts are in real life**, where sensors are noisy, drift, or drop out.

This project builds a **virtual test bench** to answer that question. Instead of recruiting patients, it uses a **Monte Carlo simulation** to run the same rule-based health-alert logic thousands of times under two conditions:

| Environment | Description |
|---|---|
| **Controlled** | Clinic-like: low sensor noise, few dropouts |
| **Residential** | Home-like: more noise, bias and missing readings |

The logic being tested is an 8-condition rule set (e.g. bradycardia, tachycardia, hypothermia) taken from an earlier IoT healthcare prototype built with a Raspberry Pi, MAX30100 (heart rate / SpO₂) and DS18B20 (temperature) sensors.

## The problem being solved

The earlier prototype only demonstrated that the system **works**. It lacked:

- **Quantitative validation**: no evidence of how reliable the rule logic is, or how often it raises false alarms or misses real events.
- **Stress testing**: it was never tested under noisy, uncontrolled (home) conditions.
- **A verified implementation**: it was unclear whether the logic in the simulation matched the logic on the real device.

Collecting patient data was not feasible (ethics, time, hardware), so the research question became:

> *How reliable and clinically valid is the same 8-condition rule-based logic when stressed under controlled versus home-like environments?*

## Method in brief

1. **Simulate patients**: heart rate and temperature follow an Ornstein–Uhlenbeck (mean-reverting) process, with injected clinical events (such as bradycardia or tachycardia episodes).
2. **Simulate sensors**: add environment-specific noise, fixed bias (e.g. DS18B20 reads about −0.5 °C low) and random dropouts.
3. **Apply the rules**: run the 8-condition decision logic on the *measured* values and compare alerts against the *true* underlying state.
4. **Repeat 1,000 times** (about 86.4 million per-second records) and compute sensitivity, specificity, precision, alert rate, mean time to detection (MTTD) and event-level detection, with confidence intervals and significance tests.
5. **Sanity-check** key sensor parameters against a real-world dataset ([Zenodo, DOI 10.5281/zenodo.20793920](https://doi.org/10.5281/zenodo.20793920)).

## Key finding

With the rule logic implemented faithfully, the simulation shows that the system **raises far too many false alarms** in both environments:

| Metric | Result (both environments) |
|---|---|
| Specificity | ≈ 72 % |
| Precision | ≈ 1.3 % |
| Alert rate | ≈ 27 % of all time points |

**Root cause:** the hypothyroidism rule fires when temperature is below 36.0 °C. Combined with the sensor's −0.5 °C bias, about 27 % of perfectly healthy readings fall below that threshold, and this single rule accounts for roughly 98 % of all alerts.

**Takeaway:** fixed-threshold rules must account for realistic sensor bias. Simulation-based "stress testing" can expose this kind of design flaw before any clinical deployment.

## Repository structure

```
├── 01_Bibliography/                 BibTeX reference library
├── 04_Figures/                      Early-stage figures (superseded)
└── 05_Simulation_Code_and_Data/
    ├── full_simulation_v2/          Current simulation (use this one)
    │   ├── config.py                Parameters and the 8 decision rules
    │   ├── sim_engine.py            Patient + sensor simulation engine
    │   ├── run_batch.py, run_all.py Run the simulation (N = 1,000)
    │   ├── analyze_results.py       Metrics and statistics
    │   ├── recompute_*.py           Re-evaluate stored output with updated rules
    │   ├── build_figures.py         Generate paper figures
    │   ├── paper_figures_v2*/       Generated figures and summary tables
    │   └── simulation_summary_v2.json  Aggregated results
    ├── real_world_dataset/          Plausibility-check script and results
    └── (earlier prototype scripts)  Legacy code, kept for reference only
```

> The ~6 GB of raw per-second simulation output is not stored in this repository. It can be regenerated with `run_all.py`.

## How to run

Requirements: Python 3.9+ with `numpy`, `pandas`, `scipy`, `matplotlib`, `pyarrow`.

```bash
cd 05_Simulation_Code_and_Data/full_simulation_v2
python run_all.py          # full 1,000-run simulation (long; writes raw_data_v2/)
python analyze_results.py  # compute metrics and statistics
python build_figures.py    # generate figures
```

Simulation settings (number of runs, environments, noise levels, rules) are all in `config.py`.

## Project highlights

| # | Highlight | Detail |
|---|---|---|
| 1 | **Reproducible simulator** | Rebuilt the simulation pipeline from scratch; completed N = 1,000 runs (about 86.4 million records) |
| 2 | **Dual detection metrics** | Reports per-second sensitivity alongside event-level detection, so detection ability is not understated |
| 3 | **Rule-table audit** | Compared the simulator's rule table line by line against the original prototype source and found that 3 of the 8 rules had been silently altered from two-sensor to single-sensor conditions |
| 4 | **Re-evaluation without re-simulation** | Restored the original rules and re-evaluated the stored output by post-processing, which reversed the headline results (see below) |
| 5 | **Verified references** | Audited the bibliography and checked cited claims against their sources; final library of 86 entries |

**Effect of the rule-table correction**

| Metric | Altered rules | Faithful rules |
|---|---|---|
| Specificity | 99.83 – 99.96 % | **72.28 – 72.90 %** |
| Precision | 61.57 – 87.84 % | **1.27 – 1.32 %** |
| Alert rate | 0.31 – 0.43 % | **27.18 – 27.81 %** |

Results that contradicted the earlier narrative were reported as found; no sensor-bias values were invented to make them look better.

## Skills and methods demonstrated

- Discrete-event and Monte Carlo simulation (Python, streaming analysis of 86.4M rows)
- Statistical analysis: confidence intervals, significance tests, event-level vs. sample-level metrics
- Root-cause analysis: reconciling code against source documents
- Systematic literature verification and citation management (BibTeX)

## Limitations

- Purely simulated; no clinical or patient validation.
- Sensor noise and bias values are modelling assumptions, partly checked against a small real-world dataset.
- No quantitative comparison against standard early-warning scores such as NEWS2 or MEWS.
- The home environment is modelled at a single severity level.

## Status

Research project carried out as part of an internship / degree programme at MMU. A manuscript based on this work is in preparation.
