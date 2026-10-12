# full_simulation_v2 — rebuilt simulation engine (2026-08-05)

## Why this exists

The original scripts in `05_Simulation_Code_and_Data/` (`conference_sim.py`,
`run_batch.py`, `run_all_batches.py`, `plots.py`, `stats.py`,
`create_figures_publication.py`, `setup.py`) all have their indentation
stripped out and fail to parse as Python (`ast.parse` raises
`SyntaxError` on every one of them — confirmed the same corruption already
exists in the original `Research Output/IoT Healthcare_Simv2/` source, so it
isn't something introduced later). Only `analyze_1000runs.py`,
`fix_imports.py`, `full_spec_config.py`, and `run_batch_integrated.py`
parsed, and even those implemented a simplified model that didn't match the
manuscript's Methodology section: heart rate was a plain random walk (no
mean-reversion), the alert rule was a single crude threshold, and there was
no shared ground-truth anomaly label connecting the "descriptive" (Table 5)
and "clinical" (Table 6) result sets — they came from two differently-scaled,
differently-configured simulations (500 runs / 24h vs. 100 runs / 6h) that
were presented in the paper as if part of one coherent experiment.

This folder is a clean rebuild that:
- **Parses and runs** (verified with a reduced-scale smoke test, see below).
- Implements a **true Ornstein-Uhlenbeck process** for heart rate (mean-reverting
  toward 72 BPM), matching the Methodology section's description.
- Implements **all 8 rules from Table 2** (the old code only had 6; it was
  missing "Electrical System Issue", "Hypothyroidism", "Shock / Sepsis").
- Runs **one single N=1000 (500+500), 24-hour, dt=1s** experiment that produces
  both the descriptive stats (Table 5) and, from the same ground-truth
  anomaly labels, the clinical confusion matrix (Table 6), MTTD (Table 7),
  and rule distribution (Table 8) — no more two-simulations-glued-together.

## Assumptions made (please review before treating a run as final)

The manuscript describes several mechanisms qualitatively without giving
exact numbers. These are marked `ASSUMPTION` in `config.py`:

1. **OU process parameters** (reversion rate `theta`, volatility `sigma`) —
   not specified anywhere in the manuscript. Chose values giving a resting
   heart-rate std-dev of ~3 BPM and a multi-minute relaxation timescale.
2. **Tachycardia anomaly duration** — Table 1 only says "Rapid Onset (Step
   Increase)" with no sustain duration. Used 5 minutes (same as bradycardia).
3. **Temperature noise magnitude** — "tiny, unpredictable jitters" isn't
   quantified. Used 0.03°C std, small relative to the 0.5°C circadian swing.
4. **"Undefined" in Table 2's rule table** (Electrical System Issue,
   Hypothyroidism, Shock/Sepsis rows) — interpreted as "that vital sign's
   reading was missing at evaluation time" rather than "any value accepted".
   This is the only reading that makes those three rules distinct from the
   two-sensor rules above them; the alternative (Undefined = wildcard) would
   make them overlap/duplicate other rules. Worth confirming with the
   project owner since Table 2 itself doesn't say which reading was intended.

**None of these were pinned down by the original methodology text — they are
modeling choices made here to produce a runnable, internally-consistent
simulation. If the project owner has different numbers in mind, only
`config.py` needs to change.**

## ⚠️ The numbers will change

A reduced-scale smoke test (6h, 12 runs total, run in this session to verify
the code works) already shows **substantially different** clinical metrics
than the current manuscript: precision came out around 0.6–0.8 instead of
0.06–0.08, because the OU-process heart rate no longer wanders as
erratically as the old plain-random-walk model did, so far fewer readings
spuriously cross a rule threshold. This is expected — the whole point of
option B was to make the code match the described methodology — but it means
**the Results section numbers (Table 5–8, Abstract, Discussion) will need to
be rewritten once the real 1000-run output comes in.** Don't assume the
current "93% false positives" narrative survives; it may substantially
improve, which is itself a real and reportable finding, but changes the
paper's conclusion. Flag this to the project owner before finalizing anything
downstream (Abstract, Discussion, Conclusion all currently quote the old
numbers).

## How to run the real thing

```
cd full_simulation_v2

# Each environment needs ceil(500/25) = 20 batches (batch size is in config.py)
for i in $(seq 0 19); do python run_batch.py Controlled $i; done
for i in $(seq 0 19); do python run_batch.py Residential $i; done

python analyze_results.py
```

Runtime estimate from the smoke test: ~1.5s per 24-hour run on this machine,
so 1000 runs ≈ 25–30 minutes total (single process, no parallelism). Batches
are independently seeded and resumable — if a batch fails partway, just
re-run that one batch, it reproduces the same data deterministically.

Requires `numpy` and `pandas` (already available in this environment).
`pyarrow` is optional — batches save as `.parquet` if it's installed, else
fall back to `.csv` automatically. `scipy` is optional for the significance
test's exact p-value; without it, a normal-approximation p-value is used
(labeled as such in the output) — install it if you want the same
Welch's-t-test method the original `stats.py` used.

Output: `raw_data_v2/*.parquet` (or `.csv`) per batch, plus
`simulation_summary_v2.json` after `analyze_results.py` runs.
