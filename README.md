# EC-QAD for fixed-wing UAV control

Research pipeline for **Estimator-Conditioned Quantization-Aware Distillation**: JSBSim Rascal110 → simulated sensors → navigation UKF → PID-imitation FP32 teacher → fake-quantized students → uniform/estimator-weighted distillation → paired closed-loop evaluation.

The question is whether physical estimator covariance and innovation statistics identify compression-critical conditions, and whether emphasizing those samples during training improves low-bit flight. **Covariance and NIS are training-only signals, never extra deployed student inputs.** Positive results are not presumed.

## Setup

Python 3.11 is the recorded environment. From this repository:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Exact executed package versions and model hashes are in `results/environment.json`. Aircraft XML is vendored locally; no aircraft download or C172 fallback is required for the study. The small Rascal model is publicly sourced but has approximate BETA aerodynamics. See `models/jsbsim/README.md` and `docs/METHOD.md`.

## Reproduce the initial experiment

```bash
bash scripts/run_initial_results.sh
```

The stages can also be run separately:

```bash
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ecqad-mpl JSBSIM_DEBUG=0
python experiments/run_study.py collect --workers 4
python experiments/run_study.py teacher --seed 7
python experiments/run_study.py teacher_gate --seed 7 --workers 4
python experiments/run_study.py students --seed 7
python experiments/run_study.py evaluate --seed 7 --episodes 20 --workers 4
python experiments/run_study.py report
```

Ablations and PTQ:

```bash
python experiments/run_study.py students --seed 7 --ablations
python experiments/run_study.py evaluate --seed 7 --episodes 20 --workers 4 --ablations
python experiments/run_study.py report
```

Run `bash scripts/run_extended_results.sh` to preserve the initial result, add all four-bit weighting controls and PTQ, and repeat the principal and weighting comparisons with training seeds 19 and 43. The configuration is `configs/study.yaml`. Whole trajectories are split into training, validation, and evaluation; fault severity and onset intervals differ across these splits. Runs resume from saved datasets, checkpoints, and per-episode metrics. Use a clean results directory when changing configuration; existing artifacts are not automatically invalidated. Do not run two evaluations that write the same metrics CSV concurrently. `--tag pilot_` provides a separate output table for an explicitly smaller pilot.

## Outputs and interpretation

- `results/per_episode_metrics.csv`: individual paired rollout results, seeds, fault parameters, failures.
- `results/metrics.csv`, `results/summary.csv`: method/bit-width/condition summaries.
- `results/training_runs.csv`: optimization and storage records.
- `results/ecqad_ablation.csv`, `results/paired_comparisons.csv`, `results/signal_analysis.csv`: ablations, paired comparisons and H1 analyses.
- `results/figures/`: PNG and PDF plots.
- `results/data/`, `results/traces/`: local compressed trajectories; representative runs include full covariance and innovation matrices.
- `results/checkpoints/`: locally saved teacher/student weights.

Large reproducible data and checkpoint artifacts are ignored by Git and retained in the working repository. Summary CSVs, configuration, provenance, figures, and documentation should accompany results. Approximate packed storage is not measured hardware latency, energy, or executable binary size. The deployment path still uses floating-point simulation of quantization.

Read `docs/METHOD.md`, `docs/EXPERIMENT_DESIGN.md`, `docs/INITIAL_RESULTS.md`, and `docs/AIAA_AVIATION_SUMMARY.md` for scientific details and executed findings. `docs/WORK_LOG.md` records the flight-baseline repair. H1 (predictive information) and H2 (benefit from weighting) are assessed separately; neither correlation nor a single-seed improvement establishes a general robustness claim.

The original numbered baseline experiments remain available for focused controller/sensor checks; the study command above is the main research path.
