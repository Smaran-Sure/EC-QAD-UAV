# EC-QAD UAV experiments

A reproducible six-DoF fixed-wing simulation study of training-only estimator conditioning for quantized neural flight control. Deployed teacher and student use estimated physical navigation information and reference commands only: no PID memory, covariance, or NIS inputs.

## Executed result

The completed study contains 3,280 rollouts, three student optimization seeds, one frozen FP32 teacher, and 20 matched scenarios per condition. At four bits, covariance-only conditioning reduced off-nominal tracking score by 29.74% relative to uniform QAD (95% paired bootstrap interval 21.65–35.48%). It did **not** consistently beat distribution-matched shuffled covariance weights. Incremental covariance/NIS prediction was unsupported under the specified H1 test. Combined EC-QAD-v1 performed worse overall. These are preliminary simulation results, not evidence of estimator-specific robustness or real-flight safety.

See [initial results](docs/INITIAL_RESULTS.md), [research verdict](docs/RESEARCH_VERDICT.md), [weighting diagnosis](docs/ECQAD_V1_DIAGNOSIS.md), and [AIAA summary](docs/AIAA_AVIATION_SUMMARY.md).

## Setup and reproduction

Run from this repository. The executed dependency versions are pinned in `requirements-lock.txt`; The executed interpreter was Python 3.11.18 (see `results/environment.json`); Python 3.10 or newer is required by the package.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
bash scripts/run_initial_results.sh
bash scripts/run_extended_results.sh
```

The initial script collects expert trajectories, validates the independent teacher, trains 8/4/2-bit students, and evaluates the first seed. The extended script adds all three student seeds and four-bit ablations. Evaluation resumes completed episode identities; training preserves existing checkpoints. Full reproduction is computationally substantial. Use a clean checkout of the same repository revision for regeneration while retaining these archived artifacts.

To regenerate figures and reports from the completed local trajectory logs and episode tables:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/run_study.py report
```

Stage commands are `collect`, `teacher`, `teacher_gate`, `students`, `deployment_gate`, `evaluate`, and `report`; run `experiments/run_study.py --help` for arguments. Configuration and seeds are in `configs/study.yaml`. Ignored large trajectory/checkpoint artifacts remain in the local `results/data`, `results/checkpoints`, and evaluation log directories; a source-only clone regenerates them using the scripts.

## Methods and artifacts

- [Method](docs/METHOD.md): aircraft provenance, controller, sensors, UKF, policy, quantizer, and exact weighting equations.
- [Experiment design](docs/EXPERIMENT_DESIGN.md): scenario splits, seeds, controls, metrics, and statistical analysis.
- `results/per_episode_metrics.csv`: canonical 3,280 executed rollouts; tagged CSVs are source subsets, not additional independent trials.
- `results/definitive_preliminary_table.csv`, `results/covariance_comparisons.csv`, and `results/signal_analysis.csv`: final comparisons and H1 estimates.
- `results/figures/`: PNG and PDF scientific figures.
- `results/study_audit.json`: complete grid, matched scenarios, deployment isolation, and frozen-artifact preservation checks.
- `results/diagnosis_manifest.json`: final analysis and result hashes; the original freeze remains in `results/frozen_ecqad_v1/`.

The frozen combined method is **EC-QAD-v1** (raw ID `ecqad`). `shuffled` matches combined weights; `covariance_shuffled` preserves the exact covariance-only weight distribution. No revised fusion rule was selected on evaluation data. Older PID-dependent results remain exploratory under `results/archive/` and are excluded. The accepted teacher is seed 7; three student seeds do not constitute independent teacher replications. Model-size figures are theoretical packed storage, not hardware measurements.
