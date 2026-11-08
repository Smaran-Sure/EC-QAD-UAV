# EC-QAD Fixed-Wing UAV Conventional GNC Baseline

This repository is the **pre-EC-QAD baseline**. It builds and validates a conventional fixed-wing 6-DoF simulation, simulated sensors, a UKF, cascaded PID flight-control loops, logging, NIS/covariance diagnostics, and fault injection.

Do **not** add neural policy compression until the conventional-system acceptance checks pass.

## Research architecture

```text
flight commands
      |
      v
cascaded PID autopilot <-----------------------+
      |                                        |
      v                                        |
control surfaces + throttle                    |
      |                                        |
      v                                        |
JSBSim nonlinear 6-DoF plant                   |
      | truth                                  |
      v                                        |
simulated IMU / GPS / heading / pitot          |
      | measurements                           |
      v                                        |
UKF --> x_hat, P, innovations, NIS ------------+
```

Later EC-QAD work will use `P` and innovation/NIS as **training-side conditioning signals**. This repository deliberately stops before any neural teacher/student or quantization code.

## Important model note

The config prefers `Rascal110` as the research airframe. Your JSBSim wheel may not include that exact model name/data package. If it is unavailable, the wrapper can load `c172p` **only as a smoke-test fallback**. Do not publish UAV results from the fallback.

The repository includes the Rascal flight dynamics model and matching engine/propeller
under `models/jsbsim`; no environment override is needed. Model lookup tries
`JSBSIM_ROOT` (if set), the repository data, then the installed JSBSim data, using
a fresh simulator for every attempt. Stock `c172p` remains available even when
`JSBSIM_ROOT` points at a custom directory. Initialization metadata records the
selected `model_root`. See `models/jsbsim/README.md` for model provenance.

## Setup

Python 3.10+ recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Run unit tests first:

```bash
pytest -q
```

Check the JSBSim environment/model:

```bash
python experiments/00_check_environment.py
```

## Run order

Run the experiments in this exact order.

### 1. Plant/sign check

```bash
python experiments/01_open_loop.py
```

Small aileron, elevator, and throttle pulses are applied around trim. Check plots under `results/01_open_loop/plots/`.

Pass when:
- simulation remains finite;
- control signs make physical sense;
- no unexpected immediate divergence occurs.

### 2. Trim hold

```bash
python experiments/02_trim.py
```

Requests JSBSim simple trim and then holds the returned command values for 60 s.

Pass when altitude, airspeed, attitude, and rates remain near the operating point. Small drift is acceptable; immediate divergence is not.

### 3. Sensor validation

```bash
python experiments/03_sensor_validation.py
```

Compares GPS, gyro, heading, and airspeed measurements with truth and prints empirical mean/std error.

The sensor values in `configs/sensors.yaml` are **development values**. Replace them with cited sensor-grade/datasheet values before final paper experiments.

### 4. UKF/covariance/NIS validation

```bash
python experiments/04_ukf_validation.py
```

The estimator state is:

```text
[pn, pe, pd,
 vn, ve, vd,
 roll, pitch, yaw,
 bgx, bgy, bgz,
 bax, bay, baz]
```

The UKF logs:
- state estimate;
- covariance diagonal and trace;
- measurement innovation;
- innovation covariance internally;
- NIS for GPS, heading, and airspeed;
- 95% chi-square thresholds.

Inspect `results/04_ukf_validation/plots/`.

### 5. Truth-state PID control

```bash
python experiments/05_truth_control.py
```

The controller temporarily receives perfect truth state. **Tune `configs/controller.yaml` here.**

Do not move on until the truth-state controller stably tracks the standard 180-s profile.

The provided gains are starting values only and are not claimed to be tuned for every JSBSim UAV model.

### 6. Estimated-state PID control

```bash
python experiments/06_estimated_control.py
```

Now the autopilot uses the UKF state instead of JSBSim truth. Truth is retained only for evaluation.

This is the healthy conventional baseline.

### 7. Gyro-bias fault

```bash
python experiments/07_fault_gyro_bias.py
```

Default fault: +0.5 deg/s z-axis gyro bias at 60 s.

Inspect sensor error, estimator state, NIS, covariance, and tracking performance.

### 8. GPS dropout

```bash
python experiments/08_fault_gps_dropout.py
```

Default dropout: 60-80 s.

A key pass condition is that position uncertainty grows during loss of GPS and contracts after GPS returns.

### 9. Airspeed bias

```bash
python experiments/09_fault_airspeed_bias.py
```

Default fault: +3 m/s pitot/airspeed bias after 60 s.

The baseline UKF airspeed observation assumes still air (`|v_NED| ~= true airspeed`). Before paper experiments with wind, add wind states and a proper air-relative measurement model.

### 10. Aileron loss of effectiveness

```bash
python experiments/10_fault_actuator_loe.py
```

Default: actual aileron effectiveness is reduced to 60% after 60 s.

This is intentionally important for EC-QAD: a navigation-state estimator may remain healthy while control effectiveness degrades. If covariance/NIS do not respond strongly, that is **scientifically meaningful** rather than something to hide.

### 11. Nominal Monte Carlo

```bash
python experiments/11_nominal_monte_carlo.py --runs 20
```

For publication, increase the run count after runtime is understood. This script varies sensor seed and small initial-condition offsets.

### 12. Fault Monte Carlo

```bash
python experiments/12_fault_monte_carlo.py --runs 20
```

Randomizes fault onset/severity across gyro bias, GPS dropout, airspeed bias, and aileron loss of effectiveness.

### 13. Conventional-baseline gate

```bash
python experiments/13_acceptance_report.py
```

This checks that required results exist, covariance is finite/nonnegative, nominal NIS is at least broadly consistent, and GPS dropout increases position covariance.

Only after failures are **understood and corrected** should you start the full-precision neural teacher.

## One-command baseline run

After tuning the controller:

```bash
bash scripts/run_baseline.sh
```

This intentionally excludes Monte Carlo because it may take substantially longer.

## Output layout

Each deterministic experiment produces:

```text
results/<experiment>/
  run.csv
  metadata.json
  metrics.json
  plots/
```

The CSV records truth, available sensor measurements, estimator state/covariance, NIS, controller commands, references, and known simulation fault labels.

## What must be improved before a paper

This repository is a research scaffold, not a flight-certified navigation stack. Before final AIAA experiments:

1. Pin the exact UAV model and JSBSim version/commit.
2. Tune PID gains for the chosen UAV and document the tuning procedure.
3. Replace development sensor noise with cited hardware-grade values.
4. Validate the UKF statistically over many nominal trajectories; tune Q/R based on consistency rather than desired results.
5. Add wind states before using airspeed fusion in nonzero-wind experiments.
6. Verify coordinate/sign conventions against the chosen aircraft model.
7. Define held-out fault severity/onset/combinations **before** EC-QAD training.
8. Preserve truth only for evaluation/oracles; never feed truth to the deployed student later.
9. Save code version, configs, random seeds, and scenario manifests for every paper result.

## Why NIS is logged

For innovation `nu` and innovation covariance `S`:

```text
NIS = nu^T S^-1 nu
```

NIS is an estimator-consistency statistic. It is **not automatically a calibrated probability of fault**. EC-QAD should treat it as innovation-derived evidence/consistency information unless you separately calibrate a fault probability model.

## Next stage after this baseline

When the conventional gate passes:

1. train a full-precision neural teacher to imitate/control the validated system;
2. build ordinary 8/4/2-bit QAD students;
3. test whether `P`/NIS predict harmful quantization error in a uniformly trained student;
4. only then finalize the EC-QAD weighting function and baselines.
