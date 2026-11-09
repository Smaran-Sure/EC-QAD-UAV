# Corrected-architecture initial results

These are executed validation results for the independent `physical_reference_v2` neural controller. Prior PID-memory-dependent compression results are preserved as exploratory under `results/archive/v1_pid_dependent_exploratory/` and are excluded here.

The seed-7 memoryless FP32 teacher completed eight held-out 180-second validation episodes (two each nominal, gyro bias, GPS dropout, and airspeed bias) with PID construction/execution disabled. Demonstrations and validation are split by complete trajectory.

| Controller | Mean tracking score | Altitude RMSE (m) | Heading RMSE (deg) | Airspeed RMSE (m/s) | Failed episodes |
|---|---:|---:|---:|---:|---:|
| PID expert | 0.268092 | 4.522916 | 5.252540 | 0.907639 | 0/8 |
| Independent FP32 teacher | 0.268495 | 4.646266 | 5.290896 | 0.904671 | 0/8 |

Values are means of episode-level metrics from `results/conventional_metrics.csv` (validation rows) and `results/teacher_validation_seed7.csv`. The teacher/PID mean score ratio is 1.001504. These flights pass the prespecified teacher gate; they do not establish general robustness beyond this validation regime.

All 17 regression tests pass, including physical feature names/values, PID-disabled FP32 and 8/4/2-bit execution, quantizer parity, rejection of old-schema checkpoints, and invariance of deployment to training-conditioning metadata. Student training and fresh EC-QAD evaluation are in progress; no corrected-architecture EC-QAD advantage is claimed yet.
