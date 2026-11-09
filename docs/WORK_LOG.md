# Execution record

## 2026-11-08: repository inspection and flight baseline

Inspected source, configuration, experiments, tests, model XML, prior results and Git history. Starting application changes were uncommitted; the environment was also incorrectly tracked in Git. Six existing tests passed. No teacher/student implementation existed.

One Rascal stabilization investigation, no new trim solver: disabled the runner's forced trim, applied configured feedforward controls and corrected rudder damping sign. Identified zero-RPM electric-motor torque startup artifact (initial roll rate exceeded 180 rad/s) matching https://github.com/JSBSim-Team/jsbsim/issues/1269. Disabled `set-running` (which invokes a large-step propulsion steady-state calculation) for the electric motor and bounded initial throttle by `1e-5 + t/2` for startup. The actuator ramp is common to every controller. The aerodynamic/engine XML is unchanged. The initial operating point is stabilized by feedback, not claimed exactly trimmed.

Retained ArduPilot Rascal110 at the vendored commit recorded in models/jsbsim/README.md. This is a public small fixed-wing model with approximate BETA aerodynamics, not a flight-validated identified airframe. Source provenance alone does not validate its quantitative fidelity.

Executed 180-s standard reference with truth PID and estimated PID. Machine-readable records: results/truth_pid_baseline.json and results/estimated_pid_baseline.json. Estimated PID RMSE: altitude 4.251670 m; heading 4.939169 deg; airspeed 0.742579 m/s. Roll/pitch/yaw estimation RMSE 0.132602/0.143599/0.170796 deg. GPS/heading/airspeed NIS fraction below 95% thresholds 0.947836/0.956679/0.958067. These include startup and reference steps. Full trajectory CSVs are retained locally.

Vectorized UKF sigma propagation and covariance operations without changing mathematics. Compared vectorized propagation to scalar reference on randomized states: maximum difference 2.78e-17. Runtime for the executed 180-s estimator baseline was 12.66 s on this machine; this is simulation wall time, not deployed policy latency.

Scientific interpretation follows H1 (information beyond fault labels/generic weighting) separately from H2 (benefit from weighting). No positive EC-QAD conclusion has yet been established.

## Coordinate-consistency correction before scientific interpretation

The first complete one-seed comparison executed 640 rollouts (20 per condition per controller), plus a 32-rollout pilot. It did not favor EC-QAD: overall normalized scores were 0.282428 vs 0.276285 at 4 bits and 7.723699 vs 3.421615 at 2 bits (EC-QAD vs uniform); two-bit failure fractions were 0.9625 vs 0.3500. These are **superseded engineering results**, not the corrected scientific table.

Saved innovation matrices exposed a nominal north-position innovation mean around 1.67 m. The plant wrapper multiplied geodetic latitude changes by the WGS84 equatorial radius instead of the local meridional radius. Corrected the N/E tangent metric using WGS84 meridional/prime-vertical radii plus initial altitude. A new regression test checks position finite differences against navigation velocity; all 13 tests passed. Corrected nominal GPS innovation mean after 20 s was [0.024118, -0.022489, -0.106348, 0.004127, 0.001214, -0.000480] in position/velocity units.

Archived all original learning data, checkpoints, metrics and traces in `results/archive/v0_spherical_coordinate_bug/`. Regenerated the complete training/validation population and retrained from scratch with **unchanged learning settings**. This correction is a simulator-interface repair, not tuning to favor EC-QAD. The original evaluation was confirmed terminal before archival; a previously attempted stop was not executed because automatic approval review hit an account usage limit.

The virtual environment was removed from Git's index without deleting installed files; this cleanup is staged. Further milestone commits await approval-review availability. Old duplicate trim scripts are archived under `experiments/debug/` and excluded from the main execution path.

## Corrected learning experiment in execution

The corrected 40-trajectory data population completed without failure. Seed-7 teacher validation score/PID score = 0.999508 with zero failures; seed-19 ratio = 0.998633 with zero failures. All teacher/student/PTQ checkpoints now exist for seeds 7, 19, and 43. Every seed includes combined/covariance/innovation/shuffled, a bounded inverse-frequency control, a teacher-Jacobian sensitivity control, and an exact inverse-frequency fault-balanced control. The latter has equal total weight per active-fault/nominal category and mean weight one, without EC-QAD's clipping bounds. Adding it did not change the main EC-QAD/uniform settings.

Canonical seed-7 initial evaluation is run by `scripts/run_initial_results.sh` (log `/tmp/ecqad_initial_v1.log`). Seed-19 and seed-43 extended comparisons run with separate `seed19_` and `seed43_` output tags to avoid concurrent CSV writes. They must be merged by episode identity after their processes finish. The first canonical 640-row corrected result must be snapshotted before adding seed-7 ablation/PTQ evaluations or merging other seeds. Signal analysis filters to the canonical metrics scope, and example plots use its lowest numbered training seed, so partial parallel evaluations cannot contaminate the first result.

Training CSV reconciliation reads authoritative checkpoint metadata in the report step, preventing a lost concurrent CSV append from omitting a trained model. Episode durations for gyro/airspeed faults are metadata-corrected from their configured 145-s recovery time; outcomes are unchanged. Current analysis controls include fault category/status, time, attitude magnitude, teacher sensitivity, current evaluation-only tracking severity, and training-seed indicators. This is a restricted predictive test, not an information-theoretic or causal claim.

### Architecture correction: independent neural controller

The user identified that the first policy interface included five PID integrals and three derivative-memory features. That interface was incompatible with the intended research architecture, despite excluding covariance/NIS. All current outputs and generated result summaries were moved intact to `results/archive/v1_pid_dependent_exploratory/`. Those results are exploratory only. The replacement has sixteen explicitly named physical navigation/reference features, no PID object during neural flight, and schema-checked checkpoints. Regression tests prohibit PID construction for FP32/8/4/2-bit deployment and assert the exact feature whitelist/values. The initial memoryless teacher must pass fresh closed-loop validation before the EC-QAD study restarts.

The corrected memoryless teacher (seed 7) passed all eight held-out 180-s flights with zero failures and mean tracking-score ratio 1.001504191039118 relative to the PID expert. All metadata records confirm `pid_executed=False` and `physical_reference_v2`. The 16-test suite passes. No PID memory, history, or recurrence was needed for this validation regime.
