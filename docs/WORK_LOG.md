# Execution record

## 2026-11-08: repository inspection and flight baseline

Inspected source, configuration, experiments, tests, model XML, prior results and Git history. Starting application changes were uncommitted; the environment was also incorrectly tracked in Git. Six existing tests passed. No teacher/student implementation existed.

One Rascal stabilization investigation, no new trim solver: disabled the runner's forced trim, applied configured feedforward controls and corrected rudder damping sign. Identified zero-RPM electric-motor torque startup artifact (initial roll rate exceeded 180 rad/s) matching https://github.com/JSBSim-Team/jsbsim/issues/1269. Disabled `set-running` (which invokes a large-step propulsion steady-state calculation) for the electric motor and bounded initial throttle by `1e-5 + t/2` for startup. The actuator ramp is common to every controller. The aerodynamic/engine XML is unchanged. The initial operating point is stabilized by feedback, not claimed exactly trimmed.

Retained ArduPilot Rascal110 at the vendored commit recorded in models/jsbsim/README.md. This is a public small fixed-wing model with approximate BETA aerodynamics, not a flight-validated identified airframe. Source provenance alone does not validate its quantitative fidelity.

Executed 180-s standard reference with truth PID and estimated PID. Machine-readable records: results/truth_pid_baseline.json and results/estimated_pid_baseline.json. Estimated PID RMSE: altitude 4.251670 m; heading 4.939169 deg; airspeed 0.742579 m/s. Roll/pitch/yaw estimation RMSE 0.132602/0.143599/0.170796 deg. GPS/heading/airspeed NIS fraction below 95% thresholds 0.947836/0.956679/0.958067. These include startup and reference steps. Full trajectory CSVs are retained locally.

Vectorized UKF sigma propagation and covariance operations without changing mathematics. Compared vectorized propagation to scalar reference on randomized states: maximum difference 2.78e-17. Runtime for the executed 180-s estimator baseline was 12.66 s on this machine; this is simulation wall time, not deployed policy latency.

Scientific interpretation follows H1 (information beyond fault labels/generic weighting) separately from H2 (benefit from weighting). No positive EC-QAD conclusion has yet been established.
