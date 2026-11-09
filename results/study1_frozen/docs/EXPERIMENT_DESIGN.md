# Experiment design

## Fixed comparison

All methods share a 16→32→32→4 architecture, deployment feature builder, sensor suite/UKF, 180-s reference family, data population and quantizer. The teacher is FP32; students use 8-, 4-, or 2-bit weights and hidden activations with FP32 interfaces/biases. Uniform QAD and combined EC-QAD receive equal 100-epoch optimization budgets and matched minibatch ordering. No held-out evaluation result is used to tune the conditioning function. Additional methods at four bits: covariance-only, innovation-only, distribution-matched shuffled weighting, bounded inverse-frequency weighting, exact fault-balanced weighting, and teacher-Jacobian sensitivity weighting. PTQ checkpoints are exported without optimization; no PTQ closed-loop result is claimed unless it appears in the episode table.

Teacher weights are frozen after 180 epochs and separate validation-based checkpoint selection. Student checkpoints also minimize unweighted validation action MSE. The sensitivity comparator is an explicitly simple policy-Jacobian baseline, not a claimed faithful SQIL reproduction. True-error/fault-label oracles and covariance-input policies are not prerequisites for the initial study.

## Scenario split and reproducibility

The config `configs/study.yaml` is authoritative. Training uses 8 complete trajectories per condition (32 total); validation uses 2 per condition (8 total). Initial evaluation uses 20 per condition per method/training seed (80 rollouts). Planned training seeds: 7,19,43. Dataset scenario seeds are fixed independently of optimization seeds. A scenario seed is `split_seed_base + 1000*condition_index + episode_index`, with bases 10000/20000/30000 for training/validation/evaluation. Common random numbers pair all controllers on the same scenario. No timestep-random train/test split is used.

Conditions: nominal, one-axis gyro bias (axis and sign randomized), GPS dropout, and airspeed bias (sign randomized). Gyro and airspeed biases recover at 145 s; GPS recovers after the sampled duration. No actuator-fault detection claim is made.

| Parameter | Training | Validation | Evaluation |
|---|---:|---:|---:|
| Fault onset (s) | 45–65 | 68–78 | 80–105 |
| Gyro bias magnitude (deg/s) | 0.2–0.5 | 0.5–0.65 | 0.65–1.0 |
| GPS dropout duration (s) | 8–15 | 15–18 | 18–25 |
| Airspeed bias magnitude (m/s) | 0.5–1.5 | 1.5–2.0 | 2.0–3.0 |

Across all splits: initial altitude ±10 m, initial calibrated airspeed ±1 m/s, heading ±5°, maneuver amplitude factor 0.8–1.2, reference-time factor 0.93–1.07. Sensor noise and constant biases are resampled per episode. The initial implementation is still-air and does not claim wind/disturbance generalization. Separate scenarios rather than arbitrary timesteps define the split.

## Metrics and acceptance

Primary physical metrics: altitude RMSE (m), heading RMSE (deg), airspeed RMSE (m/s), peaks, constraint fractions and failure rates. RMSE includes startup and command steps. Peak tracking changes include the commanded step itself. The normalized combined score is the square root of the mean of squared altitude error/20 m, heading error/20°, and airspeed error/3 m/s, equally averaged. It facilitates comparison but is not a calibrated safety measure.

A constraint sample occurs at |roll|>60°, |pitch|>30°, airspeed outside [12,40] m/s, or altitude below 100 m. Failure termination: nonfinite state, altitude below 30 m, airspeed outside [5,65] m/s, or |pitch|>1.2 rad. For terminated trajectories, the unflown fraction receives squared normalized-score penalty 100. Physical RMSE on terminated runs covers only the flown prefix; always interpret it with failure rates and penalized score. Constraint fraction refers to the observed prefix. This avoids silently dropping failures or presenting a short failed flight as successful tracking.

Plant baseline: finite controlled flight for at least 60 s; full 180-s sequence without sustained saturation. Estimator: finite PSD covariance, reasonable nominal state errors and NIS, growing/recovering position covariance under GPS loss. Teacher gate: no validation failures and mean normalized tracking score no more than 1.5 times the conventional-controller score. This gate is evaluated on validation trajectories, before interpreting student comparisons. Two-bit failure is a reportable experimental outcome, not grounds to exclude the bit width.

Secondary metrics: teacher/student action MSE, control effort, bit width and ideal packed storage (including specified FP32 overhead). No hardware performance claims.

## H1 and H2

H1 analysis uses uniformly distilled students only. Record covariance score, EWMA NIS/dof, teacher–student action discrepancy, five-second future tracking error, and excess future tracking relative to a paired teacher rollout. Report global and within-condition rank correlations as descriptive, temporally dependent quantities without timestep-independent significance claims.

Conditional regression compares a baseline with fault category, active-fault status/category interactions, time and time², estimated roll/pitch magnitudes teacher-Jacobian sensitivity, current truth-based tracking severity (an evaluation-only nuisance control), and training-seed indicators against the same baseline augmented with covariance/NIS. Five folds hold out whole episode-index groups; the same scenario across training seeds stays in one fold. Predictive targets use an asinh transform. A reduction in held-out prediction MSE is evidence of incremental information within this simulation, not causation. Paired excess tracking from diverging closed-loop trajectories is not a controlled one-action causal perturbation test.

H2 uses common-scenario paired differences between EC-QAD and uniform QAD at equal bits. Report separate nominal/off-nominal means and paired bootstrap intervals that resample training seeds and shared scenario IDs, stratified by condition. A one-training-seed interval describes scenario variation conditional on that trained policy. Multiple training seeds must also be examined separately; pooling episodes alone does not create independent training replications. Small effects, failed low-bit policies, or shuffled/saliency baselines explaining the improvement limit the claim.

The first result is preserved before extending the training-seed count. Extensions should not change the aircraft, data splits, weighting function or tuning budget in response to a preferred evaluation outcome.

The conditional regression is a restricted predictive comparison, not an information-theoretic result: for a fixed deterministic teacher/student, pointwise action discrepancy is itself a function of policy inputs. Incremental covariance/NIS prediction relative to a limited regression baseline should not be interpreted as information unavailable to an arbitrarily expressive model of the full policy input.

## Corrected deployment architecture gates

All v1 PID-memory-dependent outputs are exploratory and archived. The v2 MLP receives only the named physical-state/reference feature vector. Before compression evaluation, the independent teacher must complete eight held-out 180-second validation episodes without failure and with mean normalized tracking score at most 1.5 times the PID baseline. Runtime metadata must show no PID execution. The student deployment gate runs all six uniform/EC-QAD bit-width variants with this identical interface; low-bit flight failures are reported as findings, not hidden by the interface gate. Tests forbid PID construction in neural deployment and reject old-schema checkpoints.

The seed-7 teacher is the single accepted frozen teacher for student optimization seeds 7, 19, and 43. These are distillation replications, not independent end-to-end teacher replications. A corrected-interface seed-19 teacher candidate failed validation (score/PID ratio 7.70608, zero early terminations) and is preserved under `results/archive/v2_rejected_teacher_seed19/`; it is not used. No EC-QAD weighting, model budget, or evaluation scenario was changed in response. Teacher-training robustness remains a limitation.

## Frozen EC-QAD-v1 diagnosis protocol (2026-11-09)

The pre-diagnosis corrected source/results are frozen in `results/frozen_ecqad_v1/`. `ecqad` is the immutable code identifier for **EC-QAD-v1 combined**; `shuffled` permutes those combined weights. The additional `covariance_shuffled` control permutes the covariance-only weight array exactly, using the student seed, and changes no runtime interface or existing formula. Primary diagnosis uses four bits and student seeds 7, 19, 43 against the single frozen teacher.

H1 now tests covariance, log-normalized NIS, and both separately. The baseline includes fault category, signed/absolute severity interactions, gyro axis, active-fault status, onset/duration, time/time², commanded altitude/speed/heading sine/cosine, estimated roll/pitch magnitudes, current truth-based tracking severity, teacher Jacobian sensitivity, and student-seed indicators. Predictors are standardized within each training fold. Fixed ridge 1e-6 stabilizes collinear linear fits; the intercept is unpenalized. Targets use asinh. Five folds hold out whole episode-index groups across all seeds. Primary prediction MSE gives equal weight to episodes. Paired hierarchical bootstrap intervals resample student seeds and scenarios within condition, using fixed out-of-fold prediction errors; they do not include uncertainty from refitting the predictive model. Constraint violations are analyzed separately at episode level because complete per-timestep violation signals were not retained for every rollout.

H1 is supported for a signal when both four-bit future-tracking targets show positive incremental prediction improvement with 95% intervals above zero; partially supported when at least one of the three targets does; otherwise unsupported by this specified test. This is a restricted predictive test with a strong truth-based nuisance control, not proof that a signal contains no useful information or evidence of causality. No revised fusion is selected from held-out evaluation outcomes.
