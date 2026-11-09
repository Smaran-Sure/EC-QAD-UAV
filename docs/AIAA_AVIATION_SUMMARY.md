# AIAA AVIATION initial research summary

## 1. Literature gap

Quantized policy learning, importance-weighted distillation, uncertainty-aware distillation, and belief-space control already exist. [QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) and [SQIL](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.pdf) motivate comparison to quantization and policy-saliency weighting. The narrower question is whether uncertainty/innovation signals from a physical navigation estimator identify compression-critical samples and improve low-bit control beyond generic weighting. This is not a claim of inventing uncertainty-weighted distillation or a comprehensive priority review.

## 2. Research question

Does the association between physical estimator information and individual training timesteps provide useful compression-training importance for an independently deployed neural UAV controller?

## 3. H1 and H2

H1 tests incremental prediction of teacher–student action discrepancy, future tracking, and excess tracking beyond fault/status/severity, time, maneuvers, current tracking, and teacher sensitivity. H2 tests the effect of the frozen weighting interventions with matched scenarios and student seeds. H1 overall: unsupported; covariance: unsupported; NIS: unsupported. Neither correlation nor success of a weighted loss establishes causality.

## 4. Method

Teacher and student receive only physical navigation/reference features. PID executes solely for expert demonstrations and baseline flights. Covariance/NIS set bounded [0.5,2], population-mean-one training weights multiplying teacher-action MSE. EC-QAD-v1 combines independently standardized/clipped covariance and log-NIS scores as 0.25*(z_C+z_N). Covariance-only and innovation-only use 0.5*z_C and 0.5*z_N respectively. A normalizing log offset precedes clipping. Confidence signals are not deployed inputs; bit width is fixed.

## 5. Experimental design

JSBSim Rascal110, synthetic sensors, 15-state UKF, and a 16→32→32→4 MLP; 8/4/2-bit affine weight/hidden-activation fake quantization. Nominal, gyro-bias, GPS-dropout, and airspeed-bias scenarios hold out stronger severities and later onsets by whole trajectory. Student seeds are [7, 19, 43]; all share teacher seed 7. Incomplete planned grid; conclusions are provisional; 1120 executed evaluation rollouts are represented. The independent FP32 teacher completed 8 held-out 180-s validation flights with 0 failures; mean score 0.268495 versus 0.268092 for the PID expert. Neural PID execution is disabled. All student bit widths use the same physical-reference feature builder; covariance/NIS influence only the training loss.

## 6. Strongest actual evidence

The lowest observed four-bit off-nominal mean is Covariance-only EC-QAD at 0.3124; this descriptive ranking is not validation-based algorithm selection. At four bits, covariance-only had mean nominal/off-nominal tracking scores 0.2827/0.3124, compared with 0.3043/0.4083 for uniform QAD. Supported within this experiment: covariance-only has +23.50% mean off-nominal improvement (95% paired bootstrap interval [+14.27, +31.38]%); it improves the per-seed mean in 1/1 student seeds.

Pending matched evaluation. Supported within this experiment: covariance-only has +35.12% mean off-nominal improvement (95% paired bootstrap interval [+29.96, +39.79]%); it improves the per-seed mean in 1/1 student seeds. Supported within this experiment: covariance-only has +78.74% mean off-nominal improvement (95% paired bootstrap interval [+74.77, +81.95]%); it improves the per-seed mean in 1/1 student seeds.

## 7. Strongest negative evidence

EC-QAD-v1 combined has mean nominal/off-nominal scores 3.4403/3.2953, versus 0.3043/0.4083 for uniform QAD, with 15 failures across its 80 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement. Two-bit failures are retained, with pre-fault termination distinguished from actual fault exposure. A separately trained seed-19 teacher failed its validation tracking gate and is preserved; it is not used in the compression study.

## 8. Defensible contribution

- A training-only physical-estimator interface for independently deployed quantized flight-control policies.
- Reproducible comparisons of covariance, innovation, combined, matched-shuffle, fault-balanced, and sensitivity weighting.
- Separate episode-held-out H1 tests and matched closed-loop H2 measurements, including negative results.
- A six-DoF simulation demonstration with explicit teacher, quantization, and failure-accounting limitations.

The matched three-seed comparison is incomplete. No final algorithm is selected. No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion.

## 9. Limitations

Approximate public BETA aircraft aerodynamics, synthetic still-air sensor scenarios, a narrow maneuver family, one accepted teacher, three student optimization seeds, and 20 unique scenarios per condition. Reusing matched scenarios across seeds does not create 60 independent scenario draws. H1 uses a restricted predictive family and a truth-based current-tracking nuisance covariate; failure prefixes limit future-error interpretation. Confidence intervals with three seeds are exploratory. No measured hardware latency, energy, speedup, real-flight safety, or causal compression-criticality claim is supported.

## 10. Draft abstract-results paragraph

In 1120 executed nonlinear fixed-wing closed-loop rollouts, we evaluated training-only estimator conditioning for quantized neural control. The lowest observed four-bit off-nominal mean is Covariance-only EC-QAD at 0.3124; this descriptive ranking is not validation-based algorithm selection. At four bits, covariance-only had mean nominal/off-nominal tracking scores 0.2827/0.3124, compared with 0.3043/0.4083 for uniform QAD. Supported within this experiment: covariance-only has +23.50% mean off-nominal improvement (95% paired bootstrap interval [+14.27, +31.38]%); it improves the per-seed mean in 1/1 student seeds. Pending matched evaluation. EC-QAD-v1 combined has mean nominal/off-nominal scores 3.4403/3.2953, versus 0.3043/0.4083 for uniform QAD, with 15 failures across its 80 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement. Incremental covariance/NIS prediction was unsupported under the prespecified whole-episode analysis. These results are conditional on one frozen teacher and the tested simulation; no hardware or causal safety claim is made.
