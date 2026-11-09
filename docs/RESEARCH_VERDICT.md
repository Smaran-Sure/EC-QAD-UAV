# Research verdict

Complete planned four-bit grid. All results here use the independent `physical_reference_v2` interface. Three student optimization seeds share one frozen, validated FP32 teacher; these are not three independent teachers. No revised fusion formula was fitted to evaluation results.

**A. Does covariance contain useful compression-criticality information?** Unsupported by the specified incremental prediction test. See the target-specific estimates below; this restricted linear test with current truth tracking as a nuisance covariate cannot prove absence of all useful information.

**B. Does innovation/NIS contain useful compression-criticality information?** Unsupported by the same test. Correlations are descriptive, not causal evidence.

**C. Does covariance conditioning consistently outperform uniform QAD?** Supported within this experiment: covariance-only has +29.74% mean off-nominal improvement relative to Uniform QAD (95% paired bootstrap interval [+21.65, +35.48]%); it improves the per-seed mean in 3/3 student seeds.

**D. Does it outperform distribution-matched shuffled covariance weights?** Consistent superiority is not established: covariance-only has -2.02% mean off-nominal improvement relative to Shuffled covariance weights (95% paired bootstrap interval [-6.32, +5.93]%); it improves the per-seed mean in 1/3 student seeds. This is the relevant estimator-specificity comparison; the older `shuffled` control matches combined weights and answers a different question.

**E. Does it outperform fault-balanced QAD?** Supported within this experiment: covariance-only has +58.50% mean off-nominal improvement relative to Fault-balanced QAD (95% paired bootstrap interval [+24.72, +77.92]%); it improves the per-seed mean in 3/3 student seeds.

**F. Does it outperform policy-saliency weighting?** Consistent superiority is not established: covariance-only has +64.00% mean off-nominal improvement relative to Policy-saliency comparator (95% paired bootstrap interval [+22.08, +78.72]%); it improves the per-seed mean in 2/3 student seeds. The comparator is teacher-Jacobian sensitivity weighting, not an exact SQIL reproduction.

**G. Does combined EC-QAD-v1 work?** EC-QAD-v1 combined has mean nominal/off-nominal scores 1.3788/1.3449, versus 0.3838/0.4601 for uniform QAD, with 15 failures across its 240 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement.

**H. What is the algorithm for the next stage?** No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion.

**I. Supported claims.** An independent estimator-based neural controller can be distilled with training-only physical-estimator importance weights. The tables establish the measured within-testbed effects, their student-seed variability, and the negative outcomes. H1 overall is unsupported under this specified analysis. Any improvement is conditional on the tested aircraft, single teacher, trajectories, and quantizer.

**J. Unsupported claims.** General safety or fault tolerance; causal compression criticality from correlation; calibrated fault probabilities from NIS; hardware latency/energy/speedup; universal superiority of covariance conditioning; or estimator specificity without consistent superiority to the covariance-matched shuffle. No result from a PID-dependent policy is included.

## H1 estimates

| Signal | Target | Prediction MSE reduction (%) | 95% interval (%) |
|---|---|---:|---:|
| covariance | action_error | +0.028 | [-0.054, +0.117] |
| nis | action_error | +0.004 | [-0.169, +0.118] |
| covariance+nis | action_error | +0.035 | [-0.172, +0.179] |
| covariance | future_tracking | -0.018 | [-0.048, -0.000] |
| nis | future_tracking | +0.036 | [-0.157, +0.176] |
| covariance+nis | future_tracking | +0.018 | [-0.199, +0.162] |
| covariance | excess_future_tracking | -0.006 | [-0.049, +0.039] |
| nis | excess_future_tracking | +0.028 | [-0.206, +0.369] |
| covariance+nis | excess_future_tracking | +0.027 | [-0.213, +0.384] |

Intervals resample student seeds and common scenarios within fault category, conditional on fixed out-of-fold predictions. Whole episodes, including the same scenario across seeds, remain within a single validation fold. Tests are exploratory and are not corrected for multiple comparisons. Severe-failure trajectories contribute only observed prefixes; no counterfactual future after termination is invented.
