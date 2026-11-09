# Research verdict

Incomplete planned grid; conclusions are provisional. All results here use the independent `physical_reference_v2` interface. Three student optimization seeds share one frozen, validated FP32 teacher; these are not three independent teachers. No revised fusion formula was fitted to evaluation results.

**A. Does covariance contain useful compression-criticality information?** Unsupported by the specified incremental prediction test. See the target-specific estimates below; this restricted linear test with current truth tracking as a nuisance covariate cannot prove absence of all useful information.

**B. Does innovation/NIS contain useful compression-criticality information?** Unsupported by the same test. Correlations are descriptive, not causal evidence.

**C. Does covariance conditioning consistently outperform uniform QAD?** Supported within this experiment: covariance-only has +23.50% mean off-nominal improvement (95% paired bootstrap interval [+14.27, +31.38]%); it improves the per-seed mean in 1/1 student seeds.

**D. Does it outperform distribution-matched shuffled covariance weights?** Pending matched evaluation. This is the relevant estimator-specificity comparison; the older `shuffled` control matches combined weights and answers a different question.

**E. Does it outperform fault-balanced QAD?** Supported within this experiment: covariance-only has +35.12% mean off-nominal improvement (95% paired bootstrap interval [+29.96, +39.79]%); it improves the per-seed mean in 1/1 student seeds.

**F. Does it outperform policy-saliency weighting?** Supported within this experiment: covariance-only has +78.74% mean off-nominal improvement (95% paired bootstrap interval [+74.77, +81.95]%); it improves the per-seed mean in 1/1 student seeds. The comparator is teacher-Jacobian sensitivity weighting, not an exact SQIL reproduction.

**G. Does combined EC-QAD-v1 work?** EC-QAD-v1 combined has mean nominal/off-nominal scores 3.4403/3.2953, versus 0.3043/0.4083 for uniform QAD, with 15 failures across its 80 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement.

**H. What is the algorithm for the next stage?** The matched three-seed comparison is incomplete. No final algorithm is selected. No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion.

**I. Supported claims.** An independent estimator-based neural controller can be distilled with training-only physical-estimator importance weights. The tables establish the measured within-testbed effects, their student-seed variability, and the negative outcomes. H1 overall is unsupported under this specified analysis. Any improvement is conditional on the tested aircraft, single teacher, trajectories, and quantizer.

**J. Unsupported claims.** General safety or fault tolerance; causal compression criticality from correlation; calibrated fault probabilities from NIS; hardware latency/energy/speedup; universal superiority of covariance conditioning; or estimator specificity without consistent superiority to the covariance-matched shuffle. No result from a PID-dependent policy is included.

## H1 estimates

| Signal | Target | Prediction MSE reduction (%) | 95% interval (%) |
|---|---|---:|---:|
| covariance | action_error | +0.002 | [-0.061, +0.057] |
| nis | action_error | +0.017 | [-0.328, +0.335] |
| covariance+nis | action_error | +0.015 | [-0.329, +0.340] |
| covariance | future_tracking | -0.030 | [-0.064, -0.005] |
| nis | future_tracking | -0.085 | [-0.179, +0.000] |
| covariance+nis | future_tracking | -0.117 | [-0.215, -0.025] |
| covariance | excess_future_tracking | -0.014 | [-0.070, +0.045] |
| nis | excess_future_tracking | +0.225 | [-0.314, +0.736] |
| covariance+nis | excess_future_tracking | +0.218 | [-0.316, +0.744] |

Intervals resample student seeds and common scenarios within fault category, conditional on fixed out-of-fold predictions. Whole episodes, including the same scenario across seeds, remain within a single validation fold. Tests are exploratory and are not corrected for multiple comparisons. Severe-failure trajectories contribute only observed prefixes; no counterfactual future after termination is invented.
