# Study 2 signal analysis

NO-GO: the propagated signal does not satisfy the predefined evidence gate. No CR-EC-QAD or revised covariance formula will be trained.

This analysis uses only VALIDATION_SIGNAL trajectories, disjoint from TRAIN, checkpoint-selection validation, teacher-gate validation, and unopened TEST. It includes five independently optimized uniform four-bit students sharing the predefined accepted bounded teacher. No evaluation outcome selected a checkpoint. These are exploratory simulation findings, not causal effects.

**A. Raw estimator covariance:** Not supported on the prespecified future-tracking outcomes under this analysis.
**B. Teacher sensitivity:** Not supported on the prespecified future-tracking outcomes under this analysis.
**C. Propagated uncertainty beyond covariance and sensitivity:** Not supported on the prespecified future-tracking outcomes under this analysis.

**D. Uncertainty:** the table reports hierarchical paired bootstrap intervals over student seeds and common scenario groups, conditional on fixed episode-held-out predictions. The two primary outcomes use 97.5% intervals (Bonferroni for two 95%-family tests). A lower bound above zero is required, not merely a positive point estimate.

**E. Seed consistency:** each row records positive-seed counts; exact per-seed reductions are in the CSV. The gate requires at least four of five directions positive for the same primary outcome whose interval excludes zero.

| Predictor | Baseline | Outcome | MSE reduction (%) | 97.5% interval (%) | Positive seeds |
|---|---|---|---:|---:|---:|
| covariance | baseline | action_error | -0.0102 | [-0.0280, +0.0066] | 0/5 |
| trace_P | baseline | action_error | -0.0029 | [-0.0375, +0.0312] | 3/5 |
| sensitivity | baseline | action_error | +16.3769 | [+11.3398, +21.4430] | 5/5 |
| q | baseline | action_error | +5.7163 | [+3.7352, +7.5960] | 5/5 |
| nis | baseline | action_error | -0.0079 | [-0.0390, +0.0293] | 1/5 |
| covariance+sensitivity+q | covariance+sensitivity | action_error | +0.4992 | [-0.0258, +1.0129] | 5/5 |
| covariance | baseline | future_tracking | +0.0181 | [-0.0262, +0.0780] | 2/5 |
| trace_P | baseline | future_tracking | +0.0224 | [-0.0228, +0.0716] | 5/5 |
| sensitivity | baseline | future_tracking | -0.0322 | [-0.1797, +0.0460] | 2/5 |
| q | baseline | future_tracking | -0.0031 | [-0.0347, +0.0247] | 3/5 |
| nis | baseline | future_tracking | -0.0627 | [-0.1895, +0.0440] | 0/5 |
| covariance+sensitivity+q | covariance+sensitivity | future_tracking | +0.0905 | [-0.1273, +0.2321] | 4/5 |
| covariance | baseline | excess_future_tracking | +0.0799 | [-0.1047, +0.2069] | 3/5 |
| trace_P | baseline | excess_future_tracking | +0.0302 | [-0.0840, +0.1061] | 3/5 |
| sensitivity | baseline | excess_future_tracking | -0.0903 | [-0.2688, +0.0227] | 1/5 |
| q | baseline | excess_future_tracking | -0.0237 | [-0.1188, +0.1052] | 2/5 |
| nis | baseline | excess_future_tracking | +0.2262 | [-0.1352, +0.4770] | 5/5 |
| covariance+sensitivity+q | covariance+sensitivity | excess_future_tracking | +0.0202 | [-0.0632, +0.0877] | 3/5 |

Whole scenario episodes and all their student-seed copies share a fold (episode index modulo five). Baseline covariates include fault category/active status/severity, time, onset/duration, gyro axis, current truth tracking severity, attitude magnitude, command references, and seed indicators. Fault labels and current truth tracking are analysis-only nuisance covariates, not controller or estimator inputs. Fold-local standardization and fixed ridge 1e-6 predict asinh-transformed outcomes. Episode-average errors receive equal weight; severe failures contribute only observed prefixes with a full five-second future. Missing futures are not imputed. Covariance/NIS rank correlations are descriptive. The primary q test adds log1p(q) beyond both old scalar covariance and unscaled-feature teacher-Jacobian Frobenius norm; raw trace(P) is a unit-dependent diagnostic only.

The score propagates UKF uncertainty conditional on fixed pitot and latest gyro readings. It does not include independent sensor-readout uncertainty outside P, or capture nonlinear finite excursions. A negative finding under this restricted predictive model cannot prove absence of all estimator information. TEST remains unopened; no held-out TEST performance claim follows from these validation results.

## Interpretation of the action-discrepancy diagnostic

Sensitivity alone has a positive 16.3769% action-error prediction MSE reduction (97.5% interval 11.3398–21.4430%; 5/5 positive seed directions). q alone also predicts action discrepancy, but its increment beyond covariance plus sensitivity is 0.4992%, with an interval crossing zero (−0.0258–1.0129%). This positive sensitivity result is retained. It does not satisfy the gate, which concerns future or excess future tracking and requires incremental q evidence beyond both comparators.
