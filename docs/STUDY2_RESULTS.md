# Study 2 executed results

**NO-GO.** The preregistered propagated-covariance signal gate failed. CR-EC-QAD and its shuffled control were not trained, and TEST remains unopened. No alternative score, fusion, or threshold was tried.

Study 2 executed 44 expert trajectories, 116 candidate-selection flights, 24 separate teacher-gate flights, and 480 signal-analysis flights (664 experiment flights in total, excluding short unit tests and the reproducibility replay). The signal set contains 80 unique scenarios, shared across one selected teacher and five student optimization seeds. These are VALIDATION_SIGNAL results, not final TEST results. Study 1 is unchanged.

## Teacher robustness

| Teacher seed | Nominal score / expert | Off-nominal score / expert | Failures / flights | Accepted |
|---:|---:|---:|---:|---|
| 7 | 0.2337 / 0.2256 | 0.2699 / 0.2658 | 0/8 | True |
| 19 | 0.2285 / 0.2256 | 0.2799 / 0.2658 | 0/8 | True |
| 43 | 0.4851 / 0.2256 | 0.4586 / 0.2658 | 0/8 | False |

Seeds 7 and 19 pass the separately applied nominal/off-nominal ≤1.5× expert score criterion with zero failures. Seed 43 fails tracking despite completing all flights; it is retained. The first-passing rule selects seed 7 before student outcomes. This is stronger teacher acceptance evidence than Study 1’s single accepted teacher, but the student signal experiment still uses one frozen teacher.

## Closed-loop checkpoint selection

| Policy | Seed | Selected epoch | Lowest-MSE epoch | Selected offline MSE | Selected flight score |
|---|---:|---:|---:|---:|---:|
| teacher | 7 | 120 | 120 | 0.0032767 | 0.2499 |
| teacher | 19 | 180 | 180 | 0.0034242 | 0.2544 |
| teacher | 43 | 60 | 120 | 0.0037142 | 0.3952 |
| uniform | 7 | 25 | 75 | 0.0015160 | 0.2707 |
| uniform | 19 | 75 | 75 | 0.0011545 | 0.3321 |
| uniform | 43 | 25 | 75 | 0.0016048 | 0.3001 |
| uniform | 61 | 50 | 50 | 0.0012353 | 0.2625 |
| uniform | 89 | 25 | 100 | 0.0013200 | 0.3680 |

The chosen epoch disagrees with the minimum offline-MSE epoch for 3/5 student seeds. Every one of the 29 candidates, including failures, remains in `results/study2_checkpoint_selection.csv`; all 116 selection flights remain in `results/study2/selection_rollouts.csv`. Selection uses four fixed validation flights. That small set is an explicit limitation: no checkpoint is reselected after SIGNAL outcomes.

## Independent signal-validation flight performance

Lower tracking scores are better. N/F means nominal/off-nominal. Physical RMSE is over the observed flight; the composite score additionally penalizes early termination. Each row represents 20 nominal and 60 off-nominal flights.

| Policy | Seed | Score N/F | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 7 | 0.2443 / 0.2773 | 4.4106 / 4.6175 | 5.1110 / 5.0486 | 0.7650 / 0.9866 | 0/80 |
| uniform | 7 | 0.2581 / 0.2829 | 4.9247 / 5.1451 | 5.2251 / 5.1713 | 0.7976 / 0.9611 | 0/80 |
| uniform | 19 | 0.3193 / 0.5305 | 6.1398 / 10.0823 | 5.8277 / 8.8586 | 1.0641 / 1.1888 | 2/80 |
| uniform | 43 | 0.2765 / 0.3062 | 5.2568 / 5.6347 | 5.0591 / 5.1896 | 0.9297 / 1.0865 | 0/80 |
| uniform | 61 | 0.2395 / 0.2731 | 4.3919 / 4.7356 | 4.9090 / 5.1068 | 0.7556 / 0.9381 | 0/80 |
| uniform | 89 | 0.3230 / 0.3499 | 5.7663 / 6.0733 | 5.6774 / 5.6908 | 1.1568 / 1.3123 | 0/80 |

| Policy | Seed | Mean peak altitude error (m) | Mean peak heading error (deg) | Mean peak airspeed error (m/s) | Mean control effort | Mean actuator RMSE vs teacher | Mean constraint fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| teacher | 7 | 30.452283 | 38.373728 | 3.431246 | 0.001166 | 0.000000 | 0.000000 |
| uniform | 7 | 32.443086 | 38.444391 | 3.643117 | 0.001145 | 0.078079 | 0.000000 |
| uniform | 19 | 42.622911 | 45.015121 | 3.904682 | 0.001290 | 0.111366 | 0.000955 |
| uniform | 43 | 35.384142 | 38.933119 | 4.643895 | 0.001247 | 0.098945 | 0.000000 |
| uniform | 61 | 29.753240 | 38.949365 | 3.466142 | 0.001237 | 0.104931 | 0.000000 |
| uniform | 89 | 35.673182 | 38.767844 | 5.768947 | 0.001135 | 0.107486 | 0.000000 |

Peak columns average each episode’s maximum absolute tracking error. Control effort is mean squared aileron/elevator/rudder command, as in Study 1. Action RMSE compares bounded student and teacher commands at the same student feature vector. The teacher row therefore has zero action discrepancy by construction. Full per-episode metrics are in `results/study2_evaluation.csv`; grouped physical metrics are in `results/study2/validation_summary.csv`. Two failures occur for student seed 19; none are removed.

## Signal result and stopping decision

Adding q beyond both old covariance C and teacher sensitivity reduces held-out future-tracking prediction MSE by **0.0905%**, with 97.5% interval **[−0.1273%, +0.2321%]** and positive directions in 4/5 seeds. For excess future tracking the reduction is **0.0202%**, interval **[−0.0632%, +0.0877%]**, positive in 3/5 seeds. Neither interval excludes zero, so the gate fails. These percentages are prediction-MSE changes, not flight-performance improvements.

A useful positive diagnostic remains: sensitivity alone reduces action-discrepancy prediction MSE by **16.3769%** [11.3398%, 21.4430%], with positive directions in 5/5 seeds. q alone reduces that MSE by **5.7163%** [3.7352%, 7.5960%]. Once covariance and sensitivity are included, q adds only **0.4992%** [−0.0258%, 1.0129%]. Thus instantaneous discrepancy is predictable from generic policy sensitivity, while this study does not establish additional covariance-specific prediction of harmful closed-loop effects. No causal interpretation is justified.

The primary intervals use the preregistered two-outcome adjustment. All intervals condition on fitted episode-held-out models and only five student seeds. Strong current truth-tracking nuisance controls, a restricted linear predictor, prefix censoring after failures, narrow still-air maneuvers, approximate Rascal aerodynamics, and conditional covariance excluding independent pitot/gyro-readout noise limit the conclusion.

See [signal analysis](STUDY2_SIGNAL_ANALYSIS.md), [research verdict](STUDY2_RESEARCH_VERDICT.md), and [signal-gate figure](../results/study2/figures/signal_gate.png). The protocol, candidate schedule, quantizer, and physical q definition were not tuned against SIGNAL or TEST. The required downstream covariance/CR/shuffle method grid is intentionally not run because the explicit no-go rule takes precedence. No hardware or flight-safety claim is made.

## Reproduction and preservation

Run `bash scripts/run_study2.sh` from the repository. It stops after the signal decision. The frozen source/config manifest is `results/study2/protocol_manifest.json`; teacher/student checkpoint hashes have separate manifests. `results/study2/completion_audit.json` records the scope and checks. Study 1 is preserved at tag `study1-complete-20261109` and `results/study1_frozen/`. Large trajectories and checkpoints are preserved locally and regenerated from the scripts in a source-only checkout. This document includes reviewed interpretation and tables derived from the generated CSVs; rerunning `analyze` regenerates the core reports.

The descriptive paired uniform-versus-teacher comparisons in `results/study2_paired_comparisons.csv` include 97.5% bootstrap intervals by seed and pooled across seeds, preserving common scenarios and stratifying conditions. Positive relative change means the uniform student has a worse score. These baseline contrasts do not test CR-EC-QAD and do not change the signal-gate decision.
