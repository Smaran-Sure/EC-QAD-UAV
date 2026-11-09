# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains 3280 executed rollouts across training seeds [7, 19, 43]. Unique randomized scenario seeds per condition: {'airspeed_bias': 20, 'gps_dropout': 20, 'gyro_bias': 20, 'nominal': 20}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here. The PID-memory-dependent study in `results/archive/v1_pid_dependent_exploratory/` is also exploratory and excluded. Current teacher/student inputs use `physical_reference_v2`; neural deployments neither instantiate nor execute PID.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. A condition labels the scheduled scenario; a failed flight may terminate before its fault begins. See `results/failure_timing.csv` for actual fault exposure and pre-onset failures. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 teacher | 32 | 20 / 60 | 0.2561 | 0.2792 | 4.5210 / 4.5818 | 5.2600 / 5.1194 | 0.8288 / 0.9993 | 0/80 | 0.0000 | 7120 |
| Uniform QAD | 8 | 60 / 180 | 0.2572 | 0.2771 | 4.5541 / 4.5705 | 5.2567 / 5.1212 | 0.8338 / 0.9833 | 0/240 | 0.0000 | 2128 |
| Uniform QAD | 4 | 60 / 180 | 0.3838 | 0.4601 | 6.2130 / 7.0499 | 6.4598 / 7.0820 | 1.4630 / 1.8398 | 0/240 | 0.0005 | 1296 |
| Uniform QAD | 2 | 60 / 180 | 7.8682 | 7.1405 | 109.2979 / 104.2007 | 76.3160 / 80.9705 | 5.6702 / 5.0148 | 190/240 | 0.1719 | 880 |
| EC-QAD-v1 combined | 8 | 60 / 180 | 0.2563 | 0.2779 | 4.5627 / 4.5870 | 5.2515 / 5.1225 | 0.8268 / 0.9881 | 0/240 | 0.0000 | 2128 |
| EC-QAD-v1 combined | 4 | 60 / 180 | 1.3788 | 1.3449 | 30.7740 / 31.9183 | 15.3898 / 15.2523 | 3.1061 / 3.1835 | 15/240 | 0.0475 | 1296 |
| EC-QAD-v1 combined | 2 | 60 / 180 | 9.1430 | 9.2129 | 141.7781 / 144.5580 | 50.2845 / 47.7224 | 7.3509 / 7.8480 | 240/240 | 0.3950 | 880 |

## What changed and what did not

At 8 bits, off-nominal EC-QAD-v1 combined score changed by +0.30% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD-v1 combined score changed by +192.28% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD-v1 combined score changed by +29.02% relative to uniform QAD (negative is better).

No evaluated bit width improved the mean off-nominal score. These are mean comparisons; see `results/paired_comparisons.csv` for paired intervals and `results/ecqad_ablation.csv` for alternative weighting methods. A small positive difference alone does not establish robustness or estimator-specific benefit. FP32 teacher failures: 0/80. The nominal columns quantify the cost, if any, of off-nominal weighting.

## H1: incremental information

For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for action_error by -0.04% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for future_tracking by -0.02% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for excess_future_tracking by -0.03% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity.

Global and within-condition rank correlations are in `results/signal_analysis.csv`. The conditional analysis holds out complete episode groups. Excess future tracking compares paired but diverging teacher/student rollouts and is not a causal one-action intervention.

## H2: weighting intervention

Against covariance, 4-bit off-nominal EC-QAD-v1 combined score changed by +315.99% on 180 matched rollouts (negative favors EC-QAD-v1). Against innovation, 4-bit off-nominal EC-QAD-v1 combined score changed by +259.50% on 180 matched rollouts (negative favors EC-QAD-v1). Against shuffled, 4-bit off-nominal EC-QAD-v1 combined score changed by +183.64% on 180 matched rollouts (negative favors EC-QAD-v1). Against balanced, 4-bit off-nominal EC-QAD-v1 combined score changed by +293.64% on 180 matched rollouts (negative favors EC-QAD-v1). Against fault_balanced, 4-bit off-nominal EC-QAD-v1 combined score changed by +72.62% on 180 matched rollouts (negative favors EC-QAD-v1). Against saliency, 4-bit off-nominal EC-QAD-v1 combined score changed by +49.75% on 180 matched rollouts (negative favors EC-QAD-v1). These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.

The table and paired comparisons evaluate H2 separately from H1. Covariance-only, innovation-only, shuffled, fault-balanced and sensitivity-weighted results appear when executed, rather than being assumed. Favorable correlation does not establish that this bounded weighting function exploits it.

## Figures

- [Architecture](../results/figures/architecture.png)
- [Conventional tracking](../results/figures/conventional_tracking.png)
- [Estimator envelopes](../results/figures/estimator_envelopes.png)
- [Estimator degradation](../results/figures/estimator_degradation.png)
- [Closed-loop comparison](../results/figures/closed_loop_comparison.png)
- [Performance versus bits](../results/figures/performance_vs_bits.png)
- [Ablations](../results/figures/ablation_comparison.png)
- [Signal analysis](../results/figures/signal_compression_analysis.png)

PDF versions are available alongside each PNG. Machine-readable raw and summary tables are retained.

## Limitations

The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller using a memoryless physical-state/reference interface. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. H1 uses observed trajectory prefixes and cannot reconstruct counterfactual tracking after early termination; severe low-bit failures therefore restrict its interpretation. All student seeds share the accepted frozen seed-7 teacher; inference is conditional on that one teacher. A separately trained seed-19 teacher failed its tracking gate and is preserved in the rejected-candidate archive. A small number of student training seeds limits inference about optimization variability.

## Definitive preliminary comparison

Complete planned four-bit grid. Relative differences are percentages of the matched uniform-QAD score at the same bit width; negative is better. Teacher comparison to a nonexistent 32-bit QAD baseline is not reported. Seed SD is across student-seed means, not across timesteps. Paired 95% intervals appear in `results/definitive_preliminary_table.csv`; the covariance-specific comparisons are in `results/covariance_comparisons.csv`.

| Method | Bits | Nominal score | Off-nominal score | Relative to uniform, off-nominal (%) | Student seeds | Off-nominal seed SD |
|---|---:|---:|---:|---:|---:|---:|
| FP32 teacher | 32 | 0.2561 | 0.2792 | — | 1 | — |
| EC-QAD-v1 combined | 8 | 0.2563 | 0.2779 | +0.30 | 3 | 0.0031 |
| Uniform QAD | 8 | 0.2572 | 0.2771 | +0.00 | 3 | 0.0016 |
| Bounded fault balancing | 4 | 0.3018 | 0.3417 | -25.75 | 3 | 0.0499 |
| Covariance-only EC-QAD | 4 | 0.3015 | 0.3233 | -29.74 | 3 | 0.0341 |
| Shuffled covariance weights | 4 | 0.2934 | 0.3169 | -31.13 | 3 | 0.0282 |
| EC-QAD-v1 combined | 4 | 1.3788 | 1.3449 | +192.28 | 3 | 1.6902 |
| Fault-balanced QAD | 4 | 0.8145 | 0.7791 | +69.33 | 3 | 0.5199 |
| Innovation-only EC-QAD | 4 | 0.3178 | 0.3741 | -18.70 | 3 | 0.0947 |
| Policy-saliency comparator | 4 | 0.8155 | 0.8981 | +95.18 | 3 | 0.5677 |
| Shuffled combined weights | 4 | 0.4436 | 0.4742 | +3.05 | 3 | 0.2483 |
| Uniform QAD | 4 | 0.3838 | 0.4601 | +0.00 | 3 | 0.0615 |
| EC-QAD-v1 combined | 2 | 9.1430 | 9.2129 | +29.02 | 3 | 0.4507 |
| Uniform QAD | 2 | 7.8682 | 7.1405 | +0.00 | 3 | 1.3009 |

## Testbed and independent deployment

JSBSim Rascal110 uses public approximate BETA aerodynamics, nonlinear six-DoF dynamics, simulated IMU/GPS/pitot/heading observations, and a 15-state UKF. A 16→32→32→4 MLP outputs all four actuator commands. Weights and hidden activations use 8/4/2-bit affine fake quantization; interfaces/biases stay FP32. The independent FP32 teacher completed 8 held-out 180-s validation flights with 0 failures; mean score 0.268495 versus 0.268092 for the PID expert. Neural PID execution is disabled. All student bit widths use the same physical-reference feature builder; covariance/NIS influence only the training loss.

The exact frozen EC-QAD-v1 rule and complete training-signal distributions are in ECQAD_V1_DIAGNOSIS.md. Covariance-only is a legitimate distinct EC-QAD candidate, not a failed approximation to combined fusion. Both shuffle controls remain separately labeled.

## Scientific conclusion

H1 overall: **unsupported** (covariance: unsupported; NIS: unsupported). Supported within this experiment: covariance-only has +29.74% mean off-nominal improvement relative to Uniform QAD (95% paired bootstrap interval [+21.65, +35.48]%); it improves the per-seed mean in 3/3 student seeds. Consistent superiority is not established: covariance-only has -2.02% mean off-nominal improvement relative to Shuffled covariance weights (95% paired bootstrap interval [-6.32, +5.93]%); it improves the per-seed mean in 1/3 student seeds. EC-QAD-v1 combined has mean nominal/off-nominal scores 1.3788/1.3449, versus 0.3838/0.4601 for uniform QAD, with 15 failures across its 240 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement.

No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion. Detailed A–J answers are in RESEARCH_VERDICT.md.
