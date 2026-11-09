# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains 1120 executed rollouts across training seeds [7]. Unique randomized scenario seeds per condition: {'airspeed_bias': 20, 'gps_dropout': 20, 'gyro_bias': 20, 'nominal': 20}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here. The PID-memory-dependent study in `results/archive/v1_pid_dependent_exploratory/` is also exploratory and excluded. Current teacher/student inputs use `physical_reference_v2`; neural deployments neither instantiate nor execute PID.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. A condition labels the scheduled scenario; a failed flight may terminate before its fault begins. See `results/failure_timing.csv` for actual fault exposure and pre-onset failures. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 teacher | 32 | 20 / 60 | 0.2561 | 0.2792 | 4.5210 / 4.5818 | 5.2600 / 5.1194 | 0.8288 / 0.9993 | 0/80 | 0.0000 | 7120 |
| Uniform QAD | 8 | 20 / 60 | 0.2575 | 0.2788 | 4.6120 / 4.6104 | 5.2562 / 5.1170 | 0.8299 / 0.9922 | 0/80 | 0.0000 | 2128 |
| Uniform QAD | 4 | 20 / 60 | 0.3043 | 0.4083 | 5.1429 / 6.0151 | 5.9113 / 6.6966 | 1.0533 / 1.6111 | 0/80 | 0.0001 | 1296 |
| Uniform QAD | 2 | 20 / 60 | 8.6954 | 8.1190 | 159.1020 / 141.9872 | 78.1043 / 78.3438 | 6.3227 / 6.1868 | 70/80 | 0.3732 | 880 |
| EC-QAD-v1 combined | 8 | 20 / 60 | 0.2546 | 0.2751 | 4.5442 / 4.5481 | 5.2258 / 5.1030 | 0.8183 / 0.9723 | 0/80 | 0.0000 | 2128 |
| EC-QAD-v1 combined | 4 | 20 / 60 | 3.4403 | 3.2953 | 80.0869 / 83.4879 | 33.8332 / 33.3679 | 6.8192 / 6.7621 | 15/80 | 0.1423 | 1296 |
| EC-QAD-v1 combined | 2 | 20 / 60 | 9.3780 | 9.4094 | 162.5375 / 164.2435 | 40.4183 / 40.2095 | 7.9752 / 8.5721 | 80/80 | 0.4793 | 880 |

## What changed and what did not

At 8 bits, off-nominal EC-QAD-v1 combined score changed by -1.34% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD-v1 combined score changed by +707.09% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD-v1 combined score changed by +15.89% relative to uniform QAD (negative is better).

The largest relative off-nominal improvement was 1.34% at 8 bits. These are mean comparisons; see `results/paired_comparisons.csv` for paired intervals and `results/ecqad_ablation.csv` for alternative weighting methods. A small positive difference alone does not establish robustness or estimator-specific benefit. FP32 teacher failures: 0/80. The nominal columns quantify the cost, if any, of off-nominal weighting.

## H1: incremental information

For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for action_error by -0.02% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for future_tracking by +0.12% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for excess_future_tracking by -0.22% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity.

Global and within-condition rank correlations are in `results/signal_analysis.csv`. The conditional analysis holds out complete episode groups. Excess future tracking compares paired but diverging teacher/student rollouts and is not a causal one-action intervention.

## H2: weighting intervention

Against covariance, 4-bit off-nominal EC-QAD-v1 combined score changed by +954.99% on 60 matched rollouts (negative favors EC-QAD). Against innovation, 4-bit off-nominal EC-QAD-v1 combined score changed by +893.83% on 60 matched rollouts (negative favors EC-QAD). Against shuffled, 4-bit off-nominal EC-QAD-v1 combined score changed by +335.59% on 60 matched rollouts (negative favors EC-QAD). Against balanced, 4-bit off-nominal EC-QAD-v1 combined score changed by +742.89% on 60 matched rollouts (negative favors EC-QAD). Against fault_balanced, 4-bit off-nominal EC-QAD-v1 combined score changed by +584.43% on 60 matched rollouts (negative favors EC-QAD). Against saliency, 4-bit off-nominal EC-QAD-v1 combined score changed by +124.32% on 60 matched rollouts (negative favors EC-QAD). These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.

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

Incomplete planned grid; conclusions are provisional. Relative differences are percentages of the matched uniform-QAD score at the same bit width; negative is better. Teacher comparison to a nonexistent 32-bit QAD baseline is not reported. Seed SD is across student-seed means, not across timesteps. Paired 95% intervals appear in `results/definitive_preliminary_table.csv`; the covariance-specific comparisons are in `results/covariance_comparisons.csv`.

| Method | Bits | Nominal score | Off-nominal score | Relative to uniform, off-nominal (%) | Student seeds | Off-nominal seed SD |
|---|---:|---:|---:|---:|---:|---:|
| FP32 teacher | 32 | 0.2561 | 0.2792 | — | 1 | — |
| EC-QAD-v1 combined | 8 | 0.2546 | 0.2751 | -1.34 | 1 | — |
| Uniform QAD | 8 | 0.2575 | 0.2788 | +0.00 | 1 | — |
| Bounded fault balancing | 4 | 0.3720 | 0.3910 | -4.25 | 1 | — |
| Covariance-only EC-QAD | 4 | 0.2827 | 0.3124 | -23.50 | 1 | — |
| EC-QAD-v1 combined | 4 | 3.4403 | 3.2953 | +707.09 | 1 | — |
| Fault-balanced QAD | 4 | 0.4814 | 0.4815 | +17.92 | 1 | — |
| Innovation-only EC-QAD | 4 | 0.3064 | 0.3316 | -18.79 | 1 | — |
| Policy-saliency comparator | 4 | 1.7300 | 1.4691 | +259.80 | 1 | — |
| Shuffled combined weights | 4 | 0.6967 | 0.7565 | +85.29 | 1 | — |
| Uniform QAD | 4 | 0.3043 | 0.4083 | +0.00 | 1 | — |
| EC-QAD-v1 combined | 2 | 9.3780 | 9.4094 | +15.89 | 1 | — |
| Uniform QAD | 2 | 8.6954 | 8.1190 | +0.00 | 1 | — |

## Testbed and independent deployment

JSBSim Rascal110 uses public approximate BETA aerodynamics, nonlinear six-DoF dynamics, simulated IMU/GPS/pitot/heading observations, and a 15-state UKF. A 16→32→32→4 MLP outputs all four actuator commands. Weights and hidden activations use 8/4/2-bit affine fake quantization; interfaces/biases stay FP32. The independent FP32 teacher completed 8 held-out 180-s validation flights with 0 failures; mean score 0.268495 versus 0.268092 for the PID expert. Neural PID execution is disabled. All student bit widths use the same physical-reference feature builder; covariance/NIS influence only the training loss.

The exact frozen EC-QAD-v1 rule and complete training-signal distributions are in ECQAD_V1_DIAGNOSIS.md. Covariance-only is a legitimate distinct EC-QAD candidate, not a failed approximation to combined fusion. Both shuffle controls remain separately labeled.

## Scientific conclusion

H1 overall: **unsupported** (covariance: unsupported; NIS: unsupported). Supported within this experiment: covariance-only has +23.50% mean off-nominal improvement (95% paired bootstrap interval [+14.27, +31.38]%); it improves the per-seed mean in 1/1 student seeds. Pending matched evaluation. EC-QAD-v1 combined has mean nominal/off-nominal scores 3.4403/3.2953, versus 0.3043/0.4083 for uniform QAD, with 15 failures across its 80 four-bit rollouts. Its off-nominal mean is worse; the combined formulation is not supported as a robustness improvement.

The matched three-seed comparison is incomplete. No final algorithm is selected. No final superiority claim is established. Covariance-only remains the simplest physically interpretable EC-QAD candidate for a preregistered next-stage replication; retain uniform QAD as the reference and do not promote EC-QAD-v1 fusion. Detailed A–J answers are in RESEARCH_VERDICT.md.
