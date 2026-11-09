# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains 640 executed rollouts across training seeds [7]. Unique randomized scenario seeds per condition: {'airspeed_bias': 20, 'gps_dropout': 20, 'gyro_bias': 20, 'nominal': 20}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher | 32 | 20 / 60 | 0.2502 | 0.2815 | 4.5531 / 4.4737 | 5.2658 / 5.1084 | 0.7744 / 1.0143 | 0/80 | 0.0000 | 7120 |
| uniform | 8 | 20 / 60 | 0.2507 | 0.2820 | 4.5470 / 4.4644 | 5.2593 / 5.1065 | 0.7807 / 1.0195 | 0/80 | 0.0000 | 2128 |
| uniform | 4 | 20 / 60 | 0.3583 | 0.3717 | 7.9163 / 7.4245 | 7.2571 / 7.1160 | 0.8212 / 1.0505 | 0/80 | 0.0000 | 1296 |
| uniform | 2 | 20 / 60 | 6.3559 | 5.8359 | 106.9849 / 120.0343 | 81.0915 / 85.2626 | 3.1091 / 3.1428 | 54/80 | 0.0695 | 880 |
| ecqad | 8 | 20 / 60 | 0.2493 | 0.2810 | 4.5609 / 4.4811 | 5.2399 / 5.0906 | 0.7696 / 1.0114 | 0/80 | 0.0000 | 2128 |
| ecqad | 4 | 20 / 60 | 0.2661 | 0.3015 | 4.6030 / 4.8035 | 5.6115 / 5.5099 | 0.8517 / 1.0854 | 0/80 | 0.0000 | 1296 |
| ecqad | 2 | 20 / 60 | 1.2024 | 1.4961 | 31.0603 / 42.2417 | 17.7313 / 21.0682 | 3.0402 / 2.8943 | 0/80 | 0.0188 | 880 |

## What changed and what did not

At 8 bits, off-nominal EC-QAD score changed by -0.35% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD score changed by -18.89% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD score changed by -74.36% relative to uniform QAD (negative is better).

The largest relative off-nominal improvement was 74.36% at 2 bits. A relative score improvement does not establish acceptable two-bit flight; failures and constraint violations remain essential. These are mean comparisons; see `results/paired_comparisons.csv` for paired intervals and `results/ecqad_ablation.csv` for alternative weighting methods. A small positive difference alone does not establish robustness or estimator-specific benefit. FP32 teacher failures: 0/80. The nominal columns quantify the cost, if any, of off-nominal weighting.

## H1: incremental information

For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for action_error by +0.11% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for future_tracking by +0.02% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for excess_future_tracking by -0.52% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity.

Global and within-condition rank correlations are in `results/signal_analysis.csv`. The conditional analysis holds out complete episode groups. Excess future tracking compares paired but diverging teacher/student rollouts and is not a causal one-action intervention.

## H2: weighting intervention

Additional weighting controls have not yet completed evaluation. These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.

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

The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller with shared error-memory preprocessing. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. A small number of training seeds limits inference about optimization variability.
