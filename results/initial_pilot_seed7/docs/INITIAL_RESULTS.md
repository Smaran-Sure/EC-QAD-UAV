# Executed EC-QAD results

This report is generated from `results/per_episode_metrics.csv`; it contains 160 executed rollouts across training seeds [7]. Unique randomized scenario seeds per condition: {'airspeed_bias': 5, 'gps_dropout': 5, 'gyro_bias': 5, 'nominal': 5}. Scenarios are shared across methods and training seeds. Counts below identify the actual scope, rather than the planned scope. The prior spherical-coordinate experiment is quarantined under `results/archive/v0_spherical_coordinate_bug/` and is excluded here. The PID-memory-dependent study in `results/archive/v1_pid_dependent_exploratory/` is also exploratory and excluded. Current teacher/student inputs use `physical_reference_v2`; neural deployments neither instantiate nor execute PID.

## Main table

Score is the normalized heading/altitude/airspeed tracking metric defined in EXPERIMENT_DESIGN.md. N/F means nominal/off-nominal. Failure penalty applies to the score; physical RMSE is over the observed trajectory. A condition labels the scheduled scenario; a failed flight may terminate before its fault begins. See `results/failure_timing.csv` for actual fault exposure and pre-onset failures. Storage is ideal packed weights plus documented FP32 overhead, not an on-disk or hardware measurement.

| Method | Bits | Episodes nominal / faulted | Nominal score | Faulted score | Altitude RMSE N/F (m) | Heading RMSE N/F (deg) | Airspeed RMSE N/F (m/s) | Failures | Constraint fraction | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| teacher | 32 | 5 / 15 | 0.2610 | 0.2840 | 4.5780 / 4.5451 | 5.4237 / 4.9977 | 0.8397 / 1.0458 | 0/20 | 0.0000 | 7120 |
| uniform | 8 | 5 / 15 | 0.2633 | 0.2846 | 4.6653 / 4.5763 | 5.4164 / 4.9938 | 0.8492 / 1.0450 | 0/20 | 0.0000 | 2128 |
| uniform | 4 | 5 / 15 | 0.3582 | 0.4182 | 5.6343 / 6.0861 | 6.8335 / 6.8714 | 1.2952 / 1.6524 | 0/20 | 0.0002 | 1296 |
| uniform | 2 | 5 / 15 | 8.6738 | 8.1264 | 174.7071 / 131.3920 | 75.8620 / 75.4149 | 6.4276 / 6.1754 | 18/20 | 0.3430 | 880 |
| ecqad | 8 | 5 / 15 | 0.2604 | 0.2806 | 4.5884 / 4.5349 | 5.3749 / 4.9716 | 0.8401 / 1.0228 | 0/20 | 0.0000 | 2128 |
| ecqad | 4 | 5 / 15 | 3.0774 | 3.1845 | 76.3930 / 83.2129 | 36.9920 / 32.2446 | 6.7288 / 6.8369 | 3/20 | 0.1422 | 1296 |
| ecqad | 2 | 5 / 15 | 9.3572 | 9.4077 | 162.3516 / 165.1778 | 40.1512 / 40.5164 | 7.5976 / 8.5352 | 20/20 | 0.4780 | 880 |

## What changed and what did not

At 8 bits, off-nominal EC-QAD score changed by -1.38% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD score changed by +661.42% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD score changed by +15.77% relative to uniform QAD (negative is better).

The largest relative off-nominal improvement was 1.38% at 8 bits. These are mean comparisons; see `results/paired_comparisons.csv` for paired intervals and `results/ecqad_ablation.csv` for alternative weighting methods. A small positive difference alone does not establish robustness or estimator-specific benefit. FP32 teacher failures: 0/20. The nominal columns quantify the cost, if any, of off-nominal weighting.

## H1: incremental information

For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for action_error by +0.28% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for future_tracking by +1.44% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for excess_future_tracking by +1.79% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity.

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

The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller using a memoryless physical-state/reference interface. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. H1 uses observed trajectory prefixes and cannot reconstruct counterfactual tracking after early termination; severe low-bit failures therefore restrict its interpretation. All student seeds share the accepted frozen seed-7 teacher; inference is conditional on that one teacher. A separately trained seed-19 teacher failed its tracking gate and is preserved in the rejected-candidate archive. A small number of student training seeds limits inference about optimization variability.
