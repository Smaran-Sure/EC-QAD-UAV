# EC-QAD-v1 diagnosis

This is a read-only analysis of 115232 logged training samples from 32 complete trajectories. The deployed physical-reference interface, teacher, quantizer, and existing weighting algorithms remain frozen. Existing `ecqad` identifiers mean **EC-QAD-v1 combined**. Negative results are preserved under `results/frozen_ecqad_v1/` and in the episode tables.

## Exact frozen rule

C = log(1 + mean of selected covariance diagonals divided by fixed physical scales). N = log(1 + mean of the three sensor EWMA(NIS/dof) values). Standardize each independently using the entire training population, then clip each z-score to [-3,3]. Covariance-only uses score 0.5*z_C; innovation-only uses 0.5*z_N; EC-QAD-v1 uses 0.25*(z_C+z_N). An offset b is solved so mean(clip(exp(score+b),0.5,2))=1. All calculations precede distillation. These signals are never policy inputs.

## Executed distribution measurements

Covariance/log-NIS Pearson correlation is -0.05965; Spearman correlation is -0.10476. Elevated means above the training mean by more than one training standard deviation, before clipping: covariance is elevated on 1.512% of samples; NIS on 9.421%; both on 0.000%.

EC-QAD-v1 weights have mean 1.00000000, standard deviation 0.27270, 5th/50th/95th percentiles 0.79146/0.91737/1.77915, and effective sample fraction (sum w)^2/(n*sum w²) 0.93078. Within absolute 0.005 of the lower/upper bounds are 0.000% / 3.409% of samples. These are measured saturation fractions, not assumed explanations for instability.

| Active state | Covariance mean weight | Innovation mean weight | EC-QAD-v1 mean weight |
|---|---:|---:|---:|
| nominal | 0.99890 | 0.88298 | 0.91556 |
| gyro_bias | 0.96650 | 0.88909 | 0.90172 |
| gps_dropout | 1.66645 | 0.82778 | 1.44630 |
| airspeed_bias | 0.96154 | 1.82469 | 1.54566 |

Complete marginal quantiles, scenario-wide distributions, active-fault distributions, correlations, update ages, and all three weight distributions are in `results/estimator_signal_distributions.csv` and `results/weight_diagnostics.csv`. Figures: `results/figures/training_signal_distributions.png` and `training_weight_distributions.png`.

## What the implementation establishes

- Raw NIS cannot dominate merely through its units: the two components are standardized separately and receive equal coefficients in the combined score. The variances of their clipped weighted contributions are 0.007669 (covariance) and 0.043722 (NIS). The NIS contribution variance is 5.701 times the covariance contribution variance after clipping. Before clipping both standardized components have unit population variance; clipping changes their effective spread differently. Equal coefficients therefore do not imply equal population influence, physical meaning, or equally useful ranking. This is a measured imbalance, not proof that it caused flight instability.
- The combined score is an average of standardized components, not an unnormalized sum of their standalone scores. The joint-elevation fraction above quantifies overlap; correlation alone cannot establish harmful double-counting.
- NIS is available only at measurement updates. The implementation holds each sensor's previous EWMA between updates and averages all three values. During active GPS dropout, 100.000% of logged samples have no new GPS innovation; the mean/max age of the last GPS statistic is 5.617/14.100 seconds. Heading/airspeed updates may still change the composite. Missing GPS is therefore not represented by a fresh GPS inconsistency statistic or an availability gate.
- Alpha=0.1 corresponds to an EWMA e-folding time of 1.898 s at the nominal 5 Hz GPS rate and 0.475 s at 20 Hz heading/airspeed rates. Measurement loss extends the GPS hold beyond that nominal time scale. Covariance propagates during loss; NIS describes consistency of received measurements. They measure different physical phenomena.
- Population mean-one normalization is enforced, while each fault-state mean can differ, as the table shows. Mean-one does not make gradients identical or guarantee stable low-bit closed-loop control.

## Causal limits and algorithm decision

These diagnostics identify measurable saturation, ranking overlap, freshness, and time-scale differences. They do not establish which mechanism caused a particular trained policy to fail. Closed-loop comparisons across student seeds, covariance-specific shuffled weights, fault balancing, and saliency are required; offline action MSE alone is insufficient. No revised fusion formula or test-set hyperparameter search has been introduced. Covariance-only and innovation-only remain legitimate, separately labeled candidates. The final evidence-based decision is recorded in RESEARCH_VERDICT.md after the matched evaluation finishes.
