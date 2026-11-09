# AIAA AVIATION research summary

## 1. Problem

Low-bit policy approximation can alter closed-loop UAV tracking. A state estimator supplies physical uncertainty and consistency information that could identify where preserving teacher fidelity matters most.

## 2. Literature gap and scope

Quantized policies, quantization-aware policy distillation, uncertainty-aware distillation and importance-weighted losses already exist. [QuaRL](https://research.google/pubs/quantized-reinforcement-learning-quarl/) studies policy quantization. [SQIL (Park et al., ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Park_Saliency-Aware_Quantized_Imitation_Learning_for_Efficient_Robotic_Control_ICCV_2025_paper.pdf) combines quantization-aware imitation and saliency-weighted action distillation. [Logit-uncertainty distillation](https://pmc.ncbi.nlm.nih.gov/articles/PMC11682123/) is another existing uncertainty-weighting example. These sources support a narrow positioning, not a universal priority claim: this study tests covariance and innovation statistics from a physical flight-state estimator as compression-training importance signals. The literature review is targeted, not exhaustive. The implemented Jacobian comparator is sensitivity-inspired and is not presented as a reproduction of SQIL.

## 3. Research question

H1: do estimator signals predict compression-related action error or subsequent tracking degradation beyond fault labels and policy sensitivity? H2: does using those signals as bounded distillation weights improve a low-bit policy? These hypotheses are tested separately.

## 4. EC-QAD algorithm

Dimensionless selected-covariance and normalized-NIS summaries are standardized using training data, combined, exponentiated with a population-normalizing offset, and clipped to [0.5,2]. Their training-population mean is one. The student minimizes weighted teacher-action MSE. Covariance/NIS do not become deployed student inputs. Teacher and student share a physical-navigation/reference interface with no PID internal state or computation. Fixed bit width redistributes training emphasis; it does not allocate bits dynamically.

## 5. Experimental setup

JSBSim Rascal110, simulated IMU/GPS/heading/airspeed, 15-state UKF, cascaded PID expert, 16→32→32→4 MLP teacher/student, 8/4/2-bit affine weight and hidden-activation fake quantization. Whole trajectory splits hold out stronger faults and later onsets. Nominal, gyro-bias, GPS-dropout and airspeed-bias episodes use common random numbers across methods. 160 rollouts and training seeds [7] are represented in the current results. Exact parameters are in METHOD.md and EXPERIMENT_DESIGN.md.

## 6. Initial quantitative findings

At 8 bits, off-nominal EC-QAD score changed by -1.38% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD score changed by +661.42% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD score changed by +15.77% relative to uniform QAD (negative is better).

The largest relative off-nominal improvement was 1.38% at 8 bits.

For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for action_error by +0.28% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for future_tracking by +1.44% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity. For 4-bit uniform QAD, adding covariance/NIS changed held-out prediction MSE for excess_future_tracking by +1.79% beyond fault labels, time, attitude magnitudes, current tracking severity, training-seed indicators and teacher-Jacobian sensitivity.

Additional weighting controls have not yet completed evaluation. These are descriptive paired contrasts; the per-seed table must also be examined. If shuffled or generic balancing/sensitivity performs as well, superiority to uniform QAD alone is not estimator-specific evidence.

The full physical-metric table, failure counts, ablations and paired comparisons are in INITIAL_RESULTS.md and machine-readable results tables.

## 7. Defensible contribution

A reproducible estimator-to-compression training interface, an explicit bounded mean-one algorithm, and executed fixed-wing evidence testing both signal information and weighting efficacy. Improvements, null effects and low-bit failures are outcomes to report. Estimator-specific advantage requires beating shuffled/fault-balanced/sensitivity comparators, not merely uniform QAD.

## 8. Limitations

The aircraft has approximate public BETA aerodynamics; sensors and faults are synthetic; flight is still-air; the teacher imitates a conventional controller using a memoryless physical-state/reference interface. Quantization is simulated, and no hardware latency, energy or speedup was measured. Two-bit prefix RMSE can look deceptively favorable when an episode terminates: use failures and the penalized score. Correlations and conditional prediction do not establish causal compression criticality. H1 uses observed trajectory prefixes and cannot reconstruct counterfactual tracking after early termination; severe low-bit failures therefore restrict its interpretation. All student seeds share the accepted frozen seed-7 teacher; inference is conditional on that one teacher. A separately trained seed-19 teacher failed its tracking gate and is preserved in the rejected-candidate archive. A small number of student training seeds limits inference about optimization variability.

## 9. Candidate introduction contributions

- A training-only interface connecting physical navigation covariance and innovation consistency to quantized-policy fidelity weighting.
- An interpretable, dimensionless, bounded, mean-normalized distillation algorithm with unchanged deployed policy inputs.
- A reproducible 6-DoF fixed-wing benchmark with whole-trajectory held-out sensor faults and degraded observability.
- Separate empirical tests of incremental signal information and the closed-loop benefit of weighting, including negative outcomes and matched controls.

## 10. Draft abstract results paragraph

We evaluated estimator-conditioned quantization-aware distillation in a nonlinear six-degree-of-freedom fixed-wing simulation with simulated sensors and a navigation UKF. Across 160 executed closed-loop rollouts and training seeds [7], At 8 bits, off-nominal EC-QAD score changed by -1.38% relative to uniform QAD (negative is better). At 4 bits, off-nominal EC-QAD score changed by +661.42% relative to uniform QAD (negative is better). At 2 bits, off-nominal EC-QAD score changed by +15.77% relative to uniform QAD (negative is better). The largest relative off-nominal improvement was 1.38% at 8 bits. These results characterize this bounded weighting formulation; they do not establish general robustness or hardware efficiency.
