# Study 1 frozen

Completed commit: `576f03bb`; tag: `study1-complete-20261109`.

Study 1 contains 3,280 executed rollouts with three student optimization seeds and one accepted frozen teacher. At four bits covariance-only QAD improves mean off-nominal score by 29.74% versus uniform QAD (95% paired bootstrap interval 21.65–35.48%), but does not consistently outperform covariance-distribution-matched shuffled weights (−2.02% improvement, interval −6.32–5.93%; wins in 1/3 seeds). H1 is unsupported under its specified incremental prediction test. Combined covariance+NIS EC-QAD-v1 is unsupported as a robustness improvement and has 15 failures in 240 four-bit rollouts.

These findings and their limitations remain unchanged. Existing root results and documentation are Study 1. Study 2 uses separate configuration, code entry point, prefixed tables/documents, and `results/study2/` artifacts. `results/study1_frozen/` copies configs, documentation, checkpoints, CSVs, manifests, and figures. Its manifest hashes all pre-existing research files, including large local trajectory archives retained at their original paths. The Git tag preserves source history. No Study 1 checkpoint is converted or retrained in place.
