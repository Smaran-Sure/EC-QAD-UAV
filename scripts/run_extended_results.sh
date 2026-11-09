#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JSBSIM_DEBUG=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ecqad-mpl OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-.venv/bin/python}
bash scripts/run_initial_results.sh
"$PYTHON" - <<'PY'
from pathlib import Path
import shutil
import pandas as pd
out=Path('results/initial_seed7')
if not out.exists():
    data=pd.read_csv('results/per_episode_metrics.csv')
    data=data[(data.training_seed==7)&data.method.isin(['pid','teacher','uniform','ecqad'])&(data.episode<20)]
    assert len(data)==640, 'Initial comparison must finish before snapshotting'
    out.mkdir()
    data.to_csv(out/'per_episode_metrics.csv',index=False)
    for name in ['metrics.csv','summary.csv','paired_comparisons.csv','signal_analysis.csv','training_runs.csv']:
        shutil.copy2(Path('results')/name,out/name)
    shutil.copy2('docs/INITIAL_RESULTS.md',out/'INITIAL_RESULTS.md')
    shutil.copytree('results/figures',out/'figures')
PY
for seed in 7 19 43
do
    "$PYTHON" experiments/run_study.py students --seed "$seed" --ablations
    "$PYTHON" experiments/run_study.py deployment_gate --seed "$seed" --workers 4
    if [ "$seed" = 7 ]; then
        "$PYTHON" experiments/run_study.py evaluate --seed "$seed" --episodes 20 --workers 4 --ablations
    else
        "$PYTHON" experiments/run_study.py evaluate --seed "$seed" --episodes 20 --workers 4 --ablations --methods teacher,uniform,ecqad,covariance,innovation,shuffled,balanced,saliency,fault_balanced
    fi
    "$PYTHON" experiments/run_study.py report
done
