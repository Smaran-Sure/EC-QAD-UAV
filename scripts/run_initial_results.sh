#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JSBSIM_DEBUG=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ecqad-mpl OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" -m pytest -q
"$PYTHON" experiments/run_study.py collect --workers 4
"$PYTHON" experiments/run_study.py teacher --seed 7
"$PYTHON" experiments/run_study.py teacher_gate --seed 7 --workers 4
"$PYTHON" experiments/run_study.py students --seed 7
"$PYTHON" experiments/run_study.py deployment_gate --seed 7 --workers 4
"$PYTHON" experiments/run_study.py evaluate --seed 7 --episodes 20 --workers 4
"$PYTHON" experiments/run_study.py report
