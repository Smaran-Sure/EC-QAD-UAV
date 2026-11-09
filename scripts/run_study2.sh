#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JSBSIM_DEBUG=0 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/ecqad-mpl OPENBLAS_NUM_THREADS=1
PYTHON=${PYTHON:-.venv/bin/python}
WORKERS=${WORKERS:-6}
"$PYTHON" -m pytest -q
for stage in freeze collect teachers teacher_gate students signals analyze
do
    "$PYTHON" experiments/run_study2.py "$stage" --workers "$WORKERS"
done
# analyze writes the preregistered go/no-go verdict. A negative gate ends work.
# Positive evidence permits a separately frozen downstream training stage;
# this script deliberately cannot silently launch new weighting or TEST.
