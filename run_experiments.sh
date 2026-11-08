#!/bin/sh
# Run every arm x seed of experiments.py in parallel (resumable: finished runs are skipped).
# Usage: sh run_experiments.sh [parallel jobs, default 26]
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/rlsom/bin/python}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
mkdir -p out
ARMS=$($PY -c "import experiments as E; print(' '.join(E.ARMS))")
for seed in 100 101 102 103 104 105 106 107 108 109; do
  for arm in $ARMS; do
    echo "$arm $seed"
  done
done | xargs -P "${1:-26}" -n 2 sh -c '$0 -u experiments.py --arm "$1" --seed "$2" || echo "FAILED $1 $2"' "$PY" >> out/runner.log 2>&1
echo "ALL DONE" >> out/runner.log
