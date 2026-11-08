#!/bin/sh
# Run every arm x seed of experiments_rate.py in parallel (resumable).
# Usage: sh run_rate.sh [parallel jobs, default 26]
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/rlsom/bin/python}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
mkdir -p out/rate_200k
ARMS=$($PY -c "import experiments_rate as R; print(' '.join(R.STUDY))")
for seed in $(seq 160 179); do
  for arm in $ARMS; do
    echo "$arm $seed"
  done
done | xargs -P "${1:-26}" -n 2 sh -c '$0 -u experiments_rate.py --arm "$1" --seed "$2" || echo "FAILED $1 $2"' "$PY" >> out/rate_200k/runner.log 2>&1
echo "ALL DONE" >> out/rate_200k/runner.log
