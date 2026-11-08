#!/bin/sh
# Run every arm x seed of experiments_burnin.py in parallel (resumable).
# Usage: sh run_burnin.sh [parallel jobs, default 26]
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/rlsom/bin/python}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
mkdir -p out/burnin_200k
ARMS=$($PY -c "import experiments_burnin as B; print(' '.join(B.STUDY))")
for seed in $(seq 140 159); do
  for arm in $ARMS; do
    echo "$arm $seed"
  done
done | xargs -P "${1:-26}" -n 2 sh -c '$0 -u experiments_burnin.py --arm "$1" --seed "$2" || echo "FAILED $1 $2"' "$PY" >> out/burnin_200k/runner.log 2>&1
echo "ALL DONE" >> out/burnin_200k/runner.log
