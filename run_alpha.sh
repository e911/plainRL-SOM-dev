#!/bin/sh
# Run every arm x seed of experiments_alpha.py in parallel (resumable).
# Usage: sh run_alpha.sh [parallel jobs, default 26]
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/rlsom/bin/python}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
mkdir -p out/alpha_200k
ARMS=$($PY -c "import experiments_alpha as A; print(' '.join(A.STUDY))")
for seed in 110 111 112 113 114 115 116 117 118 119; do
  for arm in $ARMS; do
    echo "$arm $seed"
  done
done | xargs -P "${1:-26}" -n 2 sh -c '$0 -u experiments_alpha.py --arm "$1" --seed "$2" || echo "FAILED $1 $2"' "$PY" >> out/alpha_200k/runner.log 2>&1
echo "ALL DONE" >> out/alpha_200k/runner.log
