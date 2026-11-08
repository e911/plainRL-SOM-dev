#!/bin/sh
# Run every arm x seed of experiments_replication.py in parallel (resumable).
# Usage: sh run_replication.sh [parallel jobs, default 26]
set -e
cd "$(dirname "$0")"
PY=${PY:-$HOME/.virtualenvs/rlsom/bin/python}
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1
mkdir -p out/replication_200k
ARMS=$($PY -c "import experiments_replication as R; print(' '.join(R.STUDY))")
for seed in $(seq 120 139); do
  for arm in $ARMS; do
    echo "$arm $seed"
  done
done | xargs -P "${1:-26}" -n 2 sh -c '$0 -u experiments_replication.py --arm "$1" --seed "$2" || echo "FAILED $1 $2"' "$PY" >> out/replication_200k/runner.log 2>&1
echo "ALL DONE" >> out/replication_200k/runner.log
