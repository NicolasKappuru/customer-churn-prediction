#!/usr/bin/env bash
# Unattended w5 experiments: small Optuna search per GAN and per class, then 5 seeds at full budget.
# The two classes are independent GANs, so they run as two parallel lanes on the GPU
# (small GRUs leave most of the GPU idle; one process is bound by kernel-launch overhead).
# Usage (from anywhere):  setsid workshops/w5/run_all.sh > workshops/w5/artifacts/experiments/run_all.log 2>&1 &
# Restartable: Optuna resumes from its sqlite studies and finished seed runs are skipped.
# LANES="1" or LANES="0" runs a single class.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$HERE/artifacts/experiments"
PY="$(cd "$HERE/../.." && pwd)/.venv/bin/python"
RUN="$PY -u $HERE/run_experiments.py"
TRIALS="${TRIALS:-15}"
SEEDS="${SEEDS:-0 1 2 3 4}"
LANES="${LANES:-1 0}"
mkdir -p "$EXP"
rm -f "$EXP/DONE" "$EXP/FAILED"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

lane() {   # one class: optuna for both GANs, then the seeds
  local label=$1 status=0
  for gan in timegan wtimegan_tcn; do
    $RUN optuna --gan "$gan" --label "$label" --trials "$TRIALS" || status=1
  done
  # shellcheck disable=SC2086
  $RUN seeds --gans timegan wtimegan_tcn --labels "$label" --seeds $SEEDS || status=1
  echo "[$(stamp)] lane label$label finished (status $status)"
  return $status
}

echo "[$(stamp)] start (trials=$TRIALS, seeds=$SEEDS, lanes=$LANES)"
pids=()
for label in $LANES; do
  lane "$label" > "$EXP/run_all_label$label.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
$RUN summary || true

if [ "$status" -eq 0 ]; then
  echo "[$(stamp)] all done"; touch "$EXP/DONE"
else
  echo "[$(stamp)] finished with errors"; touch "$EXP/FAILED"
fi
exit "$status"
