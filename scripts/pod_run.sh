#!/usr/bin/env bash
# Stage 1 milestone run (Task 8, steps 2–3), after scripts/runpod_setup.sh.
#
#   bash scripts/pod_run.sh [CONFIG ...]      # default: both stage1 configs
#
# Splits the validation labels by scene (once; data/xview3/split/split.json pins it),
# trains each config, then scores its last checkpoint on the holdout scenes. Each run
# leaves <run_dir>/metrics.jsonl and <run_dir>/bench/{preds.csv,score.json}; pull those
# plus the checkpoint back to bb (Task 8, step 4). Re-running a config appends to its
# metrics.jsonl and overwrites its checkpoints — move a finished run aside first.
#
# AUTO_STOP=1 stops this Runpod pod when the script exits, success or failure, so an
# unattended run stops billing for the GPU (the disk is kept for pulling results).
set -euo pipefail
if [ "${AUTO_STOP:-0}" = 1 ]; then
  trap 'echo "== exit $? at $(date -u +%FT%TZ); stopping pod $RUNPOD_POD_ID =="; runpodctl stop pod "$RUNPOD_POD_ID"' EXIT
fi
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LABELS=${LABELS:-data/xview3/validation.csv}
SPLIT=data/xview3/split
SHORE=${SHORE:-data/xview3/shoreline}
HOLDOUT=${HOLDOUT:-10}

[ -f "$LABELS" ] || { echo "missing $LABELS — stage the xView3 validation split first (Task 0)"; exit 1; }
[ -d "$SHORE" ] || { echo "missing $SHORE — without it the close-to-shore F1 scores 0"; exit 1; }
if [ ! -f "$SPLIT/split.json" ]; then
  $PY -m planetar_fusion.bench split --labels "$LABELS" --out "$SPLIT" --holdout "$HOLDOUT" --seed 0
fi

configs=("$@")
[ ${#configs[@]} -gt 0 ] || configs=(configs/stage1.yaml configs/stage1-vv.yaml)

for cfg in "${configs[@]}"; do
  echo "== train $cfg =="
  ckpt=$($PY -m planetar_fusion.train --config "$cfg" | tail -1 | $PY -c 'import json,sys; print(json.load(sys.stdin)["last_ckpt"])')
  run_dir=$($PY -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["out"]["run_dir"])' "$cfg")
  echo "== score $ckpt on the holdout =="
  $PY -m planetar_fusion.bench score --ckpt "$ckpt" --root data/xview3 \
      --labels "$SPLIT/holdout.csv" --shore-root "$SHORE" --out "$run_dir/bench"
done
