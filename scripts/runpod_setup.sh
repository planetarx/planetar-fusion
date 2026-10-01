#!/usr/bin/env bash
# Stage 1 pod bring-up (Task 8, step 1). UNTESTED until the first pod run —
# written from the plan's description of the founder's RunPod workflow.
#
#   bash scripts/runpod_setup.sh [DATA_SRC]
#
# DATA_SRC: an rsync/s3 source for the xView3 mirror (validation split +
# labels, Task 0). Leave empty to skip staging and only prepare the env.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== env =="
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]"
git submodule update --init --quiet
.venv/bin/python - <<'PY'
import torch, planetar_fusion
assert torch.cuda.is_available(), "CUDA not visible to torch"
print("ok torch", torch.__version__, torch.cuda.get_device_name(0))
PY

if [ -n "${1:-}" ]; then
  echo "== staging xView3 from $1 =="
  mkdir -p data/xview3
  case "$1" in
    s3://*) aws s3 sync "$1" data/xview3/ ;;
    *)      rsync -a --info=progress2 "$1"/ data/xview3/ ;;
  esac
  ls data/xview3 | head
fi

echo "== smoke: tests =="
.venv/bin/pytest -q
echo "== ready: .venv/bin/python -m planetar_fusion.train --config configs/stage1.yaml =="
