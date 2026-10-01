# planetar-fusion

The **buildable v0** of planetar's learned cross-modal fusion thesis — a model that learns the
SAR↔AIS association from AIS-on co-occurrence (free labels) and applies it to flag and re-identify
**dark (AIS-off) vessels**. The full six-modality thesis lives aspirationally in the CH13 SSOT
(`../planetar/docs/recenter-learned-fusion.md`); this repo makes the core idea real on the two
modalities we actually run — Sentinel-1 SAR + AIS — then stages the rest.

> **Status: Stage 1 code built and tested (Tasks 1–7, 2026-10-01); awaiting the xView3 download (Task 0) and the first RunPod training run (Task 8).** Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/plans/2026-07-26-stage1-xview3-detector.md`](docs/plans/2026-07-26-stage1-xview3-detector.md).

## The v0, in two stages (single GPU throughout)

1. **SAR dark-vessel detector — on xView3 (now).** A strong Sentinel-1 vessel detector, trained and
   **benchmarked against the public xView3 leaderboard**; darkness is detection − AIS at deployment.
   Replaces `planetar-sat`'s classical CFAR detector.
2. **Cross-modal association / fusion head — on self-collected paired data (next).** The actual
   learned SAR↔AIS association (InfoNCE over co-occurring pairs), using AIS kinematic features
   xView3 can't provide; associate or flag-as-dark, with a gallery on-ramp to cross-time re-ID.

## What's here now

| Path | State |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | ✅ The full v0 design (data, architecture, training, eval, integration) |
| `src/planetar_fusion/data.py` | ✅ `XView3Dataset` (tiling on a shared grid, 5-channel z-scored stacks) + `gaussian_splat` CenterNet targets |
| `src/planetar_fusion/model.py` | ✅ `CenterNetSAR` — U-Net, four full-resolution heads (heatmap, log-length, is_vessel, is_fishing) |
| `src/planetar_fusion/losses.py` | ✅ Penalty-reduced focal + L1(log1p length) + BCE at GT centres; proven by an overfit-one-batch test |
| `src/planetar_fusion/train.py`, `configs/stage1.yaml` | ✅ Config-driven loop → per-epoch checkpoints + `runs/metrics.jsonl` |
| `src/planetar_fusion/infer.py` | ✅ Full-scene tile → peaks → global NMS → xView3 CSV; recovers the synthetic fixture's targets |
| `src/planetar_fusion/eval.py` + `vendor/xview3-reference` | ✅ The leaderboard `MR` via the vendored reference scorer (BSD-3; `git submodule update --init`) |
| `tests/` | ✅ 18 tests; `tests/conftest.py` writes a synthetic scene in the xView3 layout (torchgeo has no xView3 loader) |
| `scripts/runpod_setup.sh` | ⚠️ Written, not yet run on a pod |
| **Task 0 — xView3 data** | ❌ Human prerequisite: register at iuu.xview.us, download validation (50 scenes, ~123 GB) + labels to `data/xview3/` |
| **Task 8 — RunPod run + real `MR`** | ❌ Blocked on Task 0 |
| **Task 9 — `planetar-sat --detector learned`** | ❌ Needs a channel-set decision: planetar-sat scenes are single-pol GRD intensity, not the 5-channel xView3 stack |

```
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
git submodule update --init
.venv/bin/pytest -q
```

## What's coming (Stage 1 build)

Python + PyTorch + torchgeo; the xView3 dataloader (start on `torchgeo`'s "xView3 tiny"), the
detector, the reference `MR` scorer, and the `planetar-sat --detector learned` integration. See
[`docs/DESIGN.md` §3](docs/DESIGN.md).
