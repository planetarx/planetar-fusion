# planetar-fusion

The **buildable v0** of planetar's learned cross-modal fusion thesis — a model that learns the
SAR↔AIS association from AIS-on co-occurrence (free labels) and applies it to flag and re-identify
**dark (AIS-off) vessels**. The full six-modality thesis lives aspirationally in the CH13 SSOT
(`../planetar/docs/recenter-learned-fusion.md`); this repo makes the core idea real on the two
modalities we actually run — Sentinel-1 SAR + AIS — then stages the rest.

> **Status (2026-10-25): Stage 1 built and tested; xView3 validation data in place; first real run done on a local RTX 2070 — best holdout aggregate 0.136 (epoch 1), then it diverged; stability fixed 2026-10-24. Next: rerun with the fixes (2070 or Runpod), then Task 8 proper.** Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/plans/2026-07-26-stage1-xview3-detector.md`](docs/plans/2026-07-26-stage1-xview3-detector.md).

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
| `src/planetar_fusion/eval.py` + `vendor/xview3-reference` | ✅ The leaderboard `MR` via the vendored reference scorer (BSD-3; `git submodule update --init`). `scene_ids=` keeps a scene with zero detections in the score (the reference drops it) |
| `src/planetar_fusion/bench.py` | ✅ `bench split`: validation labels → scene-disjoint `train.csv` / `holdout.csv` (+ `split.json`, seeded). `bench score`: infer every holdout scene → `preds.csv` + `score.json` |
| `tests/` | ✅ 28 tests; `tests/conftest.py` writes a synthetic scene in the xView3 layout (torchgeo has no xView3 loader) |
| `scripts/runpod_setup.sh`, `scripts/pod_run.sh` | ⚠️ Env bring-up, then split → train → holdout score for each config. `pod_run.sh` smoke-tested on CPU with synthetic scenes (2026-10-22); neither yet run on a pod |
| `configs/stage1.yaml`, `configs/stage1-vv.yaml` | 5-channel (leaderboard-comparable) and **VV-only** (runnable in `planetar-sat`, Task 9; xView3 weights are CC BY-NC-SA → demo/benchmark only, never shipped) — both train in one pod session |
| **Task 0 — xView3 data** | ✅ 2026-10-22: all 50 validation scenes + labels + shoreline in `data/xview3/` (124 GB, checksums OK); split pinned (seed 0): 40 train scenes / 15,756 labels, 10 holdout / 3,468 |
| **First real run (RTX 2070)** | ⚠️ 2026-10-23, `configs/stage1-2070.yaml` (5-ch, batch 2, 4 epochs, ~3 h/epoch). Holdout scores below; epoch 2 collapsed and epoch 3 had NaN BatchNorm stats (0 predictions). Cause: fp16 overflow from unbounded z-scores of the near-constant `owiMask` + an fp16 `log(0)` in the focal loss. Fixed 2026-10-24 (6821033): input clip ±10, fp32 loss, non-finite batch skip that restores BN stats, grad clipping, fail-fast. Not yet re-run |
| **Task 8 — RunPod run + real `MR`** | ❌ Not started; needs a credit top-up (full run ≈ $11–15 on a Secure 4090) |
| **Task 9 — `planetar-sat --detector learned`** | ❌ Channel set chosen 2026-10-22: **VV-only** (`stage1-vv.yaml`) — `planetar-sat` reads only the VV measurement GeoTIFF. Open: GRD DN → dB at inference, and radar-geometry vs xView3's reprojected grid; compare against CFAR on a cached scene |

Holdout scores of the 2070 run (10 scenes, `bench score`, thresh 0.3, with shoreline):

| Checkpoint | aggregate | loc F1 | loc F1 shore | vessel F1 | fishing F1 | length acc | preds |
|---|---|---|---|---|---|---|---|
| epoch 0 | 0.101 | 0.243 | 0.000 | 0.781 | 0.000 | 0.304 | 450 |
| epoch 1 | **0.136** | 0.321 | 0.178 | 0.727 | 0.000 | 0.210 | 6,941 |
| epoch 2 | 0.002 | 0.004 | 0.000 | 0.909 | 0.000 | 0.087 | 7 |
| epoch 3 | 0.000 | — | — | — | — | — | 0 |

These are pre-fix numbers from an under-trained, unstable run — not a claimable benchmark.

```
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
git submodule update --init
.venv/bin/pytest -q
```

## What's coming (Stage 1 build)

Python + PyTorch + torchgeo; the xView3 dataloader (start on `torchgeo`'s "xView3 tiny"), the
detector, the reference `MR` scorer, and the `planetar-sat --detector learned` integration. See
[`docs/DESIGN.md` §3](docs/DESIGN.md).
