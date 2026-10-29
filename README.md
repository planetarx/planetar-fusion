# planetar-fusion

The **buildable v0** of planetar's learned cross-modal fusion thesis — a model that learns the
SAR↔AIS association from AIS-on co-occurrence (free labels) and applies it to flag and re-identify
**dark (AIS-off) vessels**. The full six-modality thesis lives aspirationally in the CH13 SSOT
(`../planetar/docs/recenter-learned-fusion.md`); this repo makes the core idea real on the two
modalities we actually run — Sentinel-1 SAR + AIS — then stages the rest.

> **Status (2026-10-29): Stage 1 built and tested; xView3 validation data in place. Best holdout aggregate 0.253 (Run 3, 5-channel, epoch 3, thresh 0.15). The dual-pol (VH+VV) model `planetar-sat` can run peaks at 0.189 (Run 4, epoch 7, thresh 0.25); its 12-epoch constant-LR run destabilised after epoch 8. Task 9 works end to end on a real Sentinel-1C scene. Next: an LR schedule + best-checkpoint selection, then the dark-vessel step.** Design: [`docs/DESIGN.md`](docs/DESIGN.md). Plan: [`docs/plans/2026-07-26-stage1-xview3-detector.md`](docs/plans/2026-07-26-stage1-xview3-detector.md).

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
| `tests/` | ✅ 29 tests; `tests/conftest.py` writes a synthetic scene in the xView3 layout (torchgeo has no xView3 loader) |
| `scripts/runpod_setup.sh`, `scripts/pod_run.sh` | ⚠️ Env bring-up, then split → train → holdout score for each config. `pod_run.sh` smoke-tested on CPU with synthetic scenes (2026-10-22); neither yet run on a pod |
| `configs/stage1.yaml`, `configs/stage1-vv.yaml` | 5-channel (leaderboard-comparable) and **VV-only** (runnable in `planetar-sat`, Task 9; xView3 weights are CC BY-NC-SA → demo/benchmark only, never shipped) — both train in one pod session |
| **Task 0 — xView3 data** | ✅ 2026-10-22: all 50 validation scenes + labels + shoreline in `data/xview3/` (124 GB, checksums OK); split pinned (seed 0): 40 train scenes / 15,756 labels, 10 holdout / 3,468 |
| **Run 1 (RTX 2070)** | ⚠️ 2026-10-23, `configs/stage1-2070.yaml` (5-ch, batch 2, 4 epochs, ~3 h/epoch). Holdout scores below; epoch 2 collapsed and epoch 3 had NaN BatchNorm stats (0 predictions). Cause: fp16 overflow from unbounded z-scores of the near-constant `owiMask` + an fp16 `log(0)` in the focal loss. Fixed 2026-10-24 (6821033): input clip ±10, fp32 loss, non-finite batch skip that restores BN stats, grad clipping, fail-fast |
| **Run 2 (2026-10-26)** | ❌ Fixes above only. Skipped batches 25 → 2,986 → 18,617 per epoch; pre-BN conv weight norms grew 40–60× (e.g. `d3.0` 6.5 → 393), BN running_var to 8e7. Stopped after epoch 2. Holdout: ep1 0.051, ep2 0.006 |
| **Run 3 (2026-10-27)** | ✅ + AdamW weight decay 0.05 on conv weights only (2771de1, `train.make_optimizer`). Skipped batches 26 / 26 / 15 / 12; train loss 1.97 → 0.81 → 0.68 → 0.65; largest conv norm levels off ~195. ~1.8 h/epoch at 4.2 it/s once the SAR rasters were re-tiled (below). Holdout scores below |
| **SAR re-tile (2026-10-27)** | ✅ xView3 ships VH/VV as uncompressed one-row strips, so each 800-px tile read pulled ~92 MB per channel and the loader was disk-bound. `scripts/retile_xview3.py data/xview3` rewrites them in place as 512×512 tiles (lossless, verified per file). Run it after any fresh download |
| **Task 8 — RunPod run + real `MR`** | ❌ Not started; needs a credit top-up (full run ≈ $11–15 on a Secure 4090) |
| **Run 4 — dual-pol (2026-10-28)** | ⚠️ `configs/stage1-dualpol.yaml` (VH+VV only, 12 epochs, wd 0.05, constant LR 3e-4). Stable to epoch 8, then skipped batches 125 → 1,469 → 2,584 → 9,259 and length accuracy collapsed; ep11 scores 0.102 at best. Best checkpoint **epoch 7: 0.189 at thresh 0.25**. Results below |
| **Task 9 — `planetar-sat --detector learned`** | ✅ 2026-10-28/29, `planetar-sat` branch `learned-detector`. Uses the **dual-pol** model (VH+VV, both in every CDSE 1SDV GRD), not VV-only. DN → 20·log10 dB, staged as tiled rasters; geocoded with the annotation grid. On the S1C 2026-05-08 scene (radar geometry, no CRS), ep7 at 0.25 gave 36 sea detections vs CFAR's 4,275; all 36 sit within 200 m of a CFAR detection (median 17 m, median CFAR SNR 21), covering 18 of CFAR's 50 detections above SNR 20. No AIS for that date, so neither side is ground-truthed. `runs/demo/` |

Holdout scores, Run 3 (10 scenes, `bench score`, thresh 0.3, with shoreline; `runs/stage1-2070-wd/`):

| Checkpoint | aggregate | loc F1 | loc F1 shore | vessel F1 | fishing F1 | length acc | preds |
|---|---|---|---|---|---|---|---|
| epoch 0 | 0.015 | 0.038 | 0.000 | 0.902 | 0.000 | 0.070 | 64 |
| epoch 1 | 0.137 | 0.353 | 0.019 | 0.705 | 0.000 | 0.213 | 686 |
| epoch 2 | 0.193 | 0.509 | 0.038 | 0.721 | 0.000 | 0.136 | 1,781 |
| epoch 3 | **0.216** | 0.503 | 0.017 | 0.715 | 0.000 | 0.413 | 1,113 |

Weak spots: close-to-shore F1 ≈ 0.02, fishing F1 = 0 in every run, and 1,113
predictions against 3,468 holdout labels at thresh 0.3, so a lower threshold may lift loc F1.
Four epochs on 40 scenes is under-trained, and the holdout is our own 10 scenes, not xView3's
withheld set — a progress number, not a leaderboard claim.

Threshold sweep of epoch 3 (2026-10-28; one inference pass at 0.05, then re-filtered by score,
which matches re-running inference because NMS only lets a higher-scoring point suppress;
`runs/stage1-2070-wd/sweep/sweep.json`):

| thresh | 0.05 | 0.1 | **0.15** | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| aggregate | 0.169 | 0.247 | **0.253** | 0.242 | 0.229 | 0.216 | 0.174 | 0.113 |
| loc F1 | 0.380 | 0.558 | 0.583 | 0.567 | 0.537 | 0.503 | 0.409 | 0.277 |
| loc F1 shore | 0.181 | 0.166 | 0.096 | 0.049 | 0.026 | 0.017 | 0.009 | 0.011 |
| preds | 6,704 | 2,711 | 1,882 | 1,525 | 1,289 | 1,113 | 799 | 493 |

The threshold was picked on the holdout itself, so 0.253 is slightly optimistic; pick it on a
train-side scene next time. Close-to-shore detections sit at low scores (shore F1 0.18 at 0.05).

Run 4 (dual-pol) threshold sweeps, aggregate per checkpoint (`runs/stage1-dualpol/sweep-ep<N>/sweep.json`; ep11 in `sweep/`):

| thresh | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|---|
| epoch 3 | 0.103 | 0.154 | 0.161 | 0.055 | 0 | 0 | 0 |
| epoch 5 | 0.123 | 0.169 | 0.148 | 0.105 | 0.020 | 0 | 0 |
| epoch 7 | 0.092 | 0.135 | 0.167 | 0.185 | **0.189** | 0.155 | 0.044 |
| epoch 8 | 0.056 | 0.109 | 0.149 | 0.171 | 0.158 | 0.131 | 0.035 |
| epoch 11 | 0.062 | 0.083 | 0.095 | 0.102 | 0.095 | 0.077 | 0.015 |

Epoch 7 at 0.25: loc F1 0.452, shore 0.023, vessel 0.727, fishing 0, length acc 0.337, 1,400
preds. Dual-pol trails the 5-channel Run 3 (0.253) — the ancillary channels (wind, bathymetry)
help — but it is the model a raw GRD scene can feed.

Run 1 (pre-fix, `runs/stage1-2070-prefix/`), for reference: ep0 0.101, ep1 0.136, ep2 0.002, ep3 0
(NaN BatchNorm stats).

```
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
git submodule update --init
.venv/bin/pytest -q
```

## What's coming (Stage 1 build)

Python + PyTorch + torchgeo; the xView3 dataloader (start on `torchgeo`'s "xView3 tiny"), the
detector, the reference `MR` scorer, and the `planetar-sat --detector learned` integration. See
[`docs/DESIGN.md` §3](docs/DESIGN.md).
