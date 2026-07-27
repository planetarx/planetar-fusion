# planetar-fusion Stage 1 (xView3 SAR detector) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A Sentinel-1 vessel detector trained on xView3-SAR and scored against the public xView3 `MR` benchmark, then exposed as a learned detector backend for `planetar-sat`.

**Architecture:** CenterNet-style heatmap detector (objects in Sentinel-1 are ~point-like at 10 m/px, so keypoint/heatmap detection beats anchor boxes). Input = VH + VV + ancillary rasters, tiled into patches; a small FPN/U-Net backbone → an objectness heatmap + per-object `length` / `is_vessel` / `is_fishing` heads. Inference tiles a full scene, extracts peaks, and NMS-stitches across tile borders. "Dark" is **not** a model head — at deployment it is detections minus the live AIS feed.

**Tech Stack:** Python 3.11+, PyTorch, torchgeo + rasterio (Sentinel-1 GeoTIFF + ancillary rasters), numpy/pandas, PyYAML, pytest. Training on **RunPod** (single GPU). Eval via a fork of the xView3 reference scorer.

## Global Constraints

- **Data is credentialed-open, not automatic.** xView3 requires registering at `https://iuu.xview.us/dataset` and accepting terms (Task 0, a human prerequisite). **Mirror our copy on first download** — it is hosted on iuu.xview.us, not AWS Open Data (retirement risk). Work from **validation (50 scenes, ~123 GB) + a train subset**, not the full ~2 TB.
- **License:** xView3 is commercial-usable (DIU "free and open"); the reference scorer is MIT/Apache. **No Global Fishing Watch (CC BY-NC) data in this model.**
- **Compute: RunPod, single GPU** (V100/3090/4090/A100-class). Training runs on the pod; checkpoints + metrics come back down. Dev/tests run locally on CPU with `torchgeo`'s "xView3 tiny".
- **"Dark" = detection − AIS at deployment**, never a SAR-only classifier (DESIGN §3.2, §6).
- **Benchmark honestly:** report the xView3 `MR` and its components (`F1_D`, `F1_S`, `is_vessel` F1, `is_fishing` F1, length error) via the reference scorer; state the split used and compare to the published reference/leaderboard.

---

## File Structure

```
pyproject.toml                       # Task 1 — package + deps
src/planetar_fusion/__init__.py      # Task 1
src/planetar_fusion/data.py          # Task 2 — XView3Dataset (tiling, channels, heatmap targets)
src/planetar_fusion/model.py         # Task 3 — CenterNetSAR (backbone + heads)
src/planetar_fusion/losses.py        # Task 4 — focal heatmap + length L1 + class BCE
src/planetar_fusion/train.py         # Task 5 — config-driven training loop
src/planetar_fusion/infer.py         # Task 6 — full-scene tile→detect→NMS-stitch→CSV
src/planetar_fusion/eval.py          # Task 7 — wrap the xView3 reference scorer (MR)
configs/stage1.yaml                  # Task 5 — hyperparams, channels, paths
scripts/runpod_setup.sh              # Task 8 — pod env + data staging
tests/test_data.py tests/test_model.py tests/test_losses.py tests/test_eval.py
vendor/xview3-reference/             # Task 7 — forked scorer (git submodule or copied)
```

Git-ignored (`.gitignore` already set): `data/`, `runs/`, `checkpoints/`, `*.tif`, `.venv/`.

---

### Task 0 (human prerequisite): obtain xView3

**Not a code task — the founder does this once.**
- [ ] Register at `https://iuu.xview.us/dataset`, accept terms, download the **validation split** (50 scenes) + the **labels** (`validation.csv`) to `data/xview3/`. Optionally a handful of train scenes. **Mirror to our own storage** (e.g. a RunPod volume / S3) on download.
- [ ] Note the exact on-disk layout (per-scene folder with `VH_dB.tif`, `VV_dB.tif`, `bathymetry.tif`, `owiMask.tif`/land-ice, `owiWindSpeed.tif`, …) — Task 2 reads these paths.

---

### Task 1: Project scaffold + environment

**Files:** Create `pyproject.toml`, `src/planetar_fusion/__init__.py`.

**Interfaces:** Produces the importable package `planetar_fusion` and the dependency set every later task uses.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "planetar-fusion"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "torch>=2.2", "torchvision>=0.17", "torchgeo>=0.6", "rasterio>=1.3",
  "numpy>=1.26", "pandas>=2.1", "pyyaml>=6", "tqdm>=4.66",
]
[project.optional-dependencies]
dev = ["pytest>=8"]
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create `src/planetar_fusion/__init__.py`** with `__version__ = "0.1.0"`.

- [ ] **Step 3: Install and verify import**

Run: `python -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -c "import planetar_fusion, torch, torchgeo, rasterio; print('ok', torch.__version__)"`
Expected: prints `ok <version>` with no ImportError. *(On a CPU dev box, CPU torch is fine for Tasks 1–7.)*

- [ ] **Step 4: Commit** — `git add pyproject.toml src && git commit -m "stage1: package scaffold + deps"`

---

### Task 2: xView3 dataset + heatmap targets

**Files:** Create `src/planetar_fusion/data.py`, `tests/test_data.py`.

**Interfaces:**
- Produces `XView3Dataset(root, labels_csv, channels, tile=800, stride=704)`, a `torch.utils.data.Dataset` whose `__getitem__` returns `(image: FloatTensor[C,tile,tile], target: dict)` where `target = {"heatmap": FloatTensor[1,tile,tile], "points": LongTensor[N,2] (row,col), "length": FloatTensor[N], "is_vessel": FloatTensor[N], "is_fishing": FloatTensor[N]}`.
- Produces `gaussian_splat(hw, points, sigma) -> FloatTensor[1,H,W]` (CenterNet target).
- `channels` default `["VH_dB","VV_dB","bathymetry","owiMask","owiWindSpeed"]` → `C=5`.

- [ ] **Step 1: Write the failing test** `tests/test_data.py`

```python
import torch, numpy as np
from planetar_fusion.data import gaussian_splat

def test_gaussian_splat_peaks_at_points():
    hm = gaussian_splat((0, 0, 32, 32), torch.tensor([[8, 8], [20, 24]]), sigma=2.0)
    assert hm.shape == (1, 32, 32)
    assert torch.isclose(hm[0, 8, 8], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(hm[0, 20, 24], torch.tensor(1.0), atol=1e-4)
    assert hm.max() <= 1.0 + 1e-5 and hm.min() >= 0.0
```

- [ ] **Step 2: Run it, expect failure** — `.venv/bin/pytest tests/test_data.py -q` → FAILS (no `gaussian_splat`).

- [ ] **Step 3: Implement `data.py`**

```python
"""xView3 Sentinel-1 dataset: tile scenes, stack channels, build CenterNet targets."""
import numpy as np, pandas as pd, rasterio, torch
from pathlib import Path
from torch.utils.data import Dataset

def gaussian_splat(box, points, sigma=2.0):
    _, _, h, w = box
    hm = torch.zeros(1, h, w)
    if len(points) == 0:
        return hm
    ys = torch.arange(h).view(h, 1); xs = torch.arange(w).view(1, w)
    for r, c in points.tolist():
        g = torch.exp(-(((ys - r) ** 2 + (xs - c) ** 2) / (2 * sigma ** 2)))
        hm[0] = torch.maximum(hm[0], g)
    return hm

def _read(path, window=None):
    with rasterio.open(path) as ds:
        a = ds.read(1, window=window).astype("float32")
    return np.nan_to_num(a, nan=0.0)

class XView3Dataset(Dataset):
    def __init__(self, root, labels_csv, channels=None, tile=800, stride=704, sigma=2.0):
        self.root = Path(root); self.tile = tile; self.stride = stride; self.sigma = sigma
        self.channels = channels or ["VH_dB","VV_dB","bathymetry","owiMask","owiWindSpeed"]
        df = pd.read_csv(labels_csv)
        self.labels = df  # columns incl. scene_id, detect_scene_row, detect_scene_column,
                          # is_vessel, is_fishing, vessel_length_m
        # Build the tile index: (scene_id, row0, col0) over each scene's VH raster grid.
        self.index = []
        for scene_id in df["scene_id"].unique():
            vh = self.root / scene_id / f"{self.channels[0]}.tif"
            with rasterio.open(vh) as ds:
                H, W = ds.height, ds.width
            for r0 in range(0, max(1, H - tile + 1), stride):
                for c0 in range(0, max(1, W - tile + 1), stride):
                    self.index.append((scene_id, r0, c0))

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        scene_id, r0, c0 = self.index[i]
        win = rasterio.windows.Window(c0, r0, self.tile, self.tile)
        chans = [_read(self.root / scene_id / f"{ch}.tif", win) for ch in self.channels]
        # normalize per-channel (dB already ~[-40,0]; scale roughly to [-1,1])
        img = torch.from_numpy(np.stack(chans)).float()
        img = (img - img.mean(dim=(1,2), keepdim=True)) / (img.std(dim=(1,2), keepdim=True) + 1e-6)
        d = self.labels[self.labels.scene_id == scene_id]
        pts, length, isv, isf = [], [], [], []
        for _, row in d.iterrows():
            r, c = int(row.detect_scene_row) - r0, int(row.detect_scene_column) - c0
            if 0 <= r < self.tile and 0 <= c < self.tile:
                pts.append([r, c]); length.append(float(row.get("vessel_length_m", 0) or 0))
                isv.append(float(bool(row.get("is_vessel", True))))
                isf.append(float(bool(row.get("is_fishing", False))))
        pts = torch.tensor(pts, dtype=torch.long) if pts else torch.zeros(0, 2, dtype=torch.long)
        target = {"heatmap": gaussian_splat((0,0,self.tile,self.tile), pts, self.sigma),
                  "points": pts, "length": torch.tensor(length),
                  "is_vessel": torch.tensor(isv), "is_fishing": torch.tensor(isf)}
        return img, target
```

- [ ] **Step 4: Run the test, expect pass** — `.venv/bin/pytest tests/test_data.py -q` → PASS.

- [ ] **Step 5: Smoke-test against real xView3-tiny** (if data present)

Run: `.venv/bin/python -c "from planetar_fusion.data import XView3Dataset as D; d=D('data/xview3-tiny','data/xview3-tiny/labels.csv'); x,t=d[0]; print(x.shape, t['heatmap'].shape, len(t['points']))"`
Expected: `torch.Size([5, 800, 800]) torch.Size([1, 800, 800]) <int>`. *(Skip if Task 0 data not yet local; the unit test covers the target logic.)*

- [ ] **Step 6: Commit** — `git commit -am "stage1: XView3Dataset + gaussian heatmap targets"`

---

### Task 3: the detector model

**Files:** Create `src/planetar_fusion/model.py`, `tests/test_model.py`.

**Interfaces:** Produces `CenterNetSAR(in_ch=5)` whose `forward(x[B,C,H,W]) -> dict{"heatmap":[B,1,H,W] (sigmoid), "length":[B,1,H,W], "is_vessel":[B,1,H,W], "is_fishing":[B,1,H,W]}`.

- [ ] **Step 1: Write the failing test** `tests/test_model.py`

```python
import torch
from planetar_fusion.model import CenterNetSAR

def test_forward_shapes():
    m = CenterNetSAR(in_ch=5)
    out = m(torch.randn(2, 5, 256, 256))
    for k in ("heatmap", "length", "is_vessel", "is_fishing"):
        assert out[k].shape == (2, 1, 256, 256), k
    assert out["heatmap"].min() >= 0 and out["heatmap"].max() <= 1  # sigmoid
```

- [ ] **Step 2: Run, expect failure** — `pytest tests/test_model.py -q` → FAIL.

- [ ] **Step 3: Implement `model.py`** — a ResNet-ish encoder + upsampling decoder (U-Net) at full input resolution, four 1×1 output heads; heatmap through sigmoid.

```python
import torch, torch.nn as nn

def cbr(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

class CenterNetSAR(nn.Module):
    def __init__(self, in_ch=5, base=32):
        super().__init__()
        self.e1 = cbr(in_ch, base);      self.e2 = cbr(base, base*2)
        self.e3 = cbr(base*2, base*4);   self.e4 = cbr(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = cbr(base*8+base*4, base*4); self.d2 = cbr(base*4+base*2, base*2)
        self.d1 = cbr(base*2+base, base)
        self.head_hm = nn.Conv2d(base, 1, 1); self.head_len = nn.Conv2d(base, 1, 1)
        self.head_v = nn.Conv2d(base, 1, 1);  self.head_f = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return {"heatmap": torch.sigmoid(self.head_hm(d1)), "length": self.head_len(d1),
                "is_vessel": self.head_v(d1), "is_fishing": self.head_f(d1)}
```

- [ ] **Step 4: Run, expect pass** — `pytest tests/test_model.py -q` → PASS.
- [ ] **Step 5: Commit** — `git commit -am "stage1: CenterNetSAR U-Net detector"`

---

### Task 4: losses + overfit sanity

**Files:** Create `src/planetar_fusion/losses.py`, `tests/test_losses.py`.

**Interfaces:** Produces `detector_loss(out, target) -> (total, parts_dict)`: penalty-reduced focal on the heatmap; L1 on `length` and BCE on `is_vessel`/`is_fishing` **sampled at GT points only**.

- [ ] **Step 1: Write the failing test — the canonical ML sanity check: the model can overfit one batch**

```python
import torch
from planetar_fusion.model import CenterNetSAR
from planetar_fusion.losses import detector_loss
from planetar_fusion.data import gaussian_splat

def test_overfits_one_batch():
    torch.manual_seed(0)
    x = torch.randn(1, 5, 128, 128)
    pts = torch.tensor([[30, 40], [90, 100]])
    target = {"heatmap": gaussian_splat((0,0,128,128), pts).unsqueeze(0),
              "points": [pts], "length": [torch.tensor([120., 80.])],
              "is_vessel": [torch.tensor([1., 1.])], "is_fishing": [torch.tensor([0., 1.])]}
    m = CenterNetSAR(5); opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = None
    for _ in range(60):
        opt.zero_grad(); out = m(x); loss, _ = detector_loss(out, target); loss.backward(); opt.step()
        first = first or loss.item()
    assert loss.item() < 0.5 * first  # loss at least halves — model+loss+optimizer learn
```

- [ ] **Step 2: Run, expect failure** — `pytest tests/test_losses.py -q` → FAIL.

- [ ] **Step 3: Implement `losses.py`**

```python
import torch, torch.nn.functional as F

def focal_heatmap(pred, gt, a=2.0, b=4.0, eps=1e-6):
    pred = pred.clamp(eps, 1 - eps)
    pos = (gt == 1).float(); neg = 1 - pos
    pos_loss = -((1 - pred) ** a) * torch.log(pred) * pos
    neg_loss = -((1 - gt) ** b) * (pred ** a) * torch.log(1 - pred) * neg
    n = pos.sum().clamp(min=1)
    return (pos_loss.sum() + neg_loss.sum()) / n

def _at_points(head, pts):  # head [1,H,W] -> values at pts[N,2]
    return head[0, pts[:, 0], pts[:, 1]] if len(pts) else head.new_zeros(0)

def detector_loss(out, target, w_len=0.1):
    hm = focal_heatmap(out["heatmap"], target["heatmap"])
    lens, vs, fs = [], [], []
    for b, pts in enumerate(target["points"]):
        if not len(pts): continue
        lp = _at_points(out["length"][b], pts)
        lens.append(F.l1_loss(lp, torch.log1p(target["length"][b])))
        vs.append(F.binary_cross_entropy_with_logits(_at_points(out["is_vessel"][b], pts), target["is_vessel"][b]))
        fs.append(F.binary_cross_entropy_with_logits(_at_points(out["is_fishing"][b], pts), target["is_fishing"][b]))
    def m(xs): return torch.stack(xs).mean() if xs else torch.tensor(0.0)
    total = hm + w_len * m(lens) + m(vs) + m(fs)
    return total, {"hm": hm.item(), "len": m(lens).item(), "v": m(vs).item(), "f": m(fs).item()}
```
*(Note the target shape contract: `train.py` (Task 5) passes `points`/`length`/`is_vessel`/`is_fishing` as per-sample lists and `heatmap` as `[B,1,H,W]`; the dataset's per-item tensors are collated into these lists by a custom `collate_fn` written in Task 5.)*

- [ ] **Step 4: Run, expect pass** — `pytest tests/test_losses.py -q` → PASS (loss at least halves).
- [ ] **Step 5: Commit** — `git commit -am "stage1: losses + overfit-one-batch sanity"`

---

### Task 5: training loop

**Files:** Create `src/planetar_fusion/train.py`, `configs/stage1.yaml`.

**Interfaces:** `train.py --config configs/stage1.yaml` trains, writes `checkpoints/stage1_ep*.pt` and a JSON metrics log to `runs/`. Defines `collate_fn` producing the batched-target contract Task 4 expects.

- [ ] **Step 1: Write `configs/stage1.yaml`**

```yaml
data: {root: data/xview3, train_labels: data/xview3/train.csv, val_labels: data/xview3/validation.csv,
       channels: [VH_dB, VV_dB, bathymetry, owiMask, owiWindSpeed], tile: 800, stride: 704}
train: {epochs: 20, batch_size: 8, lr: 0.0003, num_workers: 8, device: cuda, amp: true}
out: {ckpt_dir: checkpoints, run_dir: runs}
```

- [ ] **Step 2: Implement `train.py`** — a standard loop: `XView3Dataset` → `DataLoader(collate_fn=...)` → `CenterNetSAR` → `detector_loss` → Adam + AMP; per-epoch checkpoint; append `{epoch, train_loss, parts}` to `runs/metrics.jsonl`. `collate_fn` stacks images to `[B,C,H,W]`, stacks `heatmap` to `[B,1,H,W]`, and keeps `points`/`length`/`is_vessel`/`is_fishing` as length-B lists.

```python
import argparse, json, yaml, torch
from pathlib import Path
from torch.utils.data import DataLoader
from planetar_fusion.data import XView3Dataset
from planetar_fusion.model import CenterNetSAR
from planetar_fusion.losses import detector_loss

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    hm = torch.stack([b[1]["heatmap"] for b in batch])
    keys = ("points", "length", "is_vessel", "is_fishing")
    tgt = {"heatmap": hm, **{k: [b[1][k] for b in batch] for k in keys}}
    return imgs, tgt

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(open(ap.parse_args().config))
    dev = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"
    ds = XView3Dataset(cfg["data"]["root"], cfg["data"]["train_labels"],
                       cfg["data"]["channels"], cfg["data"]["tile"], cfg["data"]["stride"])
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], collate_fn=collate)
    m = CenterNetSAR(len(cfg["data"]["channels"])).to(dev)
    opt = torch.optim.Adam(m.parameters(), cfg["train"]["lr"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and dev == "cuda")
    Path(cfg["out"]["ckpt_dir"]).mkdir(exist_ok=True); Path(cfg["out"]["run_dir"]).mkdir(exist_ok=True)
    log = open(Path(cfg["out"]["run_dir"]) / "metrics.jsonl", "a")
    for ep in range(cfg["train"]["epochs"]):
        m.train(); tot = 0.0
        for imgs, tgt in dl:
            imgs = imgs.to(dev); tgt["heatmap"] = tgt["heatmap"].to(dev)
            for k in ("length","is_vessel","is_fishing"): tgt[k] = [t.to(dev) for t in tgt[k]]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss, parts = detector_loss(m(imgs), tgt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        log.write(json.dumps({"epoch": ep, "train_loss": tot/len(dl), "parts": parts}) + "\n"); log.flush()
        torch.save(m.state_dict(), Path(cfg["out"]["ckpt_dir"]) / f"stage1_ep{ep}.pt")

if __name__ == "__main__": main()
```

- [ ] **Step 3: Verify a short local run on xView3-tiny** — set `epochs: 1, device: cpu, num_workers: 0` and point paths at the tiny set. Run `python -m planetar_fusion.train --config configs/stage1_tiny.yaml`. Expected: completes, writes a checkpoint + a metrics line; `train_loss` finite. *(If Task 0 data absent, defer to Task 8 on the pod.)*
- [ ] **Step 4: Commit** — `git commit -am "stage1: training loop + config + collate"`

---

### Task 6: full-scene inference + tile-boundary NMS

**Files:** Create `src/planetar_fusion/infer.py`.

**Interfaces:** `infer.py --ckpt <pt> --scene <dir> --out preds.csv` → writes xView3-format predictions (`scene_id, detect_scene_row, detect_scene_column, is_vessel, is_fishing, vessel_length_m, score`).

- [ ] **Step 1: Implement `infer.py`** — tile the scene (same grid as `data.py`), run the model per tile, extract heatmap peaks (`score = local-max value > thresh`, 3×3 max-pool peak test), read `length`/`is_vessel`/`is_fishing` at each peak, map tile→scene coords, then **global NMS by pixel distance** (suppress the lower-score detection within `min_dist` px). Emit the CSV.

```python
# key pieces (peak extraction + NMS); wire into an argparse main mirroring train.py's config reads
import torch, torch.nn.functional as F, numpy as np

def peaks(hm, thresh=0.3, k=3):
    m = (F.max_pool2d(hm, k, 1, k//2) == hm) & (hm > thresh)
    ys, xs = torch.where(m[0])
    return torch.stack([ys, xs], 1), hm[0, ys, xs]

def nms(points, scores, min_dist=10):
    keep, order = [], scores.argsort(descending=True)
    pts = points[order]
    taken = torch.zeros(len(pts), dtype=torch.bool)
    for i in range(len(pts)):
        if taken[i]: continue
        keep.append(order[i].item())
        d = ((pts[i+1:] - pts[i]) ** 2).sum(1).sqrt()
        taken[i+1:] |= (d < min_dist)
    return keep
```

- [ ] **Step 2: Verify on one tiny scene** — `python -m planetar_fusion.infer --ckpt checkpoints/<tiny>.pt --scene data/xview3-tiny/<scene> --out /tmp/preds.csv`. Expected: a non-empty, well-formed CSV with the required columns; coords within scene bounds.
- [ ] **Step 3: Commit** — `git commit -am "stage1: full-scene inference + tile-boundary NMS"`

---

### Task 7: eval — the xView3 MR metric

**Files:** Create `src/planetar_fusion/eval.py`, `tests/test_eval.py`; vendor `vendor/xview3-reference/`.

**Interfaces:** `eval.py --preds preds.csv --gt data/xview3/validation.csv --shore-root data/xview3` → prints/writes `{aggregate MR, F1_D, F1_S, is_vessel F1, is_fishing F1, length_err}` using the reference scorer.

- [ ] **Step 1: Vendor the reference scorer** — `git submodule add https://github.com/DIUx-xView/xview3-reference vendor/xview3-reference` (or copy `metric.py`); it is MIT/Apache. Note its exact `score(...)` signature from its README.
- [ ] **Step 2: Write the failing test** `tests/test_eval.py` — construct a tiny synthetic preds/gt pair where predictions exactly equal ground truth and assert the scorer returns `F1_D == 1.0` (perfect detection) and an `MR` in `[0,1]`.
- [ ] **Step 3: Run, expect failure** (no `eval.py`).
- [ ] **Step 4: Implement `eval.py`** — thin wrapper importing the vendored scorer, adapting our CSV columns to its expected schema, returning the metric dict.
- [ ] **Step 5: Run, expect pass** — perfect preds → `F1_D == 1.0`.
- [ ] **Step 6: Commit** — `git commit -am "stage1: MR eval via vendored xview3 scorer"`

---

### Task 8: RunPod training run — the milestone deliverable

**Files:** Create `scripts/runpod_setup.sh`.

**Interfaces:** Produces a trained checkpoint + a real `MR` score on the validation split.

- [ ] **Step 1: Write `scripts/runpod_setup.sh`** — per the founder's existing RunPod workflow: clone the repo on the pod, `pip install -e ".[dev]"` with CUDA torch, stage `data/xview3` from our mirror to pod storage, verify `nvidia-smi` + `torch.cuda.is_available()`.
- [ ] **Step 2: Train on the pod** — `python -m planetar_fusion.train --config configs/stage1.yaml` on the **validation split + a train subset** (start small; scale scenes up if the GPU/time budget allows). Watch `runs/metrics.jsonl` for a decreasing `train_loss`.
- [ ] **Step 3: Infer + score** — run `infer.py` over the held-back validation scenes, then `eval.py` → record the `MR` + components.
- [ ] **Step 4: Pull artifacts down** — checkpoint + `runs/metrics.jsonl` + the score back to bb; write the number into `docs/DESIGN.md` (a results note) and compare to the published reference/leaderboard.
- [ ] **Step 5: Commit** — `git commit -am "stage1: first RunPod training run — MR=<score> vs reference"`

---

### Task 9: planetar-sat integration (learned detector backend)

**Files:** Modify `~/github/planetarx/planetar-sat` (add a `learned` detector backend).

**Interfaces:** `planetar-sat detect --detector learned --ckpt <pt>` emits the same `sar.chip` / detection bus messages as the CFAR path.

- [ ] **Step 1: Locate the CFAR detector seam** in `planetar-sat` (the `detect` command's detector call) and its detection→bus-message shape. `grep -rn "cfar\|def detect\|sar.chip" ~/github/planetarx/planetar-sat/src`.
- [ ] **Step 2: Add a `learned` backend** that loads a `planetar-fusion` checkpoint and runs `infer` on a scene, mapping detections to the existing detection/message schema (reuse `planetar-fusion.infer`).
- [ ] **Step 3: Verify on a cached Sentinel-1 scene** — `planetar-sat replay <scene> --detector learned` emits detections onto the bus (or to stdout); compare count/plausibility against the CFAR path on the same scene.
- [ ] **Step 4: Commit** (in `planetar-sat`) — `git commit -am "detect: learned (planetar-fusion) detector backend"`

---

## Self-Review

**Spec coverage** (DESIGN §3 → tasks): data/xView3 → Task 0+2; tiling+channels → Task 2; model (heatmap + heads) → Task 3; losses → Task 4; training → Task 5; inference+NMS-stitch → Task 6; **MR benchmark** → Task 7; **RunPod run + real score** → Task 8; **planetar-sat integration payoff** → Task 9. "Dark = detection − AIS" honored (no dark head anywhere). Stage 2 correctly out of scope.

**Placeholder scan:** no TBD/TODO in deliverable content. Code steps show real code for the load-bearing pieces (targets, model, loss, peak/NMS); boilerplate (infer main, eval wrapper, runpod script) is specified precisely because it depends on the vendored scorer's exact signature and the founder's RunPod workflow — resolved at build time, flagged as such.

**Type/name consistency:** the target-dict contract (`heatmap [B,1,H,W]` + per-sample lists `points`/`length`/`is_vessel`/`is_fishing`) is defined in Task 2, consumed identically in Task 4's loss and produced by Task 5's `collate`. `CenterNetSAR` output keys (`heatmap`/`length`/`is_vessel`/`is_fishing`) match across Tasks 3/4/5/6. Channel list (5 ch) consistent across data/config/model.

**Scope:** single subsystem (the Stage-1 detector), one plan, ends with a benchmarked model integrated into the platform. Stage 2 is its own future plan.
