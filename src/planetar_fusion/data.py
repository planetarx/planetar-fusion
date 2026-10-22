"""xView3 Sentinel-1 dataset: tile scenes, stack channels, build CenterNet targets.

On-disk layout (xView3 as downloaded):
    <root>/<scene_id>/{VH_dB,VV_dB,bathymetry,owiMask,owiWindSpeed,...}.tif
    labels CSV with scene_id, detect_scene_row, detect_scene_column, is_vessel,
    is_fishing, vessel_length_m, source, confidence, distance_from_shore_km.

A sample is one `tile`×`tile` window of the stacked channels plus a target
dict: a Gaussian centre heatmap (CenterNet-style) and per-object attributes
at the labelled points that fall inside the window. "Dark" is not a label
here (DESIGN §3, §6) — at deployment it is detection minus the AIS feed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.windows
import torch
from torch.utils.data import Dataset

DEFAULT_CHANNELS = ["VH_dB", "VV_dB", "bathymetry", "owiMask", "owiWindSpeed"]


def gaussian_splat(shape, points, sigma=2.0):
    """CenterNet target: a [1,H,W] map with a unit-peak Gaussian at each point.

    `shape` is (H, W); `points` is a LongTensor[N,2] of (row, col). Overlapping
    Gaussians combine by max, never sum, so the peak is exactly 1.0 and the
    focal loss's `gt == 1` positive mask stays well defined. Each Gaussian is
    painted only in a ±3σ window, so cost is O(N·σ²), not O(N·H·W).
    """
    h, w = int(shape[0]), int(shape[1])
    hm = torch.zeros(1, h, w)
    if points is None or len(points) == 0:
        return hm
    rad = max(1, int(3 * sigma))
    for r, c in points.tolist():
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue
        ys = torch.arange(r0, r1).view(-1, 1)
        xs = torch.arange(c0, c1).view(1, -1)
        g = torch.exp(-(((ys - r) ** 2 + (xs - c) ** 2) / (2 * sigma**2)))
        hm[0, r0:r1, c0:c1] = torch.maximum(hm[0, r0:r1, c0:c1], g)
    return hm


def tile_origins(size, tile, stride):
    """Origins of `tile`-wide windows on a `stride` grid covering [0, size),
    plus a final window flush with the far edge so no strip is left uncovered
    (the dataset and full-scene inference share this grid)."""
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


def read_window(scene_dir, channels, r0, c0, tile):
    """Stacked, per-channel z-scored window → FloatTensor[C,tile,tile].
    Shared by XView3Dataset and infer.infer_scene so both see identical input.

    The window is defined on the grid of `channels[0]`. Real xView3 scenes put SAR on a 10 m
    grid and the ancillary rasters on a 500 m one, so any channel on a different grid is read
    by the window's geographic bounds and resampled (bilinear) to tile×tile. Missing pixels
    (the raster's nodata, -32768 in xView3, or NaN) are excluded from the z-score and set to 0
    after it, so a scene edge cannot swamp the normalisation."""
    scene_dir = Path(scene_dir)
    with rasterio.open(scene_dir / f"{channels[0]}.tif") as ref:
        win = rasterio.windows.Window(c0, r0, tile, tile)
        bounds = rasterio.windows.bounds(win, ref.transform)
        ref_grid = (ref.transform, ref.width, ref.height)
    chans = [_read(scene_dir / f"{ch}.tif", win, bounds, ref_grid, tile) for ch in channels]
    out = []
    for a in chans:
        valid = np.isfinite(a)
        if valid.any():
            v = a[valid]
            a = np.where(valid, (a - v.mean()) / (v.std() + 1e-6), 0.0)
        else:
            a = np.zeros_like(a)
        out.append(a.astype("float32"))
    return torch.from_numpy(np.stack(out))


def _read(path, window, bounds, ref_grid, tile):
    """One channel's window as float32 [tile,tile], NaN where missing."""
    with rasterio.open(path) as ds:
        if (ds.transform, ds.width, ds.height) == ref_grid:
            a = ds.read(1, window=window, boundless=True, masked=True)
        else:
            w = rasterio.windows.from_bounds(*bounds, transform=ds.transform)
            a = ds.read(1, window=w, out_shape=(tile, tile), boundless=True, masked=True,
                        resampling=rasterio.enums.Resampling.bilinear)
    return np.ma.filled(a.astype("float32"), np.nan)


def _f(x, default=0.0):
    """Float with NaN/None → default (xView3 leaves non-vessel attributes blank)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(v) else v


class XView3Dataset(Dataset):
    """Tiles every scene named in `labels_csv` on a `stride` grid.

    Index entries are (scene_id, row0, col0). Per-tile normalisation is a
    per-channel z-score, which keeps dB and metre-valued ancillary rasters
    on one scale without a global statistics pass.
    """

    def __init__(self, root, labels_csv, channels=None, tile=800, stride=704, sigma=2.0):
        self.root = Path(root)
        self.tile, self.stride, self.sigma = int(tile), int(stride), float(sigma)
        self.channels = list(channels or DEFAULT_CHANNELS)
        self.labels = pd.read_csv(labels_csv)
        self.index = []
        for scene_id in self.labels["scene_id"].unique():
            with rasterio.open(self.root / scene_id / f"{self.channels[0]}.tif") as ds:
                H, W = ds.height, ds.width
            for r0 in tile_origins(H, self.tile, self.stride):
                for c0 in tile_origins(W, self.tile, self.stride):
                    self.index.append((scene_id, r0, c0))

    def __len__(self):
        return len(self.index)

    def read_tile(self, scene_id, r0, c0):
        """Stacked, z-scored channels for one window → FloatTensor[C,tile,tile]."""
        return read_window(self.root / scene_id, self.channels, r0, c0, self.tile)

    def targets(self, scene_id, r0, c0):
        d = self.labels[self.labels.scene_id == scene_id]
        pts, length, isv, isf = [], [], [], []
        for row in d.itertuples(index=False):
            r = int(row.detect_scene_row) - r0
            c = int(row.detect_scene_column) - c0
            if 0 <= r < self.tile and 0 <= c < self.tile:
                vessel = bool(_f(getattr(row, "is_vessel", True), 0.0))
                pts.append([r, c])
                length.append(_f(getattr(row, "vessel_length_m", 0.0)) if vessel else 0.0)
                isv.append(1.0 if vessel else 0.0)
                isf.append(1.0 if vessel and bool(_f(getattr(row, "is_fishing", 0.0), 0.0)) else 0.0)
        pts_t = torch.tensor(pts, dtype=torch.long) if pts else torch.zeros(0, 2, dtype=torch.long)
        return {
            "heatmap": gaussian_splat((self.tile, self.tile), pts_t, self.sigma),
            "points": pts_t,
            "length": torch.tensor(length, dtype=torch.float32),
            "is_vessel": torch.tensor(isv, dtype=torch.float32),
            "is_fishing": torch.tensor(isf, dtype=torch.float32),
        }

    def __getitem__(self, i):
        scene_id, r0, c0 = self.index[i]
        return self.read_tile(scene_id, r0, c0), self.targets(scene_id, r0, c0)
