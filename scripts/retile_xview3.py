#!/usr/bin/env python3
"""Re-tile xView3 SAR GeoTIFFs in place: 1-row strips -> 512x512 tiles, uncompressed, lossless.

xView3 ships VH/VV as uncompressed one-row strips ~29k px wide, so every 800x800 training tile
reads 800 full rows (~92 MB per channel). Tiled, a window reads ~2.5 MB. Each file is copied to a
temp file, checked (grid, nodata, dtype, six random 800x800 windows identical), then renamed over
the original. Idempotent. Usage: scripts/retile_xview3.py data/xview3
"""
import os, sys, time
from pathlib import Path
import numpy as np, rasterio, rasterio.shutil
root = Path(sys.argv[1])
files = sorted(p for d in root.iterdir() if d.is_dir() for p in d.glob("V[HV]_dB.tif"))
for i, p in enumerate(files):
    with rasterio.open(p) as ds:
        if ds.block_shapes[0] == (512, 512):
            print(f"[{i+1}/{len(files)}] skip {p}", flush=True); continue
        meta = (ds.shape, ds.transform, ds.crs, ds.nodata, ds.dtypes)
    tmp = p.with_suffix(".tiled.tmp.tif")
    t = time.time()
    rasterio.shutil.copy(p, tmp, driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="none", BIGTIFF="IF_SAFER")
    rng = np.random.default_rng(i)
    with rasterio.open(p) as a, rasterio.open(tmp) as b:
        assert (b.shape, b.transform, b.crs, b.nodata, b.dtypes) == meta, p
        assert b.block_shapes[0] == (512, 512)
        for _ in range(6):
            r, c = int(rng.integers(0, a.height - 800)), int(rng.integers(0, a.width - 800))
            w = rasterio.windows.Window(c, r, 800, 800)
            x, y = a.read(1, window=w), b.read(1, window=w)
            assert np.array_equal(x, y, equal_nan=True), p
    os.replace(tmp, p)
    print(f"[{i+1}/{len(files)}] {p} {time.time()-t:.0f}s", flush=True)
print("done", flush=True)
