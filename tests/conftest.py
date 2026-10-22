"""Shared fixtures.

`tiny_xview3` writes a synthetic scene in the xView3 on-disk layout
(<root>/<scene_id>/<channel>.tif + a labels CSV with the xView3 columns) so
the dataset, training loop and inference can be exercised locally without
the credentialed 123 GB download (Task 0). Bright blobs are painted at the
labelled points so an overfit run has something to find.
"""
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

CHANNELS = ["VH_dB", "VV_dB", "bathymetry", "owiMask", "owiWindSpeed"]
SCENE = "synthetic_scene_000"
# (row, col, is_vessel, is_fishing, length_m) — spread so each 64-px tile gets some
POINTS = [(10, 12, 1, 0, 120.0), (40, 70, 1, 1, 35.0), (80, 20, 0, 0, 0.0), (85, 88, 1, 0, 60.0)]


NODATA = -32768.0


def write_scene(root, scene=SCENE, size=96, points=POINTS, seed=0, anc_factor=1, nodata_cols=0):
    """Real xView3 scenes put SAR on a 10 m grid and the ancillary rasters on a 500 m grid, and
    mark missing pixels with -32768 (not NaN). `anc_factor` coarsens the ancillary grid and
    `nodata_cols` blanks a left-edge strip of the SAR channels, to reproduce both."""
    rng = np.random.default_rng(seed)
    d = root / scene
    d.mkdir(parents=True)
    transform = from_origin(-123.5, 48.5, 1e-4, 1e-4)
    yy, xx = np.mgrid[0:size, 0:size]
    n = size // anc_factor
    anc_transform = from_origin(-123.5, 48.5, 1e-4 * anc_factor, 1e-4 * anc_factor)
    for ch in CHANNELS:
        sar = ch in ("VH_dB", "VV_dB")
        if sar:
            a = rng.normal(-25.0, 2.0, (size, size)).astype("float32")
            for r, c, *_ in points:  # a bright target ~ +20 dB over sea clutter
                a += 20.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 1.5**2))
            a[:, :nodata_cols] = NODATA
        elif ch == "bathymetry":  # deepens left to right, so resampling is observable
            a = np.tile(np.linspace(-10.0, -100.0, n, dtype="float32"), (n, 1))
        elif ch == "owiMask":
            a = np.zeros((n, n), "float32")
        else:
            a = np.full((n, n), 6.0, "float32")
        h, w = a.shape
        with rasterio.open(d / f"{ch}.tif", "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", crs="EPSG:4326", nodata=NODATA,
                           transform=transform if sar else anc_transform) as ds:
            ds.write(a, 1)
    rows = [{"scene_id": scene, "detect_scene_row": r, "detect_scene_column": c,
             "detect_lat": 48.5 - r * 1e-4, "detect_lon": -123.5 + c * 1e-4,
             "is_vessel": bool(v), "is_fishing": bool(f) if v else None,
             "vessel_length_m": L if v else None, "source": "AIS" if v else "manual",
             "confidence": "HIGH", "distance_from_shore_km": 12.0}
            for r, c, v, f, L in points]
    labels = root / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    return d, labels


@pytest.fixture
def tiny_xview3(tmp_path):
    root = tmp_path / "xview3-tiny"
    scene_dir, labels = write_scene(root)
    return root, labels, scene_dir
