"""Task 2 — CenterNet heatmap targets and the xView3 dataset."""
import torch

from planetar_fusion.data import gaussian_splat


def test_gaussian_splat_peaks_at_points():
    hm = gaussian_splat((32, 32), torch.tensor([[8, 8], [20, 24]]), sigma=2.0)
    assert hm.shape == (1, 32, 32)
    assert torch.isclose(hm[0, 8, 8], torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(hm[0, 20, 24], torch.tensor(1.0), atol=1e-4)
    assert hm.max() <= 1.0 + 1e-5 and hm.min() >= 0.0


def test_gaussian_splat_empty_points_is_all_zero():
    hm = gaussian_splat((16, 16), torch.zeros(0, 2, dtype=torch.long), sigma=2.0)
    assert hm.shape == (1, 16, 16)
    assert hm.sum() == 0


def test_gaussian_splat_overlap_takes_max_not_sum():
    hm = gaussian_splat((16, 16), torch.tensor([[8, 8], [8, 9]]), sigma=2.0)
    assert hm.max() <= 1.0 + 1e-5  # two adjacent centres must not sum past 1


# --- XView3Dataset on the synthetic fixture (see conftest.py) ---------------
from planetar_fusion.data import XView3Dataset  # noqa: E402


def test_dataset_tiles_the_scene_on_a_stride_grid(tiny_xview3):
    root, labels, _ = tiny_xview3
    ds = XView3Dataset(root, labels, tile=64, stride=32)
    # 96-px scene, 64-px tile, 32-px stride -> origins {0, 32} on each axis
    assert len(ds) == 4
    assert sorted(ds.index) == [("synthetic_scene_000", 0, 0), ("synthetic_scene_000", 0, 32),
                                ("synthetic_scene_000", 32, 0), ("synthetic_scene_000", 32, 32)]


def test_dataset_item_shapes_and_local_points(tiny_xview3):
    root, labels, _ = tiny_xview3
    ds = XView3Dataset(root, labels, tile=64, stride=32)
    img, t = ds[0]  # tile at (0,0): contains points (10,12) and (40,70)? no — col 70 >= 64
    assert img.shape == (5, 64, 64) and img.dtype == torch.float32
    assert t["heatmap"].shape == (1, 64, 64)
    assert t["points"].tolist() == [[10, 12]]
    assert torch.isclose(t["heatmap"][0, 10, 12], torch.tensor(1.0), atol=1e-4)
    assert t["length"].tolist() == [120.0] and t["is_vessel"].tolist() == [1.0] and t["is_fishing"].tolist() == [0.0]


def test_dataset_offsets_points_into_tile_coordinates(tiny_xview3):
    root, labels, _ = tiny_xview3
    ds = XView3Dataset(root, labels, tile=64, stride=32)
    i = ds.index.index(("synthetic_scene_000", 32, 32))
    _, t = ds[i]  # tile (32,32) spans rows/cols 32..95: holds (40,70) and (85,88)
    assert sorted(t["points"].tolist()) == [[8, 38], [53, 56]]


def test_dataset_non_vessel_label_has_zero_length_and_no_fishing(tiny_xview3):
    root, labels, _ = tiny_xview3
    ds = XView3Dataset(root, labels, tile=64, stride=32)
    i = ds.index.index(("synthetic_scene_000", 32, 0))  # holds (80,20) non-vessel
    _, t = ds[i]
    k = t["points"].tolist().index([48, 20])
    assert t["is_vessel"][k] == 0.0 and t["is_fishing"][k] == 0.0 and t["length"][k] == 0.0


def test_read_window_resamples_coarse_ancillary_onto_the_sar_grid(tmp_path):
    from conftest import CHANNELS, write_scene
    from planetar_fusion.data import read_window
    scene_dir, _ = write_scene(tmp_path / "xv", anc_factor=8)   # 96 px SAR, 12 px ancillary
    img = read_window(scene_dir, CHANNELS, 16, 16, 64)
    assert img.shape == (5, 64, 64) and torch.isfinite(img).all()
    bathy = img[CHANNELS.index("bathymetry")]
    # deepens left to right on the coarse grid -> must still decrease across the SAR-grid tile
    assert bathy[:, 4].mean() > bathy[:, 32].mean() > bathy[:, 60].mean()


def test_nodata_is_masked_out_of_normalisation(tmp_path):
    from conftest import CHANNELS, write_scene
    from planetar_fusion.data import read_window
    scene_dir, _ = write_scene(tmp_path / "xv", nodata_cols=40)
    vv = read_window(scene_dir, ["VV_dB"], 0, 0, 96)[0]
    assert torch.isfinite(vv).all()
    assert (vv[:, :40] == 0).all()                                   # missing pixels -> 0 after z-score
    valid = vv[:, 40:]
    assert abs(valid.mean().item()) < 0.1 and abs(valid.std().item() - 1) < 0.1   # z-scored over valid only
    assert vv[85, 88] > 3                                            # the target at (85,88) still stands out


def test_read_window_bounds_a_near_constant_channel(tmp_path):
    """owiMask is almost constant inside a tile, so a z-score turns its rare odd pixels into
    values of ~sqrt(n) (≈90 in real xView3 tiles); that overflowed fp16 training (2026-10-24).
    The normalised input stays within ±Z_CLIP."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    from planetar_fusion.data import Z_CLIP, read_window

    d = tmp_path / "scene"
    d.mkdir()
    a = np.zeros((64, 64), "float32")
    a[5, 5] = 2.0  # one land pixel in a sea tile
    with rasterio.open(d / "owiMask.tif", "w", driver="GTiff", height=64, width=64, count=1,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(0, 0, 1e-4, 1e-4)) as ds:
        ds.write(a, 1)
    x = read_window(d, ["owiMask"], 0, 0, 64)
    assert float(x.abs().max()) <= Z_CLIP
