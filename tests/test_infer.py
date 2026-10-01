"""Task 6 — full-scene inference: peak extraction, NMS, tile stitching, xView3 CSV."""
import pandas as pd
import torch
import yaml

from planetar_fusion.infer import PRED_COLUMNS, infer_scene, nms, peaks
from planetar_fusion.train import train


def test_peaks_finds_local_maxima_above_threshold():
    hm = torch.zeros(1, 32, 32)
    hm[0, 5, 6] = 0.9
    hm[0, 5, 7] = 0.8   # neighbour of a stronger peak — not a local max
    hm[0, 20, 20] = 0.2  # below threshold
    hm[0, 25, 10] = 0.5
    pts, scores = peaks(hm, thresh=0.3)
    assert sorted(pts.tolist()) == [[5, 6], [25, 10]]
    assert set(round(s, 2) for s in scores.tolist()) == {0.9, 0.5}


def test_nms_keeps_the_stronger_of_two_close_points():
    pts = torch.tensor([[10, 10], [12, 11], [50, 50]])
    scores = torch.tensor([0.6, 0.9, 0.7])
    keep = nms(pts, scores, min_dist=5)
    assert sorted(keep) == [1, 2]


def test_nms_empty_input():
    assert nms(torch.zeros(0, 2, dtype=torch.long), torch.zeros(0)) == []


def test_infer_scene_writes_xview3_csv_and_recovers_the_synthetic_targets(tiny_xview3, tmp_path):
    root, labels, scene_dir = tiny_xview3
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {
        "data": {"root": str(root), "train_labels": str(labels), "val_labels": str(labels),
                 "channels": ["VH_dB", "VV_dB", "bathymetry", "owiMask", "owiWindSpeed"],
                 "tile": 64, "stride": 32},
        "model": {"base": 16},
        "train": {"epochs": 40, "batch_size": 4, "lr": 2e-3, "num_workers": 0, "device": dev, "amp": False},
        "out": {"ckpt_dir": str(tmp_path / "ckpt"), "run_dir": str(tmp_path / "runs")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    ckpt = train(cfg_path)["last_ckpt"]
    out = tmp_path / "preds.csv"
    df = infer_scene(ckpt, scene_dir, out, thresh=0.3, min_dist=4)
    assert list(df.columns) == list(PRED_COLUMNS)
    assert (df.scene_id == "synthetic_scene_000").all()
    assert df.detect_scene_row.between(0, 95).all() and df.detect_scene_column.between(0, 95).all()
    assert pd.read_csv(out).shape == df.shape
    # Overfit on four bright synthetic targets: at least three come back within 3 px.
    gt = [(10, 12), (40, 70), (80, 20), (85, 88)]
    hits = sum(any(abs(r - gr) <= 3 and abs(c - gc) <= 3 for r, c in zip(df.detect_scene_row, df.detect_scene_column))
               for gr, gc in gt)
    assert hits >= 3, f"recovered {hits}/4; preds=\n{df}"
    assert len(df) <= 12, f"too many spurious detections: {len(df)}"
