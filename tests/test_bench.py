"""Task 8 prep — scene-level split of the validation labels, and the holdout benchmark."""
import json

import pandas as pd
import yaml

from conftest import write_scene
from planetar_fusion.bench import benchmark, split_scenes
from planetar_fusion.eval import score_predictions
from planetar_fusion.train import train


def _labels(n_scenes, per_scene=3):
    return pd.DataFrame([{"scene_id": f"s{i:02d}", "detect_scene_row": 10 * j,
                          "detect_scene_column": 10 * j, "confidence": "HIGH"}
                         for i in range(n_scenes) for j in range(per_scene)])


def test_split_is_by_scene_disjoint_and_deterministic(tmp_path):
    src = tmp_path / "validation.csv"
    _labels(10).to_csv(src, index=False)
    a = split_scenes(src, tmp_path / "a", n_holdout=3, seed=7)
    b = split_scenes(src, tmp_path / "b", n_holdout=3, seed=7)
    tr, ho = pd.read_csv(a["train"]), pd.read_csv(a["holdout"])
    assert set(tr.scene_id).isdisjoint(ho.scene_id)
    assert ho.scene_id.nunique() == 3 and tr.scene_id.nunique() == 7
    assert len(tr) + len(ho) == 30                       # every label row lands somewhere
    assert a["holdout_scenes"] == b["holdout_scenes"]    # same seed -> same split


def test_scene_with_no_predictions_still_counts_against_recall(tmp_path):
    gt = pd.DataFrame([["s1", 100, 100, True, False, 50.0, "HIGH", 30.0],
                       ["s2", 100, 100, True, False, 50.0, "HIGH", 30.0]],
                      columns=["scene_id", "detect_scene_row", "detect_scene_column", "is_vessel",
                               "is_fishing", "vessel_length_m", "confidence", "distance_from_shore_km"])
    preds = pd.DataFrame([["s1", 100, 100, True, False, 50.0, 0.9]],
                         columns=["scene_id", "detect_scene_row", "detect_scene_column",
                                  "is_vessel", "is_fishing", "vessel_length_m", "score"])
    gt.to_csv(tmp_path / "gt.csv", index=False)
    preds.to_csv(tmp_path / "preds.csv", index=False)
    only_pred_scenes = score_predictions(tmp_path / "preds.csv", tmp_path / "gt.csv")
    all_scenes = score_predictions(tmp_path / "preds.csv", tmp_path / "gt.csv", scene_ids=["s1", "s2"])
    assert only_pred_scenes["loc_fscore"] == 1.0         # the blind spot: s2 silently dropped
    assert all_scenes["loc_fscore"] < 1.0                # s2's missed vessel is now a false negative


def test_benchmark_infers_every_holdout_scene_and_writes_score(tmp_path):
    root = tmp_path / "xv"
    _, labels = write_scene(root)
    cfg = {"data": {"root": str(root), "train_labels": str(labels), "val_labels": str(labels),
                    "channels": ["VH_dB", "VV_dB", "bathymetry", "owiMask", "owiWindSpeed"],
                    "tile": 64, "stride": 32},
           "model": {"base": 8},
           "train": {"epochs": 1, "batch_size": 2, "lr": 1e-3, "num_workers": 0, "device": "cpu", "amp": False},
           "out": {"ckpt_dir": str(tmp_path / "ckpt"), "run_dir": str(tmp_path / "runs")}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(cfg))
    ckpt = train(tmp_path / "cfg.yaml")["last_ckpt"]
    out = benchmark(ckpt, root, labels, tmp_path / "bench", device="cpu")
    score = json.loads((tmp_path / "bench" / "score.json").read_text())
    assert set(score) >= {"aggregate", "loc_fscore", "scenes", "ckpt"}
    assert score["scenes"] == ["synthetic_scene_000"]
    assert (tmp_path / "bench" / "preds.csv").exists()
    assert out == score
