"""Task 5 — the training loop runs end to end on the synthetic scene."""
import json

import torch
import yaml

from planetar_fusion.train import collate, train


def test_collate_batches_heatmaps_and_keeps_per_sample_lists():
    pts = torch.tensor([[1, 2]])
    item = (torch.zeros(5, 8, 8), {"heatmap": torch.zeros(1, 8, 8), "points": pts,
                                    "length": torch.tensor([10.0]), "is_vessel": torch.tensor([1.0]),
                                    "is_fishing": torch.tensor([0.0])})
    imgs, tgt = collate([item, item])
    assert imgs.shape == (2, 5, 8, 8) and tgt["heatmap"].shape == (2, 1, 8, 8)
    assert len(tgt["points"]) == 2 and tgt["points"][1].tolist() == [[1, 2]]


def test_one_epoch_writes_checkpoint_and_metrics(tiny_xview3, tmp_path):
    root, labels, _ = tiny_xview3
    cfg = {
        "data": {"root": str(root), "train_labels": str(labels), "val_labels": str(labels),
                 "channels": ["VH_dB", "VV_dB", "bathymetry", "owiMask", "owiWindSpeed"],
                 "tile": 64, "stride": 32},
        "model": {"base": 8},
        "train": {"epochs": 2, "batch_size": 2, "lr": 1e-3, "num_workers": 0, "device": "cpu", "amp": False},
        "out": {"ckpt_dir": str(tmp_path / "ckpt"), "run_dir": str(tmp_path / "runs")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    result = train(cfg_path)
    assert (tmp_path / "ckpt" / "stage1_ep1.pt").exists()
    lines = [json.loads(l) for l in (tmp_path / "runs" / "metrics.jsonl").read_text().splitlines()]
    assert [l["epoch"] for l in lines] == [0, 1]
    assert all(torch.isfinite(torch.tensor(l["train_loss"])) for l in lines)
    assert result["epochs"] == 2 and result["last_ckpt"].endswith("stage1_ep1.pt")
