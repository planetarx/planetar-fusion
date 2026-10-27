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


def _step_inputs(n=1, size=32):
    from planetar_fusion.data import gaussian_splat
    pts = torch.tensor([[8, 8]])
    imgs = torch.randn(n, 5, size, size)
    tgt = {"heatmap": torch.stack([gaussian_splat((size, size), pts)] * n),
           "points": [pts] * n, "length": [torch.tensor([50.0])] * n,
           "is_vessel": [torch.tensor([1.0])] * n, "is_fishing": [torch.tensor([0.0])] * n}
    return imgs, tgt


def _state(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def test_train_step_skips_a_non_finite_batch_without_touching_the_model():
    """A NaN batch must not reach the weights or the BatchNorm running stats (the epoch-3
    checkpoint of the 2026-10-23 run had NaN running stats and predicted nothing)."""
    from planetar_fusion.model import CenterNetSAR
    from planetar_fusion.train import train_step

    torch.manual_seed(0)
    model = CenterNetSAR(5, base=8)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    imgs, tgt = _step_inputs()
    imgs[0, 0, 0, 0] = float("nan")
    before = _state(model)
    assert train_step(model, opt, scaler, imgs, tgt, use_amp=False) is None
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_train_step_clips_the_gradient_norm():
    from planetar_fusion.model import CenterNetSAR
    from planetar_fusion.train import train_step

    torch.manual_seed(0)
    model = CenterNetSAR(5, base=8)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_step(model, opt, scaler, *_step_inputs(), use_amp=False, max_grad_norm=1e-3)
    assert loss is not None
    moved = torch.sqrt(sum(((p.detach() - b) ** 2).sum() for p, b in zip(model.parameters(), before)))
    assert float(moved) <= 1e-3 * 1.01  # SGD lr=1: the step is the clipped gradient


def test_train_stops_when_the_model_goes_non_finite(tiny_xview3, tmp_path, monkeypatch):
    """Fail fast: an epoch that ends with a non-finite model raises instead of training on."""
    import pytest

    from planetar_fusion import train as train_mod

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

    real_step = train_mod.train_step

    def poison(model, *a, **k):
        out = real_step(model, *a, **k)
        next(b for n, b in model.named_buffers() if n.endswith("running_mean")).fill_(float("nan"))
        return out

    monkeypatch.setattr(train_mod, "train_step", poison)
    with pytest.raises(RuntimeError, match="non-finite"):
        train_mod.train(cfg_path)
    assert not (tmp_path / "ckpt" / "stage1_ep0.pt").exists()


def test_optimizer_decays_conv_weights_but_not_batchnorm_or_biases():
    """Convs that feed a BatchNorm are scale-invariant, so plain Adam grows them without
    bound (40-60x in norm over 3 epochs of the 2026-10-26 run) until fp16 overflows.
    Decoupled weight decay on conv weights holds their scale; BN affine params and biases
    are left undecayed, as usual."""
    from planetar_fusion.model import CenterNetSAR
    from planetar_fusion.train import make_optimizer

    model = CenterNetSAR(5, base=8)
    opt = make_optimizer(model, {"lr": 1e-3, "weight_decay": 0.05})
    assert isinstance(opt, torch.optim.AdamW)
    decay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.05 for p in g["params"]}
    names = {id(p): n for n, p in model.named_parameters()}
    assert names[id(model.d3[0].weight)] and id(model.d3[0].weight) in decay
    assert id(model.d3[1].weight) not in decay  # BatchNorm gamma
    assert id(model.head_hm.bias) not in decay
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))
