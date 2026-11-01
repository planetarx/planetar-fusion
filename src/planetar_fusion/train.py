"""Config-driven training loop for the Stage 1 detector.

    python -m planetar_fusion.train --config configs/stage1.yaml

Writes checkpoints/stage1_ep<N>.pt per epoch and appends one JSON line per
epoch to runs/metrics.jsonl. `train.schedule: cosine` warms the LR up over `warmup_steps`
and cosine-decays it to `min_lr_ratio` × lr; `train.validate: true` scores the loss on the
`val_labels` tiles each epoch and keeps the lowest as stage1_best.pt. `collate` produces the batched target contract
losses.detector_loss expects (heatmap stacked, attributes as per-sample lists).
"""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from planetar_fusion.data import XView3Dataset
from planetar_fusion.losses import detector_loss
from planetar_fusion.model import CenterNetSAR

LIST_KEYS = ("points", "length", "is_vessel", "is_fishing")


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    hm = torch.stack([b[1]["heatmap"] for b in batch])
    return imgs, {"heatmap": hm, **{k: [b[1][k] for b in batch] for k in LIST_KEYS}}


def _finite(t):
    return not t.is_floating_point() or bool(torch.isfinite(t).all())


def train_step(model, opt, scaler, imgs, tgt, use_amp, max_grad_norm=None):
    """One optimiser step → the loss as a float, or None when the batch was skipped.

    A batch whose loss or gradients are non-finite is dropped before it changes anything:
    the weights are untouched and the BatchNorm running stats, which the forward pass
    updates in place, are restored. (GradScaler alone skips the weight update but not the
    running stats; NaN stats are what broke the 2026-10-23 run's last checkpoint.)
    """
    saved = {n: b.clone() for n, b in model.named_buffers()}
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", enabled=use_amp):
        loss, _ = detector_loss(model(imgs), tgt)
    if not torch.isfinite(loss):
        _restore(model, saved)
        return None
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    if not all(_finite(p.grad) for p in model.parameters() if p.grad is not None):
        _restore(model, saved)
        opt.zero_grad(set_to_none=True)
        scaler.update()  # back off the loss scale, as a skipped AMP step would
        return None
    if max_grad_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(opt)
    scaler.update()
    return loss.item()


def _restore(model, saved):
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(saved[n])


def make_optimizer(model, t):
    """AdamW with decoupled weight decay on conv weights only. The convs feed BatchNorm, so
    their scale does not change the loss and plain Adam lets it grow without bound (40-60x
    in norm over three epochs on 2026-10-26) until fp16 activations overflow. BatchNorm
    affine params and biases (all 1-D) are not decayed."""
    decay = [p for p in model.parameters() if p.ndim > 1]
    rest = [p for p in model.parameters() if p.ndim <= 1]
    return torch.optim.AdamW([{"params": decay, "weight_decay": t.get("weight_decay", 0.0)},
                              {"params": rest, "weight_decay": 0.0}], lr=t["lr"])


def lr_factor(step, total, warmup, min_ratio):
    """LR multiplier after `step` optimiser steps: linear warm-up to 1 over `warmup` steps,
    then cosine decay to `min_ratio` at `total`. Run 4's constant LR destabilised late."""
    if step < warmup:
        return (step + 1) / warmup
    p = min(1.0, (step - warmup + 1) / max(1, total - warmup + 1))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))


def make_scheduler(opt, t, total_steps):
    if t.get("schedule", "constant") == "constant":
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    warmup, floor = int(t.get("warmup_steps", 0)), float(t.get("min_lr_ratio", 0.0))
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total_steps, warmup, floor))


def evaluate_loss(model, dl, dev, use_amp):
    """Mean detector loss over `dl` in eval mode (BatchNorm stats untouched); non-finite
    batches are left out. Restores the model's train/eval mode."""
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for imgs, tgt in dl:
            imgs = imgs.to(dev)
            tgt["heatmap"] = tgt["heatmap"].to(dev)
            for k in LIST_KEYS:
                tgt[k] = [x.to(dev) for x in tgt[k]]
            with torch.autocast("cuda", enabled=use_amp):
                loss, _ = detector_loss(model(imgs), tgt)
            if torch.isfinite(loss):
                tot, n = tot + loss.item(), n + 1
    model.train(was_training)
    return tot / max(1, n)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path):
    cfg = load_config(config_path)
    d, t, o = cfg["data"], cfg["train"], cfg["out"]
    dev = t["device"] if torch.cuda.is_available() else "cpu"
    ds = XView3Dataset(d["root"], d["train_labels"], d["channels"], d["tile"], d["stride"])
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True,
                    num_workers=t["num_workers"], collate_fn=collate)
    model = CenterNetSAR(len(d["channels"]), base=cfg.get("model", {}).get("base", 32)).to(dev)
    opt = make_optimizer(model, t)
    sched = make_scheduler(opt, t, t["epochs"] * len(dl))
    val_dl = None
    if t.get("validate"):
        val_ds = XView3Dataset(d["root"], d["val_labels"], d["channels"], d["tile"], d["stride"])
        val_dl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                            num_workers=t["num_workers"], collate_fn=collate)
    use_amp = bool(t.get("amp")) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir, run_dir = Path(o["ckpt_dir"]), Path(o["run_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    last, best, best_val, best_ep = None, None, math.inf, None
    with open(run_dir / "metrics.jsonl", "a") as log:
        for ep in range(t["epochs"]):
            model.train()
            tot, done, skipped = 0.0, 0, 0
            for imgs, tgt in tqdm(dl, desc=f"epoch {ep}", leave=False, disable=len(dl) < 10):
                imgs = imgs.to(dev)
                tgt["heatmap"] = tgt["heatmap"].to(dev)
                for k in LIST_KEYS:
                    tgt[k] = [x.to(dev) for x in tgt[k]]
                loss = train_step(model, opt, scaler, imgs, tgt, use_amp, t.get("max_grad_norm"))
                if loss is None:
                    skipped += 1
                else:
                    tot, done = tot + loss, done + 1
                sched.step()  # keyed on the iteration, so skipped batches still advance it
            bad = [n for n, v in model.state_dict().items() if not _finite(v)]
            if bad:
                raise RuntimeError(f"model went non-finite in epoch {ep} ({len(bad)} tensors, e.g. {bad[0]})")
            rec = {"epoch": ep, "train_loss": tot / max(1, done), "skipped_batches": skipped,
                   "lr": opt.param_groups[0]["lr"]}
            if val_dl is not None:
                rec["val_loss"] = evaluate_loss(model, val_dl, dev, use_amp)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            state = {"model": model.state_dict(), "config": cfg, "epoch": ep}
            last = ckpt_dir / f"stage1_ep{ep}.pt"
            torch.save(state, last)
            if val_dl is not None and rec["val_loss"] < best_val:
                best_val, best_ep, best = rec["val_loss"], ep, ckpt_dir / "stage1_best.pt"
                torch.save({**state, "val_loss": best_val}, best)
    r = {"epochs": t["epochs"], "last_ckpt": str(last), "device": dev}
    if best is not None:
        r.update(best_ckpt=str(best), best_epoch=best_ep)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    r = train(ap.parse_args().config)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
