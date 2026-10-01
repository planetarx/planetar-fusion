"""Config-driven training loop for the Stage 1 detector.

    python -m planetar_fusion.train --config configs/stage1.yaml

Writes checkpoints/stage1_ep<N>.pt per epoch and appends one JSON line per
epoch to runs/metrics.jsonl. `collate` produces the batched target contract
losses.detector_loss expects (heatmap stacked, attributes as per-sample lists).
"""
import argparse
import json
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
    opt = torch.optim.Adam(model.parameters(), t["lr"])
    use_amp = bool(t.get("amp")) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt_dir, run_dir = Path(o["ckpt_dir"]), Path(o["run_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    last = None
    with open(run_dir / "metrics.jsonl", "a") as log:
        for ep in range(t["epochs"]):
            model.train()
            tot, parts = 0.0, {}
            for imgs, tgt in tqdm(dl, desc=f"epoch {ep}", leave=False, disable=len(dl) < 10):
                imgs = imgs.to(dev)
                tgt["heatmap"] = tgt["heatmap"].to(dev)
                for k in LIST_KEYS:
                    tgt[k] = [x.to(dev) for x in tgt[k]]
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", enabled=use_amp):
                    loss, parts = detector_loss(model(imgs), tgt)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot += loss.item()
            rec = {"epoch": ep, "train_loss": tot / max(1, len(dl)), "parts": parts}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            last = ckpt_dir / f"stage1_ep{ep}.pt"
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, last)
    return {"epochs": t["epochs"], "last_ckpt": str(last), "device": dev}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    r = train(ap.parse_args().config)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
