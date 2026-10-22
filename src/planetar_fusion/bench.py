"""Task 8 helpers: split the validation labels by scene, and score a checkpoint on the holdout.

    python -m planetar_fusion.bench split --labels data/xview3/validation.csv \\
        --out data/xview3/split --holdout 10 --seed 0
    python -m planetar_fusion.bench score --ckpt checkpoints/stage1_ep19.pt \\
        --root data/xview3 --labels data/xview3/split/holdout.csv --out runs/bench-ep19

Task 0 downloads only the validation split (50 scenes), so training and scoring both come
out of it: the split is by scene, never by label row, so no holdout scene is seen in training.
The full xView3 train split is ~10× larger; a run on it just points `train_labels` at it.

`score` infers every holdout scene and scores them together, passing the holdout scene list
to the scorer explicitly: a scene with zero detections must still count its missed vessels.
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

from planetar_fusion.eval import score_predictions
from planetar_fusion.infer import PRED_COLUMNS, infer_scene


def split_scenes(labels_csv, out_dir, n_holdout=10, seed=0):
    labels = pd.read_csv(labels_csv)
    scenes = sorted(labels["scene_id"].unique())
    if not 0 < n_holdout < len(scenes):
        raise ValueError(f"n_holdout={n_holdout} must be in (0, {len(scenes)})")
    holdout = sorted(random.Random(seed).sample(scenes, n_holdout))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_ho = labels["scene_id"].isin(holdout)
    labels[~is_ho].to_csv(out / "train.csv", index=False)
    labels[is_ho].to_csv(out / "holdout.csv", index=False)
    manifest = {"source": str(labels_csv), "seed": seed, "holdout_scenes": holdout,
                "train_scenes": [s for s in scenes if s not in holdout],
                "train": str(out / "train.csv"), "holdout": str(out / "holdout.csv")}
    (out / "split.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def benchmark(ckpt, root, labels_csv, out_dir, thresh=0.3, min_dist=10, device=None, shore_root=None):
    scenes = sorted(pd.read_csv(labels_csv)["scene_id"].unique())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [infer_scene(ckpt, Path(root) / s, thresh=thresh, min_dist=min_dist, device=device)
              for s in scenes]
    preds = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRED_COLUMNS)
    preds.to_csv(out / "preds.csv", index=False)
    score = score_predictions(out / "preds.csv", labels_csv, shore_root=shore_root, scene_ids=scenes)
    score = {k: float(v) for k, v in score.items()}
    score.update(ckpt=str(ckpt), scenes=scenes, n_preds=len(preds), thresh=thresh)
    (out / "score.json").write_text(json.dumps(score, indent=2))
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split")
    sp.add_argument("--labels", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--holdout", type=int, default=10)
    sp.add_argument("--seed", type=int, default=0)
    sc = sub.add_parser("score")
    sc.add_argument("--ckpt", required=True)
    sc.add_argument("--root", required=True)
    sc.add_argument("--labels", required=True, help="holdout labels CSV")
    sc.add_argument("--out", required=True)
    sc.add_argument("--thresh", type=float, default=0.3)
    sc.add_argument("--shore-root", default=None)
    a = ap.parse_args()
    if a.cmd == "split":
        m = split_scenes(a.labels, a.out, a.holdout, a.seed)
        print(f"{len(m['train_scenes'])} train / {len(m['holdout_scenes'])} holdout scenes -> {a.out}")
    else:
        print(json.dumps(benchmark(a.ckpt, a.root, a.labels, a.out, a.thresh, shore_root=a.shore_root),
                         indent=2))


if __name__ == "__main__":
    main()
