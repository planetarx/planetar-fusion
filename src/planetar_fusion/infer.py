"""Full-scene inference: tile → detect → stitch with global NMS → xView3 CSV.

    python -m planetar_fusion.infer --ckpt checkpoints/stage1_ep19.pt \\
        --scene data/xview3/<scene_id> --out preds.csv

Tiles overlap (tile - stride px), so a vessel on a seam is detected twice;
a global NMS by pixel distance keeps the higher-scoring copy. Attribute
heads are read at the heatmap peak: is_vessel / is_fishing are sigmoid >
0.5, length is expm1 of the log-length head (metres).
"""
import argparse
from pathlib import Path

import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from planetar_fusion.data import DEFAULT_CHANNELS, read_window, tile_origins
from planetar_fusion.model import CenterNetSAR

PRED_COLUMNS = ("scene_id", "detect_scene_row", "detect_scene_column",
                "is_vessel", "is_fishing", "vessel_length_m", "score")


def peaks(hm, thresh=0.3, k=3):
    """Local maxima of a [1,H,W] heatmap above `thresh` (k×k max-pool test).
    Returns (LongTensor[N,2] of (row, col), FloatTensor[N] scores)."""
    pooled = F.max_pool2d(hm.unsqueeze(0), k, stride=1, padding=k // 2)[0]
    keep = (pooled == hm) & (hm > thresh)
    ys, xs = torch.where(keep[0])
    return torch.stack([ys, xs], 1), hm[0, ys, xs]


def nms(points, scores, min_dist=10):
    """Greedy distance NMS. Returns the indices (into `points`) to keep,
    highest score first; a point within `min_dist` px of a kept point is dropped."""
    if len(points) == 0:
        return []
    order = scores.argsort(descending=True)
    pts = points[order].float()
    taken = torch.zeros(len(pts), dtype=torch.bool)
    keep = []
    for i in range(len(pts)):
        if taken[i]:
            continue
        keep.append(int(order[i]))
        if i + 1 < len(pts):
            d = ((pts[i + 1:] - pts[i]) ** 2).sum(1).sqrt()
            taken[i + 1:] |= d < min_dist
    return keep


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    channels = cfg.get("data", {}).get("channels", list(DEFAULT_CHANNELS))
    base = cfg.get("model", {}).get("base", 32)
    model = CenterNetSAR(len(channels), base=base).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, channels


@torch.no_grad()
def infer_scene(ckpt_path, scene_dir, out_csv=None, thresh=0.3, min_dist=10,
                device=None, tile=None, stride=None):
    """Detect every object in one scene directory. Returns the predictions
    DataFrame (xView3 columns + score); writes it to `out_csv` if given."""
    scene_dir = Path(scene_dir)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, channels = load_model(ckpt_path, device)
    tile = tile or cfg.get("data", {}).get("tile", 800)
    stride = stride or cfg.get("data", {}).get("stride", 704)
    with rasterio.open(scene_dir / f"{channels[0]}.tif") as ds:
        H, W = ds.height, ds.width

    rows = []
    for r0 in tile_origins(H, tile, stride):
        for c0 in tile_origins(W, tile, stride):
            img = read_window(scene_dir, channels, r0, c0, tile).unsqueeze(0).to(device)
            out = model(img)
            pts, scores = peaks(out["heatmap"][0].cpu(), thresh)
            if len(pts) == 0:
                continue
            length = torch.expm1(out["length"][0, 0, pts[:, 0], pts[:, 1]]).clamp(min=0).cpu()
            is_v = torch.sigmoid(out["is_vessel"][0, 0, pts[:, 0], pts[:, 1]]).cpu() > 0.5
            is_f = torch.sigmoid(out["is_fishing"][0, 0, pts[:, 0], pts[:, 1]]).cpu() > 0.5
            for i in range(len(pts)):
                rows.append((int(pts[i, 0]) + r0, int(pts[i, 1]) + c0, float(scores[i]),
                             bool(is_v[i]), bool(is_f[i] and is_v[i]), float(length[i])))

    if rows:
        pts = torch.tensor([[r[0], r[1]] for r in rows])
        sc = torch.tensor([r[2] for r in rows])
        rows = [rows[i] for i in nms(pts, sc, min_dist)]
    df = pd.DataFrame(
        [{"scene_id": scene_dir.name, "detect_scene_row": r, "detect_scene_column": c,
          "is_vessel": v, "is_fishing": f, "vessel_length_m": L, "score": s}
         for r, c, s, v, f, L in rows],
        columns=list(PRED_COLUMNS),
    )
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--scene", required=True, help="scene directory holding the channel GeoTIFFs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min-dist", type=int, default=10)
    a = ap.parse_args()
    df = infer_scene(a.ckpt, a.scene, a.out, a.thresh, a.min_dist)
    print(f"{len(df)} detections -> {a.out}")


if __name__ == "__main__":
    main()
