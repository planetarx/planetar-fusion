"""Detector loss: penalty-reduced focal loss on the centre heatmap (CenterNet,
Law & Deng 2018 / Zhou et al. 2019) plus attribute losses sampled only at the
ground-truth centres — L1 on log1p(length) and BCE on is_vessel / is_fishing.

Target contract (produced by data.py per item, batched by train.collate):
    heatmap    FloatTensor[B,1,H,W]
    points     list of B LongTensor[N_b,2]   (row, col)
    length     list of B FloatTensor[N_b]    metres (0 for non-vessels)
    is_vessel  list of B FloatTensor[N_b]    {0,1}
    is_fishing list of B FloatTensor[N_b]    {0,1}
"""
import torch
import torch.nn.functional as F


def focal_heatmap(pred, gt, alpha=2.0, beta=4.0, eps=1e-6):
    """Penalty-reduced pixelwise focal loss, normalised by the number of
    positives (gt == 1 exactly, which gaussian_splat guarantees at centres)."""
    pred = pred.clamp(eps, 1 - eps)
    pos = (gt == 1).float()
    neg = 1.0 - pos
    pos_loss = -((1 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1 - gt) ** beta) * (pred**alpha) * torch.log(1 - pred) * neg
    n = pos.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / n


def _at_points(head, pts):
    """head [1,H,W] -> values at pts[N,2]."""
    if len(pts) == 0:
        return head.new_zeros(0)
    return head[0, pts[:, 0], pts[:, 1]]


def detector_loss(out, target, w_len=0.1):
    """Returns (total, parts) where parts holds plain floats for logging."""
    hm = focal_heatmap(out["heatmap"], target["heatmap"])
    lens, vs, fs = [], [], []
    for b, pts in enumerate(target["points"]):
        if len(pts) == 0:
            continue
        lens.append(F.l1_loss(_at_points(out["length"][b], pts), torch.log1p(target["length"][b])))
        vs.append(F.binary_cross_entropy_with_logits(_at_points(out["is_vessel"][b], pts), target["is_vessel"][b]))
        fs.append(F.binary_cross_entropy_with_logits(_at_points(out["is_fishing"][b], pts), target["is_fishing"][b]))

    def mean(xs):
        return torch.stack(xs).mean() if xs else hm.new_zeros(())

    l_len, l_v, l_f = mean(lens), mean(vs), mean(fs)
    total = hm + w_len * l_len + l_v + l_f
    return total, {"hm": hm.item(), "len": l_len.item(), "v": l_v.item(), "f": l_f.item()}
