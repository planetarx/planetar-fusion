"""Task 4 — the canonical ML sanity check: model + loss + optimiser can overfit one batch."""
import torch

from planetar_fusion.data import gaussian_splat
from planetar_fusion.losses import detector_loss
from planetar_fusion.model import CenterNetSAR


def test_overfits_one_batch():
    torch.manual_seed(0)
    x = torch.randn(1, 5, 128, 128)
    pts = torch.tensor([[30, 40], [90, 100]])
    target = {
        "heatmap": gaussian_splat((128, 128), pts).unsqueeze(0),
        "points": [pts],
        "length": [torch.tensor([120.0, 80.0])],
        "is_vessel": [torch.tensor([1.0, 1.0])],
        "is_fishing": [torch.tensor([0.0, 1.0])],
    }
    m = CenterNetSAR(5)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss, _parts = detector_loss(m(x), target)
        loss.backward()
        opt.step()
        first = first or loss.item()
    assert loss.item() < 0.5 * first  # loss at least halves


def test_loss_with_no_points_is_finite():
    x = torch.randn(1, 5, 64, 64)
    empty = torch.zeros(0, 2, dtype=torch.long)
    target = {
        "heatmap": gaussian_splat((64, 64), empty).unsqueeze(0),
        "points": [empty], "length": [torch.zeros(0)],
        "is_vessel": [torch.zeros(0)], "is_fishing": [torch.zeros(0)],
    }
    loss, parts = detector_loss(CenterNetSAR(5)(x), target)
    assert torch.isfinite(loss)
    assert parts["len"] == 0.0 and parts["v"] == 0.0 and parts["f"] == 0.0
