"""Task 3 — the detector produces full-resolution heads of the right shape."""
import torch

from planetar_fusion.model import CenterNetSAR


def test_forward_shapes():
    m = CenterNetSAR(in_ch=5)
    out = m(torch.randn(2, 5, 256, 256))
    for k in ("heatmap", "length", "is_vessel", "is_fishing"):
        assert out[k].shape == (2, 1, 256, 256), k
    assert out["heatmap"].min() >= 0 and out["heatmap"].max() <= 1  # sigmoid
