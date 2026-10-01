"""CenterNetSAR — a small U-Net that reads a stack of SAR + ancillary channels
and emits four full-resolution maps: a sigmoid centre heatmap and three
attribute heads (log-length, is_vessel logit, is_fishing logit) that are
only ever read at object centres (see losses.py / infer.py).

Deliberately plain (DESIGN §7: pick the smallest thing that reaches a
credible MR; the novelty of this project is Stage 2's association, not the
detector architecture). Three pool levels keep the receptive field around
~50 px at 10 m/px — a few ship lengths — while staying cheap on one GPU.
"""
import torch
import torch.nn as nn


def cbr(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class CenterNetSAR(nn.Module):
    OUTPUTS = ("heatmap", "length", "is_vessel", "is_fishing")

    def __init__(self, in_ch=5, base=32):
        super().__init__()
        self.e1 = cbr(in_ch, base)
        self.e2 = cbr(base, base * 2)
        self.e3 = cbr(base * 2, base * 4)
        self.e4 = cbr(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.d3 = cbr(base * 8 + base * 4, base * 4)
        self.d2 = cbr(base * 4 + base * 2, base * 2)
        self.d1 = cbr(base * 2 + base, base)
        self.head_hm = nn.Conv2d(base, 1, 1)
        self.head_len = nn.Conv2d(base, 1, 1)
        self.head_v = nn.Conv2d(base, 1, 1)
        self.head_f = nn.Conv2d(base, 1, 1)
        # CenterNet trick: start the heatmap head biased towards "background"
        # so the focal loss does not spend its first epochs fighting a 0.5 prior.
        nn.init.constant_(self.head_hm.bias, -2.19)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.up(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up(d2), e1], 1))
        return {
            "heatmap": torch.sigmoid(self.head_hm(d1)),
            "length": self.head_len(d1),
            "is_vessel": self.head_v(d1),
            "is_fishing": self.head_f(d1),
        }
