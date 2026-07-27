# planetar-fusion

The **buildable v0** of planetar's learned cross-modal fusion thesis — a model that learns the
SAR↔AIS association from AIS-on co-occurrence (free labels) and applies it to flag and re-identify
**dark (AIS-off) vessels**. The full six-modality thesis lives aspirationally in the CH13 SSOT
(`../planetar/docs/recenter-learned-fusion.md`); this repo makes the core idea real on the two
modalities we actually run — Sentinel-1 SAR + AIS — then stages the rest.

> **Status: DESIGN APPROVED 2026-07-26 — not yet built.** Read [`docs/DESIGN.md`](docs/DESIGN.md).

## The v0, in two stages (single GPU throughout)

1. **SAR dark-vessel detector — on xView3 (now).** A strong Sentinel-1 vessel detector, trained and
   **benchmarked against the public xView3 leaderboard**; darkness is detection − AIS at deployment.
   Replaces `planetar-sat`'s classical CFAR detector.
2. **Cross-modal association / fusion head — on self-collected paired data (next).** The actual
   learned SAR↔AIS association (InfoNCE over co-occurring pairs), using AIS kinematic features
   xView3 can't provide; associate or flag-as-dark, with a gallery on-ramp to cross-time re-ID.

## What's here now

| Path | State |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | ✅ The full v0 design (data, architecture, training, eval, integration) |

## What's coming (Stage 1 build)

Python + PyTorch + torchgeo; the xView3 dataloader (start on `torchgeo`'s "xView3 tiny"), the
detector, the reference `MR` scorer, and the `planetar-sat --detector learned` integration. See
[`docs/DESIGN.md` §3](docs/DESIGN.md).
