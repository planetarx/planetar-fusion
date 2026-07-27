# planetar-fusion — design spec (v0: learned SAR↔AIS dark-vessel model)

**Status:** design approved 2026-07-26 (brainstormed with the founder). Not yet built.
**Repo:** `~/github/planetarx/planetar-fusion` (a new top-level project in the planetar workspace).

---

## 1. What this is

The **buildable v0** of planetar's central R&D thesis — a *learned cross-modal fusion model that
re-identifies dark (AIS-off) vessels*. The full thesis (six modalities, shared embedding,
uncertainty, explainability) is specified aspirationally in the CH13 SSOT
(`../planetar/docs/recenter-learned-fusion.md`). **This project makes the core idea real, small, and
evaluable** on the two modalities we actually have working (Sentinel-1 SAR + AIS), then stages the
rest.

The idea, in one line: **during AIS-on, a broadcasting vessel labels the concurrent SAR observation
for free; the model learns that cross-modal association and applies it when the vessel goes dark.**

### 1.1 Relationship to the platform and the thesis

| Piece | Role |
|---|---|
| `planetar-sat` | Already fetches Sentinel-1 (CDSE) and runs a **classical CFAR** ship detector → track. v0 Stage 1 **replaces CFAR with a learned detector**. |
| `planetar-ais` | Live/collected AIS — the supervision signal and the dark-event trigger. |
| the bus + `planetar-ontology` | Where the association runs at deployment: SAR detections − AIS matches = dark vessels; associations enrich the entity graph. |
| CH13 SSOT | The aspirational 6-modality model. **This repo is its v0** — SAR+AIS first, honest about the minimal pair. |
| `planetar-vaults` | The durable knowledge layer; a re-identified dark vessel becomes/updates a vault dossier. |

**Honesty spine:** the learned model is the *net-new science*; the substrate (bus, envelope, ontology,
detectors, UI) is *built and de-risks it*. This mirrors the CH13 build/propose split.

---

## 2. The v0 in two stages

The data forces a clean split (see §3.1):

- **Stage 1 — the SAR dark-vessel detector, on xView3 (now).** A strong Sentinel-1 vessel detector
  trained and **benchmarked against the public xView3 leaderboard**. "Dark" is determined at
  deployment by differencing detections against the live AIS feed — *not* a SAR-only head.
- **Stage 2 — the cross-modal association/fusion head, on self-collected paired data (next).** The
  actual learned SAR↔AIS association, using AIS kinematic features xView3 cannot provide. Reproduces
  the free-label recipe on our own Sentinel-1 + AIS.

Single GPU throughout. Both upgrades land back in the live platform.

---

## 3. Stage 1 — SAR dark-vessel detector

### 3.1 Data — xView3-SAR

DIU + Global Fishing Watch (NeurIPS 2022; arXiv 2206.00897). Register at
`https://iuu.xview.us/dataset`; reference/winning code at `github.com/DIUx-xView`.

- **991 Sentinel-1 scenes** (IW GRD, ~29k×24k px, 10 m), **243,018 objects, 220k+ dark instances**.
- Per scene: **VH + VV** GeoTIFFs + **5 ancillary rasters** (bathymetry/GEBCO, wind speed, wind
  direction, wind quality, land/ice mask).
- Per-detection labels: `is_vessel`, `is_fishing`, `vessel_length_m`, `confidence`, **`source`
  (AIS / VMS / manual)** — *no AIS/VMS source = the dark label, for free*, `distance_from_shore_km`.
- Splits: train 554 (~1.3 TB, GFW-automated labels), **validation 50 (~123 GB, human-verified)**,
  public 150 (leaderboard), holdout 237 (withheld). **~2 TB public.**
- **Benchmark:** aggregate `MR` = f(detection F1 `F1_D`, close-to-shore F1 `F1_S`, `is_vessel` F1,
  `is_fishing` F1, length aggregate-percent-error); reference + 1st–5th scores published.
- **License:** DIU "free and open" for academic/industrial/**commercial**/open-source use (unlike
  GFW's CC BY-NC). Reference code MIT/Apache.
- **Compute reference:** challenge baseline = **1× V100, 60 GB RAM**; 1st place ~63 h train. One GPU.
- **Practical:** don't pull all ~2 TB — work from **validation (50) + a train subset**; stand up the
  dataloader on `torchgeo`'s "xView3 tiny" first; **mirror our copy on download** (hosted on
  iuu.xview.us, not AWS Open Data — hosting-retirement risk).

### 3.2 Model

- **Input:** tile scenes into overlapping patches (e.g. 800–1024 px, ~10% overlap); channels =
  **VH, VV, bathymetry, land/ice mask, wind** (normalize per-channel; SAR in dB).
- **Backbone → heads:** an FPN / U-Net (ResNet-ish or lightweight encoder) producing a
  **detection heatmap** (object centers, Gaussian targets) + a `length_m` regression head +
  `is_vessel` / `is_fishing` classification heads.
- **Losses:** focal/penalty-reduced-focal on the heatmap; L1/smooth-L1 on length (log-space);
  BCE on `is_vessel` / `is_fishing`.
- **Inference:** per-tile peak extraction → **tile-boundary NMS** → per-detection attributes.

### 3.3 Eval

The **xView3 `MR` metric via the reference scoring code** — a real number vs. the published
leaderboard. Report `F1_D`, `F1_S`, class F1s, and length error separately too. Sanity target: beat
the reference baseline; stretch: approach the mid-leaderboard.

### 3.4 Integration payoff

The trained detector **replaces `planetar-sat`'s CFAR detector** (a learned upgrade + the first
genuinely learned model in the stack). Export as a `planetar-sat detect` backend option
(`--detector learned`), emitting the same `sar.chip` / detection bus messages.

---

## 4. Stage 2 — cross-modal association / fusion head

### 4.1 Data — self-collected paired Sentinel-1 + AIS

`planetar-sat` fetches Sentinel-1 over a pilot area; pair each scene with **concurrent AIS**:

- **First pilot: Marine Cadastre / NOAA AccessAIS** (US waters, 2009–present; GeoParquet;
  **US-Gov public domain — commercial-clean**). Fetch Sentinel-1 for US-coastal acquisition
  windows, join AIS at acquisition time.
- **Victoria: the AIS receiver** (see `../planetar-ais/docs/ais-receiver-buildout.md`) logged
  alongside Sentinel-1 fetches over the Salish Sea.
- Free labels: during AIS-on, space-time proximity gives the true SAR-detection↔AIS-vessel pairs
  (use GFW's `paper-longline-ais-sar-matching` as a reference matching method to bootstrap pairs).

### 4.2 Model

- **SAR chip encoder:** reuse the Stage-1 backbone; crop the detection chip → embedding.
- **AIS encoder:** small MLP over the candidate AIS vessel's features — **position extrapolated to
  the SAR acquisition instant** (dead-reckon from last ping via speed/course), speed, course,
  `length`, `type`, time-since-ping.
- **Shared embedding + loss:** project both to a common metric space; **InfoNCE contrastive** over
  co-occurring pairs within a scene (positive = the matched pair; negatives = other vessels in-scene).
- **Why learned beats proximity:** AIS pings lag the acquisition and carry error; vessels move;
  several can be close. The model disambiguates using SAR size/shape + AIS kinematics + dimensions —
  the "learned, adaptive association, not static aggregation" the thesis demands.

### 4.3 Inference

For each SAR detection: **spatial-gate** to nearby AIS candidates → score in the embedding → best
above threshold = **associated**; poor/no match = **dark** → match the dark chip's embedding against
a **gallery** of known vessel signatures (the on-ramp to cross-time re-ID).

### 4.4 Eval

- **Association accuracy** vs. a **nearest-neighbour proximity baseline**, on held-out AIS-on data
  where the true match is known.
- **Dark precision/recall via AIS-blinding:** hide a known vessel's AIS → is it flagged dark rather
  than mis-associated?

### 4.5 Thesis hooks (not v0-blocking)

Conformal calibration on the match score (distribution-free confidence); an explanation of *which
evidence drove the match* (which features/modality dominated). Cheap adds once the core works.

---

## 5. Stack & compute

- **Python + PyTorch**; `torchgeo` / `rasterio` for the Sentinel-1 GeoTIFF + ancillary-raster
  handling and tiling; xView3 reference scoring code (fork) for the metric.
- **Single GPU** (V100 / RTX 3090 / 4090-class) is sufficient for both stages (per the xView3
  reference setup); grants would fund cloud GPU if local is insufficient. Chips in Stage 2 are small.
- Repo layout (proposed): `src/` (data, model, train, eval, infer), `docs/`, `configs/`;
  `data/`, `runs/`, `checkpoints/`, `*.tif` are git-ignored.

---

## 6. Non-goals (v0)

- Not the full 6-modality model — EO / acoustic / RF / text are *later* (the CH13 thesis path), not v0.
- Not a SAR-only "dark classifier" — darkness is detection − AIS at deployment (§3, honesty).
- No GFW-derived data in any commercial-path model (CC BY-NC); xView3 + Marine Cadastre are clean.
- Not real-time training — inference integrates with the live bus; training is offline.
- No new SAR *detector* research beyond what hits a credible xView3 score — the novelty is the
  Stage-2 learned *association*, not another detector architecture.

## 7. Open questions (resolve during build)

- Exact detector backbone/size vs. the single-GPU budget — pick the smallest that reaches a credible
  `MR`; start from a published xView3 reference solution rather than inventing one.
- Pilot area for Stage 2 (US-waters Marine Cadastre first for clean paired data; Victoria via the
  receiver once it's up).
- Whether Stage 1's detector or a published pretrained xView3 model is the faster start (fork first,
  train second).
