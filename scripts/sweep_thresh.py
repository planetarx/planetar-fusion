"""Re-filter one low-threshold bench run (preds.csv with scores) at several thresholds and score
each. Equivalent to re-running inference per threshold: NMS only lets a higher-scoring point
suppress a lower one. Usage: scripts/sweep_thresh.py <bench dir run at --thresh 0.05>
"""
import json, sys, tempfile
from pathlib import Path
import pandas as pd
from planetar_fusion.eval import score_predictions
d = Path(sys.argv[1]); labels = "data/xview3/split/holdout.csv"
pred = pd.read_csv(d / "preds.csv"); scenes = sorted(pd.read_csv(labels).scene_id.unique())
rows = []
for t in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    p = d / f"preds_t{t}.csv"; pred[pred.score > t].to_csv(p, index=False)
    s = score_predictions(p, labels, shore_root="data/xview3/shoreline", scene_ids=scenes)
    s = {k: round(float(v), 4) for k, v in s.items()}; s.update(thresh=t, n_preds=int((pred.score > t).sum()))
    rows.append(s); print(json.dumps(s), flush=True)
(d / "sweep.json").write_text(json.dumps(rows, indent=2))
