"""The xView3 aggregate metric (MR) via the vendored reference scorer.

    python -m planetar_fusion.eval --preds preds.csv --gt data/xview3/validation.csv \\
        [--shore-root data/xview3/shoreline] [--score-all] [--output score.json]

Wraps vendor/xview3-reference/reference/metric.py (BSD-3, DIU) rather than
re-implementing it, so the number is the leaderboard's number. Mirrors the
reference `main()`: by default only HIGH/MEDIUM-confidence labels are scored;
without shoreline contour files the close-to-shore F1 is reported as 0, exactly
as the reference does. Returns {loc_fscore, loc_fscore_shore, vessel_fscore,
fishing_fscore, length_acc, aggregate}.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "xview3-reference" / "reference"


def _metric():
    if not (VENDOR / "metric.py").exists():
        raise RuntimeError(
            f"reference scorer not found at {VENDOR}; run `git submodule update --init`"
        )
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    import metric  # noqa: E402  (flat import inside the vendored package)

    return metric


def score_predictions(preds_csv, gt_csv, shore_root=None, score_all=False, drop_low_detect=False,
                      distance_tolerance=200, shore_tolerance=2, costly_dist=False):
    m = _metric()
    pred = pd.read_csv(preds_csv, index_col=False)
    gt = pd.read_csv(gt_csv, index_col=False)
    gt = gt[gt["scene_id"].isin(pred["scene_id"].unique())].reset_index(drop=True)
    if not score_all:
        if drop_low_detect:
            pred = m.drop_low_confidence_preds(pred, gt, distance_tolerance=distance_tolerance,
                                               costly_dist=costly_dist)
        gt = gt[gt["confidence"].isin(["HIGH", "MEDIUM"])].reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    return m.score(pred, gt, shore_root, distance_tolerance, shore_tolerance, costly_dist)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--shore-root", default=None)
    ap.add_argument("--score-all", action="store_true", help="score LOW-confidence labels too")
    ap.add_argument("--output", default=None, help="write the score dict as JSON")
    a = ap.parse_args()
    s = score_predictions(a.preds, a.gt, a.shore_root, a.score_all)
    print(json.dumps(s, indent=2))
    if a.output:
        Path(a.output).write_text(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
