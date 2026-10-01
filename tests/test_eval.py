"""Task 7 — the xView3 MR metric via the vendored reference scorer."""
import pandas as pd

from planetar_fusion.eval import score_predictions

GT_COLS = ["scene_id", "detect_scene_row", "detect_scene_column", "is_vessel", "is_fishing",
           "vessel_length_m", "confidence", "distance_from_shore_km"]


def _gt():
    return pd.DataFrame([
        ["s1", 100, 100, True, True, 40.0, "HIGH", 30.0],
        ["s1", 500, 700, True, False, 180.0, "HIGH", 30.0],
        ["s1", 900, 300, False, None, None, "MEDIUM", 30.0],
        ["s1", 1200, 1200, True, False, 25.0, "LOW", 30.0],   # dropped unless score_all
    ], columns=GT_COLS)


def test_perfect_predictions_score_one(tmp_path):
    gt = _gt()
    preds = gt[gt.confidence != "LOW"][["scene_id", "detect_scene_row", "detect_scene_column",
                                        "is_vessel", "is_fishing", "vessel_length_m"]].copy()
    preds["score"] = 0.99
    gt.to_csv(tmp_path / "gt.csv", index=False)
    preds.to_csv(tmp_path / "preds.csv", index=False)
    s = score_predictions(tmp_path / "preds.csv", tmp_path / "gt.csv")
    assert s["loc_fscore"] == 1.0
    assert s["vessel_fscore"] == 1.0 and s["fishing_fscore"] == 1.0
    assert s["length_acc"] == 1.0
    assert s["loc_fscore_shore"] == 0  # no shoreline files -> reported as 0, per the reference
    assert 0.0 <= s["aggregate"] <= 1.0


def test_missing_and_spurious_detections_lower_the_score(tmp_path):
    gt = _gt()
    preds = pd.DataFrame([["s1", 100, 100, True, True, 40.0, 0.9],
                          ["s1", 2000, 2000, True, False, 50.0, 0.8]],   # one hit, one false alarm
                         columns=["scene_id", "detect_scene_row", "detect_scene_column",
                                  "is_vessel", "is_fishing", "vessel_length_m", "score"])
    gt.to_csv(tmp_path / "gt.csv", index=False)
    preds.to_csv(tmp_path / "preds.csv", index=False)
    s = score_predictions(tmp_path / "preds.csv", tmp_path / "gt.csv")
    assert 0.0 < s["loc_fscore"] < 1.0
