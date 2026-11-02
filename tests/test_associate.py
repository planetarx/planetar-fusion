"""Stage 2 proximity baseline: AIS at the SAR instant, gated one-to-one association."""
import json
import math

import pytest

from planetar_fusion.associate import (
    associate,
    blind_eval,
    fix_time_ns,
    footprint_from_annotation,
    load_ais,
    main,
    position_at,
)

S = 1_000_000_000
T0 = 1_793_000_000 * S  # the SAR acquisition instant in these tests
M_PER_DEG_LAT = 111_320.0


def rec(mmsi, t_ns, lat, lon, sog=0.0, cog=0.0, **kw):
    return {"mmsi": mmsi, "lastSeenNs": str(t_ns), "lat": lat, "lon": lon, "sog": sog,
            "cog": cog, "aisUtcSecond": None, "simulated": False, "name": f"V{mmsi}",
            "length": 50, **kw}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def det(lat, lon, **kw):
    return {"lat": lat, "lon": lon, "acquired_at_ns": T0, "length_m": 40.0, **kw}


def metres(lat0, lon0, lat1, lon1):
    dy = (lat1 - lat0) * M_PER_DEG_LAT
    dx = (lon1 - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    return math.hypot(dx, dy)


# --- fix time -------------------------------------------------------------------------------

def test_fix_time_uses_transmit_second_before_receipt():
    received = T0 + 7 * S + 300_000_000  # 7.3 s past T0; T0 is :20 past the minute
    assert (T0 // S) % 60 == 20
    assert fix_time_ns({"lastSeenNs": str(received), "aisUtcSecond": 25}) == T0 + 5 * S


def test_fix_time_wraps_the_minute():
    received = T0 + 41 * S  # second 1 of the next minute
    assert fix_time_ns({"lastSeenNs": str(received), "aisUtcSecond": 58}) == T0 + 38 * S


def test_fix_time_without_transmit_second_is_receipt():
    for sec in (None, 60, 61, 63):
        assert fix_time_ns({"lastSeenNs": str(T0 + 3 * S), "aisUtcSecond": sec}) == T0 + 3 * S


# --- loading --------------------------------------------------------------------------------

def test_load_ais_keeps_window_dedupes_and_drops_simulated(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [
        rec(1, T0 - 60 * S, 48.4, -123.4),
        rec(1, T0 + 60 * S, 48.41, -123.4),
        rec(1, T0 - 3 * 3600 * S, 48.0, -123.0),            # outside the window
        rec(2, T0, 48.5, -123.5, simulated=True),          # never real evidence
    ])
    b = write_jsonl(tmp_path / "b.jsonl", [rec(1, T0 + 60 * S, 48.41, -123.4)])  # duplicate
    vessels = load_ais([a, b], T0, window_s=1800)
    assert list(vessels) == [1]
    assert [f.t_ns for f in vessels[1].fixes] == [T0 - 60 * S, T0 + 60 * S]


# --- position at the SAR instant ------------------------------------------------------------

def test_position_interpolates_between_bracketing_fixes(tmp_path):
    v = load_ais([write_jsonl(tmp_path / "a.jsonl", [
        rec(1, T0 - 60 * S, 48.40, -123.40, sog=10),
        rec(1, T0 + 180 * S, 48.44, -123.40, sog=10),
    ])], T0)[1]
    p = position_at(v, T0)
    assert p.method == "interpolated"
    assert p.lat == pytest.approx(48.41)
    assert p.dt_s == pytest.approx(60)


def test_position_dead_reckons_from_last_fix(tmp_path):
    # 10 kn due east for 120 s = 617.3 m.
    v = load_ais([write_jsonl(tmp_path / "a.jsonl", [
        rec(1, T0 - 120 * S, 48.40, -123.40, sog=10, cog=90)])], T0)[1]
    p = position_at(v, T0)
    assert p.method == "dead-reckoned"
    assert metres(48.40, -123.40, p.lat, p.lon) == pytest.approx(617.3, rel=0.01)
    assert p.lon > -123.40 and p.lat == pytest.approx(48.40, abs=1e-5)
    assert p.sigma_m > 50


def test_position_none_when_last_fix_is_too_old(tmp_path):
    v = load_ais([write_jsonl(tmp_path / "a.jsonl", [
        rec(1, T0 - 1200 * S, 48.40, -123.40, sog=10)])], T0)[1]
    assert position_at(v, T0, max_extrap_s=600) is None


def test_position_long_gap_falls_back_to_nearest_fix(tmp_path):
    # Fixes 40 min apart do not bracket reliably; dead-reckon from the nearer one.
    v = load_ais([write_jsonl(tmp_path / "a.jsonl", [
        rec(1, T0 - 2300 * S, 48.0, -123.0, sog=0),
        rec(1, T0 + 100 * S, 48.5, -123.5, sog=0),
    ])], T0, window_s=3600)[1]
    p = position_at(v, T0, max_interp_s=900)
    assert p.method == "dead-reckoned"
    assert (p.lat, p.lon) == pytest.approx((48.5, -123.5))


# --- association ----------------------------------------------------------------------------

def _vessels(tmp_path, rows):
    return load_ais([write_jsonl(tmp_path / "a.jsonl", rows)], T0)


def test_associate_matches_one_to_one_and_flags_dark(tmp_path):
    vessels = _vessels(tmp_path, [
        rec(1, T0, 48.400, -123.400),
        rec(2, T0, 48.420, -123.400),
        rec(3, T0, 48.600, -123.900),  # no detection near it
    ])
    dets = [det(48.4005, -123.400), det(48.4195, -123.400), det(48.300, -123.200)]
    r = associate(dets, vessels, T0, gate_m=500)
    assert sorted((m["det_index"], m["mmsi"]) for m in r["matched"]) == [(0, 1), (1, 2)]
    assert [d["det_index"] for d in r["dark"]] == [2]
    assert r["dark"][0]["nearest_mmsi"] == 1
    assert [u["mmsi"] for u in r["unseen"]] == [3]


def test_associate_prefers_global_assignment_over_greedy(tmp_path):
    # Greedy would give det 0 to vessel 1 (closest pair) and leave det 1 unmatched;
    # the optimal assignment matches both.
    vessels = _vessels(tmp_path, [rec(1, T0, 48.4000, -123.4), rec(2, T0, 48.4040, -123.4)])
    dets = [det(48.4010, -123.4), det(48.3975, -123.4)]
    r = associate(dets, vessels, T0, gate_m=500)
    assert sorted((m["det_index"], m["mmsi"]) for m in r["matched"]) == [(0, 2), (1, 1)]
    assert r["dark"] == []


def test_associate_gate_widens_with_position_uncertainty(tmp_path):
    # Last fix 8 min old at 12 kn: the estimate is uncertain, so a detection 700 m from it
    # still associates, though it is beyond the 500 m base gate.
    vessels = _vessels(tmp_path, [rec(1, T0 - 480 * S, 48.40, -123.40, sog=12.0, cog=0.0)])
    p = position_at(vessels[1], T0)
    dets = [det(p.lat + 700 / M_PER_DEG_LAT, p.lon)]
    r = associate(dets, vessels, T0, gate_m=500)
    assert [m["mmsi"] for m in r["matched"]] == [1]


def test_unseen_limited_to_footprint(tmp_path):
    vessels = _vessels(tmp_path, [rec(1, T0, 48.4, -123.4), rec(2, T0, 49.5, -125.0)])
    footprint = [(-124.0, 48.0), (-123.0, 48.0), (-123.0, 49.0), (-124.0, 49.0)]
    r = associate([], vessels, T0, footprint=footprint)
    assert [u["mmsi"] for u in r["unseen"]] == [1]


def test_blind_eval_counts_dark_vs_misassociated(tmp_path):
    # Hiding vessel 1 or 2 leaves its detection with no AIS in reach: flagged dark. Vessel 4
    # has no detection of its own, 334 m from vessel 3's: hiding 3 hands that detection to 4.
    vessels = _vessels(tmp_path, [
        rec(1, T0, 48.40, -123.40),
        rec(2, T0, 48.60, -123.60),
        rec(3, T0, 48.5000, -123.0),
        rec(4, T0, 48.5030, -123.0),
    ])
    dets = [det(48.40, -123.40), det(48.60, -123.60), det(48.5010, -123.0)]
    r = blind_eval(dets, vessels, T0, gate_m=500)
    # Matched: 1, 2, and det 2 ↔ vessel 3 (111 m) rather than 4 (334 m).
    assert r["matched"] == 3
    assert r["flagged_dark"] == 2       # hiding 1 or 2
    assert r["misassociated"] == 1      # hiding 3: its detection goes to vessel 4


def test_footprint_from_annotation(tmp_path):
    pts = "".join(
        f"<geolocationGridPoint><line>{ln}</line><pixel>{px}</pixel>"
        f"<latitude>{48 + ln / 100}</latitude><longitude>{-124 + px / 100}</longitude>"
        "</geolocationGridPoint>"
        for ln in (0, 50, 100) for px in (0, 50, 100))
    ann = tmp_path / "a.xml"
    ann.write_text(f"<product><geolocationGrid><geolocationGridPointList>{pts}"
                   "</geolocationGridPointList></geolocationGrid></product>")
    fp = footprint_from_annotation(ann)
    assert sorted(fp) == [(-124.0, 48.0), (-124.0, 49.0), (-123.0, 48.0), (-123.0, 49.0)]


# --- CLI ------------------------------------------------------------------------------------

def test_main_writes_jsonl_and_summary(tmp_path, capsys):
    dets_path = write_jsonl(tmp_path / "dets.jsonl", [det(48.40, -123.40), det(48.30, -123.2)])
    ais_path = write_jsonl(tmp_path / "ais.jsonl", [rec(1, T0 - 30 * S, 48.40, -123.40)])
    out = tmp_path / "out.jsonl"
    main(["--dets", str(dets_path), "--ais", str(ais_path), "--out", str(out)])
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert sorted(r["status"] for r in rows) == ["dark", "matched"]
    assert "1 matched, 1 dark" in capsys.readouterr().err
