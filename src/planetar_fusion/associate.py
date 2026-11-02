"""Stage 2 proximity baseline: SAR detections ↔ AIS at the acquisition instant.

    python -m planetar_fusion.associate --dets dets.jsonl \\
        --ais ../planetar-ais/data/ais/ais-2026-11-11.jsonl ../planetar-ais/data/ais-orca/... \\
        --annotation <vv annotation .xml> --out assoc.jsonl

`dets.jsonl` is `planetar-sat detect` output (lat, lon, acquired_at_ns per detection). Each AIS
vessel is placed at the SAR instant — interpolated between the fixes either side, or
dead-reckoned from the nearest fix by its speed and course — with an uncertainty that grows
with the time to that fix. Detections and vessels are then assigned one-to-one (Hungarian)
inside a gate of max(gate_m, 3σ). A detection with no AIS in reach is a dark candidate; an
AIS vessel inside the scene footprint with no detection is "unseen" (too small for SAR,
alongside, or a bad fix). This is the baseline DESIGN.md §4.4 measures the learned head
against, and the source of its free AIS-on labels.
"""
import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

NS = 1_000_000_000
KN_TO_MPS = 1852 / 3600
M_PER_DEG_LAT = 111_320.0


@dataclass
class Fix:
    t_ns: int
    lat: float
    lon: float
    sog: float  # kn
    cog: float  # degrees true


@dataclass
class AisVessel:
    mmsi: int
    name: str | None
    length: float | None
    type: str | None
    fixes: list[Fix] = field(default_factory=list)


@dataclass
class Position:
    lat: float
    lon: float
    sigma_m: float
    method: str  # "interpolated" | "dead-reckoned"
    dt_s: float  # seconds to the nearest fix used


def fix_time_ns(r):
    """When the vessel transmitted: the AIS UTC second (0–59) at or before receipt, else
    the receipt time. aisstream delivers fixes seconds late; a 13 kn ship covers 7 m/s."""
    recv = int(r["lastSeenNs"])
    sec = r.get("aisUtcSecond")
    if sec is None or not 0 <= sec <= 59:
        return recv
    t = recv - recv % (60 * NS) + sec * NS
    return t - 60 * NS if t > recv else t


def load_ais(paths, t_ns, window_s=1800):
    """Real (non-simulated) AIS fixes within ±window_s of t_ns, by MMSI, de-duplicated and
    time-ordered. Several recorders may overlap (Victoria and orca boxes)."""
    vessels = {}
    seen = set()
    lo, hi = t_ns - window_s * NS, t_ns + window_s * NS
    for p in paths:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("simulated") or r.get("lat") is None or r.get("lon") is None:
                    continue
                t = fix_time_ns(r)
                if not lo <= t <= hi or (r["mmsi"], t) in seen:
                    continue
                seen.add((r["mmsi"], t))
                v = vessels.setdefault(r["mmsi"], AisVessel(
                    r["mmsi"], r.get("name"), r.get("length"), r.get("type")))
                v.fixes.append(Fix(t, r["lat"], r["lon"], r.get("sog") or 0.0, r.get("cog") or 0.0))
    for v in vessels.values():
        v.fixes.sort(key=lambda f: f.t_ns)
    return dict(sorted(vessels.items()))


def _offset(lat, lon, north_m, east_m):
    return (lat + north_m / M_PER_DEG_LAT,
            lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat))))


def metres(lat0, lon0, lat1, lon1):
    """Flat-earth distance; fine at the few-km scale of a gate."""
    dy = (lat1 - lat0) * M_PER_DEG_LAT
    dx = (lon1 - lon0) * M_PER_DEG_LAT * math.cos(math.radians((lat0 + lat1) / 2))
    return math.hypot(dx, dy)


def position_at(v, t_ns, max_interp_s=900, max_extrap_s=600, base_sigma_m=50.0,
                drift_frac=0.1):
    """The vessel's estimated position at t_ns, or None when its nearest fix is older than
    max_extrap_s. σ = base_sigma_m + drift_frac × (distance run since/until that fix)."""
    before = [f for f in v.fixes if f.t_ns <= t_ns]
    after = [f for f in v.fixes if f.t_ns > t_ns]
    a, b = (before[-1] if before else None), (after[0] if after else None)
    if a and b and (b.t_ns - a.t_ns) <= max_interp_s * NS:
        w = (t_ns - a.t_ns) / (b.t_ns - a.t_ns)
        dt_s = min(t_ns - a.t_ns, b.t_ns - t_ns) / NS
        speed = max(a.sog, b.sog) * KN_TO_MPS
        return Position(a.lat + w * (b.lat - a.lat), a.lon + w * (b.lon - a.lon),
                        base_sigma_m + drift_frac * speed * dt_s, "interpolated", dt_s)
    near = min((f for f in (a, b) if f), key=lambda f: abs(f.t_ns - t_ns), default=None)
    if near is None or abs(near.t_ns - t_ns) > max_extrap_s * NS:
        return None
    dt = (t_ns - near.t_ns) / NS  # negative: reckon backwards from a later fix
    run = near.sog * KN_TO_MPS * dt
    c = math.radians(near.cog)
    lat, lon = _offset(near.lat, near.lon, run * math.cos(c), run * math.sin(c))
    return Position(lat, lon, base_sigma_m + drift_frac * abs(run), "dead-reckoned", abs(dt))


def _inside(lon, lat, poly):
    """Ray-casting point-in-polygon over [(lon, lat), ...]."""
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > lat) != (y1 > lat) and lon < x0 + (lat - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def footprint_from_annotation(path):
    """The scene's footprint, [(lon, lat), ...], as the convex hull of the annotation's
    geolocation grid."""
    root = ET.parse(path).getroot()
    pts = np.array([(float(g.findtext("longitude")), float(g.findtext("latitude")))
                    for g in root.iter("geolocationGridPoint")])
    return [tuple(map(float, pts[i])) for i in ConvexHull(pts).vertices]


def associate(dets, vessels, t_ns, gate_m=500.0, footprint=None, **pos_kw):
    """One-to-one detection↔vessel assignment minimising total distance within the gate.

    Returns {"matched": [...], "dark": [...], "unseen": [...]}; every entry carries the
    distances and AIS timing behind it so a reader can check the call.
    """
    est = {m: p for m, v in vessels.items() if (p := position_at(v, t_ns, **pos_kw))}
    mmsis = list(est)
    dist = np.array([[metres(d["lat"], d["lon"], est[m].lat, est[m].lon) for m in mmsis]
                     for d in dets]).reshape(len(dets), len(mmsis))
    gate = np.array([max(gate_m, 3 * est[m].sigma_m) for m in mmsis])
    ok = dist <= gate
    big = 1e9  # infeasible pairs: never preferred over leaving both unassigned
    rows, cols = linear_sum_assignment(np.where(ok, dist, big)) if dist.size else ([], [])
    pairs = {int(i): int(j) for i, j in zip(rows, cols) if ok[i, j]}

    matched, dark = [], []
    for i, d in enumerate(dets):
        if i in pairs:
            m = mmsis[pairs[i]]
            v, p = vessels[m], est[m]
            matched.append({"det_index": i, "mmsi": m, "name": v.name, "dist_m": dist[i, pairs[i]],
                            "gate_m": gate[pairs[i]], "method": p.method, "fix_dt_s": p.dt_s,
                            "ais_length_m": v.length, "sar_length_m": d.get("length_m")})
        else:
            j = int(np.argmin(dist[i])) if mmsis else None
            dark.append({"det_index": i,
                         "nearest_mmsi": mmsis[j] if j is not None else None,
                         "nearest_m": float(dist[i, j]) if j is not None else None,
                         "sar_length_m": d.get("length_m")})
    taken = set(pairs.values())
    unseen = [{"mmsi": m, "name": vessels[m].name, "lat": est[m].lat, "lon": est[m].lon,
               "ais_length_m": vessels[m].length, "method": est[m].method}
              for j, m in enumerate(mmsis)
              if j not in taken and (footprint is None or _inside(est[m].lon, est[m].lat, footprint))]
    return {"matched": matched, "dark": dark, "unseen": unseen}


def blind_eval(dets, vessels, t_ns, **kw):
    """DESIGN.md §4.4 AIS-blinding: hide each matched vessel's AIS in turn and re-associate.
    Its detection should come back dark; going to another vessel is a mis-association."""
    base = associate(dets, vessels, t_ns, **kw)
    dark = mis = 0
    for m in base["matched"]:
        hidden = {k: v for k, v in vessels.items() if k != m["mmsi"]}
        r = associate(dets, hidden, t_ns, **kw)
        if any(d["det_index"] == m["det_index"] for d in r["dark"]):
            dark += 1
        else:
            mis += 1
    return {"matched": len(base["matched"]), "flagged_dark": dark, "misassociated": mis}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", required=True, help="planetar-sat detect JSON-lines")
    ap.add_argument("--ais", nargs="+", required=True, help="planetar-ais recorder JSONL files")
    ap.add_argument("--annotation", help="VV annotation XML: limits 'unseen' to the footprint")
    ap.add_argument("--gate-m", type=float, default=500.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dets = [json.loads(line) for line in open(a.dets) if line.strip()]
    footprint = footprint_from_annotation(a.annotation) if a.annotation else None
    by_t = {}
    for d in dets:
        by_t.setdefault(int(d["acquired_at_ns"]), []).append(d)
    with open(a.out, "w") as out:
        for t_ns, scene in sorted(by_t.items()):
            vessels = load_ais(a.ais, t_ns)
            r = associate(scene, vessels, t_ns, gate_m=a.gate_m, footprint=footprint)
            b = blind_eval(scene, vessels, t_ns, gate_m=a.gate_m)
            for status in ("matched", "dark"):
                for e in r[status]:
                    out.write(json.dumps({"status": status, "acquired_at_ns": t_ns,
                                          **scene[e["det_index"]], **e}, default=float) + "\n")
            for e in r["unseen"]:
                out.write(json.dumps({"status": "unseen", "acquired_at_ns": t_ns, **e},
                                     default=float) + "\n")
            print(f"{len(r['matched'])} matched, {len(r['dark'])} dark, "
                  f"{len(r['unseen'])} unseen AIS ({len(vessels)} vessels with fixes ±30 min); "
                  f"blinding: {b['flagged_dark']} of {b['matched']} flagged dark, "
                  f"{b['misassociated']} mis-associated", file=sys.stderr)


if __name__ == "__main__":
    main()
