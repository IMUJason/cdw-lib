#!/usr/bin/env python3
"""CDW-LIB latent-world generator.

One latent world per (archetype, size, seed) cell: sites with coordinates,
districts, and daily demand decomposed into ~10 t atoms; facilities as
immutable entities with capacity normalized to demand x PRESSURE; service
durations; time-window centers drawn from the frozen hour histogram; and
the fleet's rated payload. Observation levels (L3 full / L2 window-free /
L1 aggregate) and the single-factor variants (SP / FL) are pure functions
over the latent world, so paired instances satisfy the entity invariants
by construction. Two-way window truncation keeps every load constructively
feasible. Run with --validate to re-verify the invariance pairs.
"""
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = Path(__file__).resolve().parent.parent / "data"
GPS_RAW = (Path(__file__).resolve().parent.parent.parent / "gps_raw" / "raw" /
           "shaoxing" / "205427_gps_snapshot")
OUT = Path(__file__).resolve().parent.parent / "instances"

DAY_START, DAY_END = 9 * 60, 21 * 60 + 30
AVG_SPEED = 30.0
PRESSURE = 1.3
TRIP_T = 10.0


def _haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl, dp = math.radians(lon2 - lon1), math.radians(lat2 - lat1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def hour_profile():
    """Hour-of-day histogram driving L3 time-window centers.

    Reads the frozen histogram shipped in frozen_v0.2/hour_profile.json when
    the raw GPS snapshots are absent, so the family regenerates from the
    package alone; otherwise rebuilds the histogram from source.
    """
    frozen = FROZEN / "hour_profile.json"
    if frozen.exists():
        with open(frozen) as fh:
            return [tuple(x) for x in json.load(fh)["hour_counts"]]
    hours = Counter()
    for p in sorted(GPS_RAW.glob("part*.csv")):
        df = pd.read_csv(p, usecols=["gxsj"])
        for h in pd.to_datetime(df["gxsj"]).dt.hour:
            hours[int(h)] += 1
    return sorted(hours.items())


def build_pools():
    t = {n: pd.read_csv(FROZEN / f"{n}.csv", low_memory=False)
         for n in ["site", "site_demand", "disposal", "vehicle"]}
    dem = t["site_demand"][~t["site_demand"].is_sentinel]
    tot = dem.groupby("site_id").amount_kg.sum().rename("total").reset_index()
    nb = t["site"][t["site"].city == "宁波"].merge(tot, on="site_id", how="inner")
    nb = nb[(nb.lat_wgs84 > 0) & (nb.total > 0)].copy()
    nb["dur"] = (pd.to_datetime(nb.end_date) - pd.to_datetime(nb.start_date)
                 ).dt.days.clip(lower=30, upper=3650)
    nb = nb[nb.dur > 0]
    nb["daily_t"] = nb.total / nb.dur / 1000.0
    nb = nb[(nb.daily_t > 10) & (nb.daily_t < 5000)]
    nb["district"] = nb.district_code.astype(str).str[:6]

    nbd = t["disposal"][t["disposal"].city == "宁波"]
    nbd = nbd[(nbd.lat_wgs84 > 0) & (nbd.total_capacity_kg > 0)].copy()

    sxd = t["disposal"][t["disposal"].city == "绍兴"].copy()
    sxd["xk"] = pd.to_numeric(sxd.remain_capacity_kg, errors="coerce")
    sxd = sxd[(sxd.lat_wgs84 > 0) & (sxd.xk > 1000)].copy()

    veh = t["vehicle"]
    rl = veh[(veh.city == "宁波") & (veh.vehicle_type.str.contains("渣土", na=False))]
    rl = pd.to_numeric(rl.rated_load, errors="coerce")
    rl = rl[(rl > 5) & (rl < 40)]

    dq = t["site"][t["site"].city == "湖州"]
    dq = dq[dq.lng_wgs84 > 0]

    nb_daily = nb.daily_t.tolist()
    return {
        "NB": {"sites": nb[["lng_wgs84", "lat_wgs84", "daily_t", "district"]].values.tolist(),
               "facilities": nbd.assign(cap_t=nbd.total_capacity_kg / 1000.0)[
                   ["lng_wgs84", "lat_wgs84", "cap_t"]].values.tolist(),
               "payload": rl.tolist(),
               "center": [float(nbd.lng_wgs84.mean()), float(nbd.lat_wgs84.mean())]},
        "SX": {"facilities": sxd.assign(cap_t=sxd.xk / 1000.0)[
                   ["lng_wgs84", "lat_wgs84", "cap_t"]].values.tolist(),
               "payload": rl.tolist(),
               "center": [float(sxd.lng_wgs84.mean()), float(sxd.lat_wgs84.mean())]},
        "DQ": {"lngs": dq.lng_wgs84.values.tolist(), "payload": rl.tolist(),
               "center": [float(dq.lng_wgs84.mean()), 30.52]},
        "WZ": {"payload": rl.tolist(), "center": [120.65, 28.01]},
        "demand_mu": float(np.mean(np.log(nb_daily))),
        "demand_sd": float(np.std(np.log(nb_daily))),
        "hour_profile": hour_profile(),
    }


def obs_rng(name: str) -> random.Random:
    return random.Random(int(hashlib.sha256(("obs:" + name).encode()).hexdigest()[:8], 16))


class LatentWorld:
    def __init__(self, arc, n, m, seed, prof):
        rng = random.Random(seed)          # stream A: sole random source of the latent world
        p = prof[arc]
        mu, sd = prof["demand_mu"], prof["demand_sd"]

        if arc == "NB":
            idx = rng.sample(range(len(p["sites"])), min(n, len(p["sites"])))
            self.sites = [list(s) for s in (p["sites"][i] for i in idx)]
        elif arc == "SX":
            facs = p["facilities"]
            self.sites = []
            for _ in range(n):
                f = facs[rng.randrange(len(facs))]
                r = 0.03 if rng.random() < 0.7 else 0.14
                self.sites.append([f[0] + rng.uniform(-r, r), f[1] + rng.uniform(-r, r),
                                   float(rng.lognormvariate(mu, sd)), f"SX{rng.randrange(5):02d}"])
        elif arc == "DQ":
            self.sites = [[float(rng.choice(p["lngs"])),
                           30.52 + rng.uniform(-0.06, 0.06),
                           float(rng.lognormvariate(mu, sd)), f"DQ{rng.randrange(5):02d}"]
                          for _ in range(n)]
        else:
            cents = [[120.65, 28.01], [120.82, 27.98], [120.60, 27.97], [121.15, 27.84],
                     [120.68, 28.15], [120.55, 27.66], [120.96, 28.12]]
            self.sites = []
            for _ in range(n):
                c = cents[rng.randrange(len(cents))]
                self.sites.append([c[0] + rng.uniform(-0.04, 0.04), c[1] + rng.uniform(-0.04, 0.04),
                                   float(rng.lognormvariate(mu, sd)), f"WZ{rng.randrange(5):02d}"])

        # Facilities: sampled once, immutable IDs; capacity normalized to demand x PRESSURE (t)
        # DQ/WZ have no real facility pool -> synthesized inside the latent world (around the centroid, lognormal capacities)
        if "facilities" in p:
            idx = rng.sample(range(len(p["facilities"])), min(m, len(p["facilities"])))
            raw = [list(p["facilities"][i]) for i in idx]
            total_d = sum(s[2] for s in self.sites)
            scale = total_d * PRESSURE / sum(f[2] for f in raw)
            self.facilities = [[f"{arc}-F{k}", f[0], f[1], f[2] * scale]
                               for k, f in enumerate(raw)]
        else:
            cents = [[120.65, 28.01], [120.82, 27.98], [120.60, 27.97], [121.15, 27.84],
                     [120.68, 28.15], [120.55, 27.66], [120.96, 28.12]] if arc == "WZ" \
                else [[float(rng.choice(p.get("lngs", [119.95]))), 30.52 + 0.03 * i]
                      for i in range(5)]
            raw = []
            for _ in range(m):
                c = cents[rng.randrange(len(cents))]
                raw.append([c[0] + rng.uniform(-0.05, 0.05), c[1] + rng.uniform(-0.05, 0.05),
                            float(rng.lognormvariate(11.5, 0.8))])
            total_d = sum(s[2] for s in self.sites)
            scale = total_d * PRESSURE / sum(f[2] for f in raw)
            self.facilities = [[f"{arc}-F{k}", f[0], f[1], f[2] * scale]
                               for k, f in enumerate(raw)]

        # Trips: same-site splitting (demand t, service, window center)
        hp, w = zip(*prof["hour_profile"])
        self.trips = []
        for si, s in enumerate(self.sites):
            k = max(1, int(math.ceil(s[2] / TRIP_T)))
            for _ in range(k):
                self.trips.append([si, s[2] / k, rng.randint(10, 30),
                                   rng.choices(hp, weights=w)[0] * 60])

        self.payload_t = float(rng.choice(p["payload"]))
        self.mean_payload_t = float(np.mean(p["payload"]))

        byd = {}
        for s in self.sites:
            byd.setdefault(s[3], []).append((s[0], s[1]))
        self.district_centroid = {d: (sum(x for x, _ in xs) / len(xs),
                                      sum(y for _, y in xs) / len(xs))
                                  for d, xs in byd.items()}


def observe(world, name, arc, maturity, prof, q_factor=1.25):
    """Observation function v3: the latent world stores ~10 t demand atoms;
    loads are greedy-packed per site up to the OBSERVED fleet payload, so the
    fleet-information level acts through load granularity (rated payload ->
    ~16 t loads; carrier aggregate q_factor x mean -> ~20 t loads; the factor
    is calibrated to the carrier-level real-to-rated tonnage ratio in the
    registry records, see the manuscript)."""
    rng = obs_rng(name)                    # stream B: observation jitter only
    show_tw = maturity == "L3"
    real_coord = maturity in ("L3", "L2", "FL")
    vehicle_fleet = maturity in ("L3", "L2", "SP")

    veh_cap = round(world.payload_t if vehicle_fleet else world.mean_payload_t * q_factor, 2)

    atoms_by_site = {}
    for si, per, service, twc in world.trips:
        atoms_by_site.setdefault(si, []).append((per, service, twc))

    customers = []
    cid = 0
    for si, s in enumerate(world.sites):
        atoms = atoms_by_site.get(si, [])
        packs, acc_d, acc_s, acc_w = [], 0.0, 0.0, 0.0
        for per, service, twc in atoms:
            if acc_d > 0 and acc_d + per > veh_cap + 1e-9:
                packs.append((acc_d, acc_s, acc_w / acc_d))
                acc_d, acc_s, acc_w = 0.0, 0.0, 0.0
            acc_d += per
            acc_s += service
            acc_w += twc * per
        if acc_d > 0:
            packs.append((acc_d, acc_s, acc_w / acc_d))

        tau_min = min(_haversine_km(s[0], s[1], fx, fy) / AVG_SPEED * 60.0
                      for _fid, fx, fy, _fc in world.facilities)
        for demand, service, twcf in packs:
            if real_coord:
                x, y = s[0], s[1]
            else:
                cx, cy = world.district_centroid[s[3]]
                x, y = cx + rng.uniform(-0.01, 0.01), cy + rng.uniform(-0.01, 0.01)
            if show_tw:
                twc = int(round(twcf))
                e, l = max(DAY_START, twc - 150), min(DAY_END, twc + 150)
                e = max(e, int(DAY_START + tau_min))            # floor: loose
                l = min(l, int(math.ceil(DAY_END - service - tau_min)))  # ceil: loose
                if l - e < 120:
                    e, l = DAY_START, DAY_END
            else:
                e, l = DAY_START, DAY_END
            customers.append({"id": cid, "name": f"{name}-c{cid}", "x": round(x, 6),
                              "y": round(y, 6), "demand": round(demand, 3),
                              "earliest": int(e), "latest": int(l),
                              "service_time": int(service), "hotspot": "",
                              "hotspot_weight": 1.0,
                              "source_project": f"{arc}-latent"})
            cid += 1

    facilities = [{"id": fid, "name": fid, "x": round(fx, 6), "y": round(fy, 6),
                   "capacity": round(fc, 2),
                   "fixed_cost": round(2000 + 2000 * math.log10(max(fc / 100.0, 1.0)), 2),
                   "source_site": f"{arc}-latent", "site_weight": 1.0}
                  for fid, fx, fy, fc in world.facilities]

    return {"name": name,
            "description": (f"CDW-LIB v2 latent | archetype={arc} observation={maturity} "
                            f"| coord={'real' if real_coord else 'district'} "
                            f"fleet={'vehicle' if vehicle_fleet else 'company'} "
                            f"tw={show_tw} | payload={veh_cap}t(ratedload)"),
            "horizon_label": "single-day",
            "vehicle_capacity": veh_cap,
            "vehicle_fixed_cost": 500.0,
            "transport_cost_per_km": 10.0,
            "max_route_minutes": DAY_END - DAY_START,
            "day_start": DAY_START, "day_end": DAY_END,
            "average_speed_kmph": AVG_SPEED,
            "restricted_center": {"x": prof[arc]["center"][0], "y": prof[arc]["center"][1]},
            "restricted_radius_km": 0.0,
            "restricted_window": [DAY_START, DAY_END],
            "restricted_penalty_factor": 1.0,
            "customers": customers, "facilities": facilities}


CORE = [("SX", "L3"), ("SX", "L2"), ("SX", "L1"),
        ("NB", "L2"), ("NB", "L1"), ("DQ", "L1"), ("WZ", "L1")]
SIZE_SEEDS = {3: 2, 5: 3, 10: 5, 20: 5, 30: 2}
M_FAC = {3: 2, 5: 3, 10: 4, 20: 6, 30: 8}


def twins_of(arc_base):
    if arc_base == ("SX", "L3"):
        return ["L2", "L1"]
    if arc_base == ("NB", "L2"):
        return ["L1"]
    return []


def validate():
    """Pair invariants (v3).

    - All pairs: facility entities identical.
    - Same-fleet pairs (identical observed payload): load-level demands and
      service times identical element-by-element (windows/coords may differ).
    - Fleet-mixed pairs (any L1/FL side): total demand identical; site-level
      identity holds structurally because both instances derive from the
      same latent world (atoms invariant) and packing is deterministic.
    """
    fails = checked = 0
    pairs = [(("SX", "L3"), "L2", "load"), (("SX", "L3"), "L1", "site"),
             (("SX", "L2"), "L1", "site"), (("NB", "L2"), "L1", "site"),
             (("NB", "L2"), "SP", "load"), (("NB", "L2"), "FL", "site")]
    for (arc, base), d, mode in pairs:
        for size in SIZE_SEEDS:
            for s in range(1, SIZE_SEEDS[size] + 1):
                bp = OUT / f"{arc}-{base}-n{size}-s{s}.json"
                dp = (OUT / f"{arc}-{d}-n{size}-s{s}.json" if d in ("L1", "L2", "L3")
                      else OUT / f"NBd-{d}-n{size}-s{s}.json")
                if not (bp.exists() and dp.exists()):
                    continue
                b, dd = json.loads(bp.read_text()), json.loads(dp.read_text())
                checked += 1
                ok = b["facilities"] == dd["facilities"]
                if ok and mode == "load":
                    ok = ([c["demand"] for c in b["customers"]] ==
                          [c["demand"] for c in dd["customers"]]
                          and [c["service_time"] for c in b["customers"]] ==
                          [c["service_time"] for c in dd["customers"]]
                          and len(b["customers"]) == len(dd["customers"]))
                elif ok:  # site-level; tolerance covers per-load round(3)
                    ok = (abs(sum(c["demand"] for c in b["customers"])
                              - sum(c["demand"] for c in dd["customers"])) < 0.5)
                if not ok:
                    fails += 1
                    print(f"[FAIL] {bp.name} vs {dp.name}")
    print(f"pair invariants: {checked} pairs, {fails} failures {'OK' if fails == 0 else 'FAIL'}")
    return fails == 0


def main(q_factor=1.25, out_dir=None):
    out = Path(out_dir) if out_dir else OUT
    prof = build_pools()
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()

    worlds = {}

    def world_of(arc, n, s):
        key = (arc, n, s)
        if key not in worlds:
            worlds[key] = LatentWorld(arc, n, M_FAC[n], 1000 * n + s, prof)
        return worlds[key]

    rows = []

    def emit(name, w, arc, mat):
        inst = observe(w, name, arc, mat, prof, q_factor)
        (out / f"{name}.json").write_text(json.dumps(inst, ensure_ascii=False), encoding="utf-8")
        rows.append({"name": name, "archetype": arc, "maturity": mat,
                     "n": len(w.sites), "m": len(w.facilities), "n_trips": len(w.trips)})

    for arc, mat in CORE:
        for size, ns in SIZE_SEEDS.items():
            for s in range(1, ns + 1):
                emit(f"{arc}-{mat}-n{size}-s{s}", world_of(arc, size, s), arc, mat)
    for factor in ["SP", "FL"]:
        for size in [5, 10]:
            for s in range(1, 4):
                emit(f"NBd-{factor}-n{size}-s{s}", world_of("NB", size, s), "NB", factor)

    pd.DataFrame(rows).to_csv(out / "library_manifest.csv", index=False)
    # Capacity sanity gate (guards against unit bugs)
    bad = 0
    for r in rows:
        d = json.loads((out / f"{r['name']}.json").read_text())
        tot_d = sum(c["demand"] for c in d["customers"])
        tot_c = sum(f["capacity"] for f in d["facilities"])
        if abs(tot_c - PRESSURE * tot_d) > 0.02 * tot_d or min(f["capacity"] for f in d["facilities"]) <= 0:
            bad += 1
            print(f"[CAP-FAIL] {r['name']}: cap {tot_c:.1f} vs demand×{PRESSURE} {PRESSURE*tot_d:.1f}")
    print(f"capacity gate: {bad} failures {'OK' if bad == 0 else 'FAIL'}")
    print(f"family complete: {len(rows)} instances (core {sum(1 for r in rows if r['maturity'] in ('L1','L2','L3'))} + 12 variants)")
    validate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CDW-LIB latent-world generator")
    ap.add_argument("--validate", action="store_true",
                    help="re-verify the entity-invariance pairs on existing instances")
    ap.add_argument("--qfactor", type=float, default=1.25,
                    help="aggregate-payload factor for L1/FL packing (default 1.25, "
                         "calibrated to the carrier-level real/rated tonnage ratio)")
    ap.add_argument("--out", default=None, help="output directory (default: ../instances)")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        main(q_factor=args.qfactor, out_dir=args.out)
