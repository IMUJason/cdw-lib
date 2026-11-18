#!/usr/bin/env python3
"""Multi-trip value-of-information (VSS) pipeline for CDW-LIB v3.

For each experimental pair (native level -> degraded level on a shared
latent world):
  1. solve BOTH instances with the multi-trip ALNS (seed 1, size schedule);
  2. take the degraded plan's chains and EXECUTE them on the native world:
     - true site coordinates (matters for L1/SP centroid plans);
     - true rated payload: a planned load heavier than Q_true is split into
       ceil(q / Q_true) co-located visits (extra round trips, possibly an
       extra vehicle when duty is exceeded);
     - true time windows (when the native level is L3): lateness is
       reported in minutes, not monetized;
  3. report in-model and execution cost changes against the native plan,
     plus overload tonnage and lateness.

The latent world (sites, demand atoms, facilities) is rebuilt
deterministically from the released generator, so the atom identity behind
every load is exact.

Usage (cplex310 python not required):
  python3 mt_vss.py [--iters-sched] PAIR-SPEC ...   # e.g. SX L3 L2 n5 s1
  python3 mt_vss.py --all        # run the full 80-pair grid
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
GEN = ROOT / "src"

RES = ROOT / "results"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    spec.loader.exec_module(mod)
    return mod


mt_alns = _load("mta", ROOT / "src" / "mt_alns.py")
from common import BenchmarkInstance  # noqa: E402

ITERS_BY_N = {3: 200, 5: 200, 10: 150, 20: 60, 30: 30}
SIZE_SEEDS = {3: 2, 5: 3, 10: 5, 20: 5, 30: 2}
M_FAC = {3: 2, 5: 3, 10: 4, 20: 6, 30: 8}
PAIRS = [("SX", "L3", "L2"), ("SX", "L3", "L1"), ("SX", "L2", "L1"),
         ("NB", "L2", "L1"), ("NB", "L2", "SP"), ("NB", "L2", "FL")]


def inst_name(arc, lvl, n, s):
    return f"{arc}-{lvl}-n{n}-s{s}" if lvl in ("L1", "L2", "L3") \
        else f"NBd-{lvl}-n{n}-s{s}"


class World:
    """Deterministic rebuild of a latent world plus per-level packing."""

    def __init__(self, gen, arc, n, s, prof):
        self.w = gen.LatentWorld(arc, n, M_FAC[n], 1000 * n + s, prof)
        self.sites = self.w.sites
        self.fac = self.w.facilities
        self.Q_true = round(self.w.payload_t, 2)
        self.Q_agg = round(self.w.mean_payload_t * 1.25, 2)

    def packs(self, q_view):
        """Greedy packs per site at payload q_view; returns list of
        (site_idx, [(tonnage, service, omega), ...]) in instance order."""
        out = []
        for si, s in enumerate(self.sites):
            atoms = [(per, svc, om) for (s_, per, svc, om) in self.w.trips
                     if s_ == si]
            cur, cur_s, cur_w, cur_q = [], 0.0, 0.0, 0.0
            for per, svc, om in atoms:
                if cur_q > 0 and cur_q + per > q_view + 1e-9:
                    out.append((si, cur))
                    cur, cur_s, cur_w, cur_q = [], 0.0, 0.0, 0.0
                cur.append((per, svc, om))
                cur_q += per
            if cur:
                out.append((si, cur))
        return out


def exec_visits(pack_atoms, q_true):
    """Split a planned load's atoms into true-payload visits."""
    visits, cur, cur_q = [], [], 0.0
    for per, svc, om in pack_atoms:
        if cur_q > 0 and cur_q + per > q_true + 1e-9:
            visits.append(cur)
            cur, cur_q = [], 0.0
        cur.append((per, svc, om))
        cur_q += per
    if cur:
        visits.append(cur)
    return visits


def exec_chain(inst_true, chain, site_of, q_true, windows):
    """Execute one planned chain on true geometry/payload/windows.
    Returns (exec_cost_excl_vehicle, n_extra_vehicles, lateness_min,
    overload_t) or None if infeasible beyond repair (never: splits absorb
    overload; duty overflow spawns extra vehicles)."""
    beta = inst_true.transport_cost_per_km
    lateness = 0.0
    overload = 0.0
    cost = 0.0
    extra_veh = 0
    # duty clock; if exceeded, open a new vehicle at the same base
    t_cur = None
    t0 = None
    prev_fac = chain["base"]
    # first leg from base to first visit's site
    first = True
    duty = 0.0

    def leg_dist(a_kind, a_id, b_kind, b_id):
        return inst_true.distance(a_kind, a_id, b_kind, b_id)

    def leg_time(a_kind, a_id, b_kind, b_id):
        return inst_true.travel_time(a_kind, a_id, b_kind, b_id, 0)

    # build the execution visit sequence
    seq = []
    for (load, fac) in chain["visits"]:
        for sub in exec_visits(site_of[load], q_true):
            ton = sum(p for p, _, _ in sub)
            svc = sum(sv for _, sv, _ in sub)
            om = sum(p * w for p, _, w in sub) / ton
            seq.append((site_of[load], ton, svc, om, fac))
    cur_t = t0 = None
    prev = ("f", chain["base"])
    for (site, ton, svc, om, fac) in seq:
        if ton > q_true + 1e-9:
            overload += ton - q_true
        # travel prev -> site
        d = leg_dist(prev[0], prev[1], "c", site)
        tt = leg_time(prev[0], prev[1], "c", site)
        cost += beta * d
        arrive = (cur_t if cur_t is not None
                  else float(inst_true.day_start)) + tt
        if t0 is None:
            t0 = max(arrive, float(inst_true.day_start))
            cur_t = t0
        start = max(arrive, float(inst_true.day_start)) if cur_t is None \
            else arrive
        start = max(start, cur_t)
        e, l = windows.get(site, (inst_true.day_start, inst_true.day_end))
        start = max(start, float(e))
        lateness += max(0.0, start - float(l))
        # haul site -> dump
        d2 = leg_dist("c", site, "f", fac)
        cost += beta * d2
        cur_t = start + svc + leg_time("c", site, "f", fac)
        prev = ("f", fac)
        if cur_t - t0 > inst_true.max_route_minutes + 1e-6:
            # duty exceeded: spawn an extra vehicle for the remainder
            extra_veh += 1
            t0 = None
            cur_t = None
            prev = ("f", fac)
    return cost, extra_veh, lateness, overload


def run_pair(gen, prof_cache, arc, base_lvl, deg_lvl, n, s, iters=None):
    prof = prof_cache
    w = World(gen, arc, n, s, prof)
    q_native = w.Q_true if deg_lvl in ("L2", "SP") else w.Q_true
    # native plan
    nname = inst_name(arc, base_lvl, n, s)
    dname = inst_name(arc, deg_lvl, n, s)
    ninst = BenchmarkInstance.load(ROOT / "instances" / f"{nname}.json")
    dinst = BenchmarkInstance.load(ROOT / "instances" / f"{dname}.json")
    it = iters or ITERS_BY_N[n]
    t0 = time.time()
    rnat = mt_alns.solve(ninst, seed=1, iterations=it)
    rdeg = mt_alns.solve(dinst, seed=1, iterations=it)
    if rnat.get("feasible") is not True or rdeg.get("feasible") is not True:
        return {"pair": f"{nname}->{dname}", "status": "SOLVE_FAIL"}
    # atom identity per degraded load
    q_deg = w.Q_agg if deg_lvl in ("L1", "FL") else w.Q_true
    packs = w.packs(q_deg)
    # degraded instance customers follow the same order as packs
    site_coords = {(round(x, 6), round(y, 6)): i
                   for i, (x, y, d, dist) in enumerate(w.sites)}
    # synthetic pseudo-site ids for common.py: reuse customer ids of native
    # instance at the same coordinates (native loads are co-located at sites)
    # build a pseudo instance on TRUE geometry: use the NATIVE instance for
    # distances/times (its customers sit on true site coords). Map each site
    # to one native customer id standing at it.
    site_anchor = {}
    for c in ninst.customers:
        key = (round(c.x, 6), round(c.y, 6))
        site_anchor.setdefault(key, c.id)
    anchor_used = {}
    # site index -> native customer id (first customer at that site)
    site_id = []
    for i, (x, y, d, dist) in enumerate(w.sites):
        aid = site_anchor.get((round(x, 6), round(y, 6)))
        site_id.append(aid if aid is not None else -1)
    # windows from native (L3 calibrated, else full day)
    windows = {}
    if base_lvl == "L3":
        for c in ninst.customers:
            key = (round(c.x, 6), round(c.y, 6))
            windows_site = windows.setdefault(
                site_anchor.get(key, c.id), (c.earliest, c.latest))
    # degraded loads -> atoms
    load_atoms = {}
    custs = dinst.customers
    if len(custs) != len(packs):
        return {"pair": f"{nname}->{dname}", "status": "PACK_MISMATCH",
                "detail": f"{len(custs)} vs {len(packs)}"}
    for c, (si, atoms) in zip(custs, packs):
        load_atoms[c.id] = atoms
        # site_of maps load -> (site anchor id, atoms)
    site_of = {c.id: packs[i][1] for i, c in enumerate(custs)}
    site_of_anchor = {c.id: site_id[packs[i][0]] for i, c in enumerate(custs)}

    # execute each degraded chain with TRUE anchor coordinates
    total_cost = 0.0
    extra_veh = 0
    lateness = 0.0
    overload = 0.0
    open_facs = set()
    for ch in rdeg["chains"]:
        ec, ev, la, ov = exec_chain_anchor(
            ninst, ch, site_of, site_of_anchor, w.Q_true, windows)
        total_cost += ec
        extra_veh += ev
        lateness += la
        overload += ov
        for _, f in ch["visits"]:
            open_facs.add(f)
    exec_cost = (total_cost
                 + ninst.vehicle_fixed_cost * (len(rdeg["chains"]) + extra_veh)
                 + sum(ninst.facility_map[f].fixed_cost for f in open_facs))
    nat_cost = rnat["objective"]
    deg_cost = rdeg["objective"]
    return {
        "pair": f"{nname}->{dname}", "status": "OK",
        "inmodel_pct": round((deg_cost - nat_cost) / nat_cost * 100, 2),
        "exec_pct": round((exec_cost - nat_cost) / nat_cost * 100, 2),
        "overload_t": round(overload, 1),
        "lateness_min": round(lateness, 1),
        "extra_veh": extra_veh,
        "sec": round(time.time() - t0, 1),
    }


def exec_chain_anchor(inst, chain, site_of, site_anchor, q_true, windows):
    """Execution with true geometry via anchor customer ids (see run_pair)."""
    beta = inst.transport_cost_per_km
    seq = []
    for (load, fac) in chain["visits"]:
        for sub in exec_visits(site_of[load], q_true):
            ton = sum(p for p, _, _ in sub)
            svc = sum(sv for _, sv, _ in sub)
            seq.append((site_anchor[load], ton, svc, fac))
    cost = 0.0
    lateness = 0.0
    overload = 0.0
    extra_veh = 0
    prev = ("f", chain["base"])
    t0 = cur_t = None
    for (site, ton, svc, fac) in seq:
        if site < 0:
            continue
        if ton > q_true + 1e-9:
            overload += ton - q_true
        d = inst.distance(prev[0], prev[1], "c", site)
        cost += beta * d
        tt = inst.travel_time(prev[0], prev[1], "c", site, 0)
        if cur_t is None:
            arrive = float(inst.day_start) + tt
            t0 = arrive
            start = arrive
        else:
            arrive = cur_t + tt
            start = arrive
        e, l = windows.get(site, (inst.day_start, inst.day_end))
        start = max(start, float(e))
        lateness += max(0.0, start - float(l))
        d2 = inst.distance("c", site, "f", fac)
        cost += beta * d2
        cur_t = start + svc + inst.travel_time("c", site, "f", fac, 0)
        prev = ("f", fac)
        if cur_t - t0 > inst.max_route_minutes + 1e-6:
            extra_veh += 1
            cur_t = t0 = None
    return cost, extra_veh, lateness, overload


def main():
    argv = [a for a in sys.argv[1:]]
    gen = _load("gen", GEN / "cdwlib_gen.py")
    prof = gen.build_pools()
    jobs = []
    if "--all" in argv:
        for arc, b, d in PAIRS:
            for n, ns in SIZE_SEEDS.items():
                for s in range(1, ns + 1):
                    if d in ("SP", "FL") and (n not in (5, 10) or s > 3):
                        continue
                    jobs.append((arc, b, d, n, s))
    else:
        # positional: arc base deg n s
        arc, b, d, n, s = argv[:5]
        jobs.append((arc, b, d, int(n), int(s)))
    out = RES / "vss_mt_results.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        for (arc, b, d, n, s) in jobs:
            r = run_pair(gen, prof, arc, b, d, n, s)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
