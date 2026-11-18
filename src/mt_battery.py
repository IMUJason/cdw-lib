#!/usr/bin/env python3
"""Certified-interval battery for the multi-trip CDW-LIB family.

Per instance:
  UB = min(arc-MIP incumbent, polished incumbent chains)   [feasible]
  LB = time-expanded model dual bound (with cutoff at UB)  [valid]
Emits one JSON line per instance plus a summary CSV.

Run under the cplex310 environment:
  /opt/anaconda3/envs/cplex310/bin/python mt_battery.py [--ub=300] [--lb=300] \
      [--lp-only] INST [INST ...]
"""
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BenchmarkInstance  # noqa: E402
import importlib.util


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = name
    spec.loader.exec_module(mod)
    return mod


mt_prototype = _load("mt_prototype_mod", ROOT / "src" / "mt_prototype.py")
mt_lb = _load("mt_lb_mod", ROOT / "src" / "mt_lb.py")
mt_alns = _load("mt_alns_mod", ROOT / "src" / "mt_alns.py")


def run_instance(nm, ub_limit=300, lb_limit=300, lp_only=False):
    inst = BenchmarkInstance.load(ROOT / "instances" / f"{nm}.json")
    row = {"instance": nm, "loads": len(inst.customers)}

    # ---- UB: arc MIP + polish ----
    t0 = time.time()
    mip = mt_prototype.solve_multitrip(inst, time_limit=ub_limit)
    row["ub_mip"] = round(mip["objective"], 1)
    row["ub_mip_gap_pct"] = mip.get("gap_pct")
    ub, ub_src = mip["objective"], "mip"
    chains = mip.get("chains")
    if chains:
        pol, pol_cost = mt_alns.polish(inst, chains)
        if mt_alns.feasible(inst, pol) and pol_cost < ub - 1e-9:
            ub, ub_src = pol_cost, "polished"
    row["ub"] = round(ub, 1)
    row["ub_source"] = ub_src
    row["ub_sec"] = round(time.time() - t0, 1)

    # ---- LB: time-expanded dual bound with cutoff ----
    t0 = time.time()
    # cutoff with a safety margin: a tight cutoff can make the model
    # infeasible when its optimum ties the UB up to float error
    cut = ub + max(1.0, 0.001 * ub)
    lb = mt_lb.solve_lb(inst, time_limit=lb_limit, ub_cutoff=cut,
                        lp_only=lp_only)
    row["lb"] = lb.get("dual_bound_lb")
    if row["lb"] is None:
        row["lb_status"] = lb.get("status", "n/a")
        row["error"] = "lb produced no dual bound"
        return row
    row["lb_status"] = lb["status"]
    row["lb_sec"] = lb["sec"]
    row["certified_gap_pct"] = round((ub - row["lb"]) / row["lb"] * 100, 3)
    return row


def main():
    argv = sys.argv[1:]
    ub_limit = lb_limit = 300
    lp_only = False
    names = []
    k = 0
    while k < len(argv):
        a = argv[k]
        if a.startswith("--ub="):
            ub_limit = int(a.split("=")[1])
        elif a.startswith("--lb="):
            lb_limit = int(a.split("=")[1])
        elif a == "--lp-only":
            lp_only = True
        else:
            names.append(a)
        k += 1
    if not names:
        names = [f"{c}-n3-s{s}" for c in ("DQ-L1", "NB-L1", "NB-L2",
                                          "SX-L1", "SX-L2", "SX-L3", "WZ-L1")
                 for s in (1, 2)]
    rows = []
    for nm in names:
        try:
            r = run_instance(nm, ub_limit, lb_limit, lp_only)
        except Exception as exc:  # noqa: BLE001
            r = {"instance": nm, "error": repr(exc)}
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)

    out = ROOT / "results" / "mt_battery.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["instance", "loads", "lb", "ub", "certified_gap_pct", "ub_source",
            "ub_mip", "ub_mip_gap_pct", "lb_status", "ub_sec", "lb_sec"]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    ok = [r for r in rows if "certified_gap_pct" in r]
    if ok:
        print(f"\n== {len(ok)} instances certified, median gap "
              f"{sorted(x['certified_gap_pct'] for x in ok)[len(ok)//2]:.3f}%, "
              f"max {max(x['certified_gap_pct'] for x in ok):.3f}% -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
