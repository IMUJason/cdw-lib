#!/usr/bin/env python3
"""Time-expanded lower-bound model for the multi-trip CDW-LIB problem.

Provides a certified LOWER bound for the continuous-time problem:
- time discretized in DELTA-minute slots over [day_start, day_end];
- a load arc (f_base, t_dep) -> (f_dump, t_dump) exists under OPTIMISTIC
  rounding (departures/service/returns rounded to earlier slots), so every
  real feasible schedule embeds in the network at equal cost and the model's
  optimum is a valid lower bound on the true optimum;
- vehicle injection at facility slot 0 costs phi; waiting arcs are free;
- per-load cover = 1 (side constraint) and facility capacity with opening
  binaries complete the model.

Combined with an incumbent from mt_prototype.py / mt_alns.py, this yields a
certified optimality interval per instance.
"""
import json
import math
import sys
import time
from pathlib import Path

from docplex.mp.model import Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkInstance  # noqa: E402

DELTA = 5  # minutes per slot (overridable via --delta)


def solve_lb(inst: BenchmarkInstance, time_limit=600, ub_cutoff=None,
             extract=False, lp_only=False):
    loads = list(inst.customers)
    F = [f.id for f in inst.facilities]
    beta, phi = inst.transport_cost_per_km, inst.vehicle_fixed_cost
    t0, t1 = inst.day_start, inst.day_end
    S = int((t1 - t0) // DELTA)  # number of slots

    m = Model(name=f"lb_{inst.name}")
    m.parameters.timelimit = time_limit
    m.parameters.threads = 4
    bin_var = (lambda name: m.continuous_var(lb=0, ub=1, name=name)) if lp_only \
        else (lambda name: m.binary_var(name=name))
    int_var = (lambda lb, ub, name: m.continuous_var(lb=lb, ub=ub, name=name)) \
        if lp_only else (lambda lb, ub, name: m.integer_var(lb=lb, ub=ub, name=name))
    # cutoff at a known feasible UB: safe for the dual bound because
    # UB >= true optimum >= this model's optimum, so the cutoff never
    # removes the model optimum while pruning the search above it


    load_arcs = {}      # (i, fb, fd, t_dep, t_dump) -> var
    by_load = {c.id: [] for c in loads}
    by_dump = {f: [] for f in F}
    for c in loads:
        for fb in F:
            tr1 = inst.travel_time("f", fb, "c", c.id, 0)
            d1 = inst.distance("f", fb, "c", c.id)
            for fd in F:
                tr2 = inst.travel_time("c", c.id, "f", fd, 0)
                d2 = inst.distance("c", c.id, "f", fd)
                cost = beta * (d1 + d2)
                # latest departure slot so that service can still start by l_i
                t_dep_max = int((c.latest - tr1 - t0) // DELTA)
                for td in range(0, max(t_dep_max, 0) + 1):
                    dep = t0 + td * DELTA
                    start = max(dep + tr1, float(c.earliest))
                    if start > c.latest + 1e-6:
                        continue
                    ret = start + c.service_time + tr2
                    tr = min(int((ret - t0) // DELTA), S)
                    if tr < td:
                        continue
                    key = (c.id, fb, fd, td, tr)
                    if key in load_arcs:
                        continue
                    v = bin_var(f"a_{c.id}_{fb}_{fd}_{td}_{tr}")
                    load_arcs[key] = v
                    by_load[c.id].append((v, fd))
                    by_dump[fd].append((v, c.demand))

    wait = {}
    for f in F:
        for t in range(S):
            wait[(f, t)] = m.continuous_var(lb=0, ub=len(loads), name=f"w_{f}_{t}")
    veh = {f: int_var(0, len(loads), f"v_{f}") for f in F}
    y = {f: bin_var(f"y_{f}") for f in F}

    # flow conservation at (f, t); arcs bucketed by endpoints for O(1) lookup
    arr_by = {}
    dep_by = {}
    for k, v in load_arcs.items():
        i, fb, fd, td, tr = k
        arr_by.setdefault((fd, tr), []).append(v)
        dep_by.setdefault((fb, td), []).append(v)
    sink = {f: m.continuous_var(lb=0, ub=len(loads), name=f"snk_{f}") for f in F}
    for f in F:
        for t in range(S + 1):
            inflow = []
            outflow = []
            if t == 0:
                inflow.append(veh[f])
            else:
                inflow.append(wait[(f, t - 1)])
                inflow.extend(arr_by.get((f, t - 1), ()))
            if t == S:
                outflow.append(sink[f])
            else:
                outflow.append(wait[(f, t)])
                outflow.extend(dep_by.get((f, t), ()))
            if inflow or outflow:
                m.add_constraint(m.sum(inflow) == m.sum(outflow))

    for c in loads:
        m.add_constraint(m.sum(v for v, fd in by_load[c.id]) == 1)

    for f in F:
        cap = inst.facility_map[f].capacity
        terms = [q * v for v, q in by_dump[f]]
        if terms:
            m.add_constraint(m.sum(terms) <= cap * y[f])
        dumpvars = [v for v, q in by_dump[f]]
        if dumpvars:
            m.add_constraint(m.sum(dumpvars) <= len(dumpvars) * y[f])

    arc_cost = m.sum(
        beta * (inst.distance("f", k[1], "c", k[0])
                + inst.distance("c", k[0], "f", k[2])) * v
        for k, v in load_arcs.items())
    m.minimize(arc_cost + phi * m.sum(veh.values())
               + m.sum(inst.facility_map[f].fixed_cost * y[f] for f in F))
    if ub_cutoff is not None:
        m.add_constraint(arc_cost + phi * m.sum(veh.values())
                         + m.sum(inst.facility_map[f].fixed_cost * y[f] for f in F)
                         <= ub_cutoff + 1e-6)

    st = time.time()
    sol = m.solve()
    el = time.time() - st
    if sol is None:
        return {"instance": inst.name, "status": "NO_SOLUTION", "sec": round(el, 1)}

    chains = None
    if extract:
        chosen = {k: v for k, v in load_arcs.items()
                  if round(sol.get_value(v)) == 1}
        succ = {}
        start_at = {f: [] for f in F}
        for (i, fb, fd, td, tr) in chosen:
            if fb not in start_at:
                start_at[fb] = []
        # Chain reconstruction: greedily thread load arcs in (fb, td) time order
        remaining = list(chosen.keys())
        # Vehicle origin = smallest td whose (fb, td) node has no in-arc
        used_dep = {(k[1], k[3]) for k in chosen}
        chains = []
        # Greedy: repeatedly take the earliest departing arc from a free node
        while remaining:
            remaining.sort(key=lambda k: (k[3], k[4]))
            k0 = remaining[0]
            i, fb, fd, td, tr = k0
            chain = {"base": fb, "visits": [(i, fd)]}
            remaining.remove(k0)
            # Continue the chain from (fd, tr): find an arc with fb2=fd and td2>=tr
            cur_f, cur_t = fd, tr
            while True:
                nxt = None
                for k in remaining:
                    if k[1] == cur_f and k[3] >= cur_t:
                        if nxt is None or (k[3], k[4]) < (nxt[3], nxt[4]):
                            nxt = k
                if nxt is None:
                    break
                chain["visits"].append((nxt[0], nxt[2]))
                cur_f, cur_t = nxt[2], nxt[4]
                remaining.remove(nxt)
            chains.append(chain)
    gap = getattr(m.solve_details, "mip_relative_gap", -1)
    # the model's DUAL bound is the valid lower bound for the true problem;
    # its own incumbent (within the optimistic network) is not
    dual = getattr(m.solve_details, "best_bound", None)
    if dual is None or (isinstance(dual, float) and dual != dual):  # NaN
        dual = sol.objective_value  # valid when solved to optimality (LP)
    out = {"instance": inst.name, "slots": S, "arcs": len(load_arcs),
           "status": m.solve_details.status,
           "dual_bound_lb": round(dual, 1),
           "network_incumbent": round(sol.objective_value, 1),
           "gap_pct": round(gap * 100, 3) if gap >= 0 else None,
           "sec": round(el, 1)}
    if extract and chains is not None:
        out["chains_n"] = len(chains)
        out["chains"] = chains
        out["chosen_arcs"] = [k for k, v in load_arcs.items()
                              if round(sol.get_value(v)) == 1]
    return out


if __name__ == "__main__":
    names, delta = [], 5
    argv = sys.argv[1:]
    k = 0
    while k < len(argv):
        if argv[k].startswith("--delta="):
            delta = int(argv[k].split("=")[1]); k += 1
        else:
            names.append(argv[k]); k += 1
    DELTA = delta
    for nm in (names or ["NB-L2-n3-s1"]):
        inst = BenchmarkInstance.load(ROOT / "instances" / f"{nm}.json")
        print(json.dumps(solve_lb(inst, time_limit=600)), flush=True)
