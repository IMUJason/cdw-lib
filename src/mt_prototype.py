#!/usr/bin/env python3
"""Multi-trip (full-truckload shuttle) exact prototype for CDW-LIB.

Decision experiment for the Path-B rebuild. Model:
  vehicle chains: facility f -> load j (haul ~10 t) -> dump at f' -> load k -> ...
- one load per visit (each trip node ~9.8 t < Q both levels)
- arcs x[i,f,j]: after load i, dump at f, drive to load j
- start s[f,j]: vehicle based at f begins with load j; end e[i,f]: chain ends at f
- windows on service start; duty (first service start -> final return) <= T_max
- facility daily capacity and opening costs

Solved with docplex + CPLEX (conda env cplex310). Usage:
  /opt/anaconda3/envs/cplex310/bin/python mt_prototype.py NB-L2-n3-s1 [inst2 ...]
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


def greedy_multitrip(inst):
    """Greedy multi-trip solution; same cost semantics as the MIP (vehicles
    based at facilities: initial leg from base facility is charged)."""
    loads = sorted(inst.customers, key=lambda c: (c.latest, c.earliest))
    F = [f.id for f in inst.facilities]
    beta, phi, T = inst.transport_cost_per_km, inst.vehicle_fixed_cost, inst.max_route_minutes
    cap_left = {f.id: f.capacity for f in inst.facilities}
    assign = {}
    for c in loads:
        for f in sorted(F, key=lambda f: inst.distance("c", c.id, "f", f)):
            if cap_left[f] >= c.demand - 1e-9:
                assign[c.id] = f
                cap_left[f] -= c.demand
                break
    chains = []
    for c in loads:
        f = assign[c.id]
        haul_t = inst.travel_time("c", c.id, "f", f, 0)
        haul_d = inst.distance("c", c.id, "f", f)
        best, best_cost = None, float("inf")
        for ch in chains:
            travel_t = inst.travel_time("f", ch["cur_fac"], "c", c.id, 0)
            travel_d = inst.distance("f", ch["cur_fac"], "c", c.id)
            arrive = ch["cur_t"] + travel_t
            start = max(arrive, float(c.earliest))
            if start > c.latest + 1e-6:
                continue
            ret = start + c.service_time + haul_t
            if ret - ch["t0"] > T + 1e-6:
                continue
            if beta * travel_d < best_cost:
                best_cost = beta * travel_d
                best = (ch, travel_d, start, ret)
        if best:
            ch, td, st, ret = best
            ch["cost"] += beta * (td + haul_d)
            ch["cur_t"] = ret
            ch["cur_fac"] = f
            ch["visits"].append(c.id)
        else:
            base = min(F, key=lambda ff: inst.distance("f", ff, "c", c.id))
            start0 = max(float(inst.day_start)
                         + inst.travel_time("f", base, "c", c.id, 0),
                         float(c.earliest))
            ret0 = start0 + c.service_time + haul_t
            chains.append(dict(cur_fac=f, cur_t=ret0, t0=start0,
                               base=base,
                               cost=beta * (inst.distance("f", base, "c", c.id)
                                            + haul_d),
                               visits=[c.id]))
    haul_total = sum(beta * inst.distance("c", c.id, "f", assign[c.id]) for c in loads)
    open_f = set(assign.values())
    fac_cost = sum(inst.facility_map[f].fixed_cost for f in open_f)
    total = sum(ch["cost"] for ch in chains) + phi * len(chains) + fac_cost
    return dict(total=round(total), chains=chains, assign=assign, loads=loads,
                haul_total=haul_total)


def warm_start_solution(m, inst, greedy, x, s, e, y, F):
    """Translate the greedy chains into a MIP start (arcs may be missing from
    pruned x if the greedy leg violates the MIP pruning rule; then skip)."""
    try:
        sol = m.create_empty_solution()
        val = {}
        for ch in greedy["chains"]:
            first = ch["visits"][0]
            key = (ch["base"], first)
            if key not in s:
                return None
            val[s[key]] = 1
            for a, b in zip(ch["visits"], ch["visits"][1:]):
                k = (a, greedy["assign"][a], b)
                if k not in x:
                    return None
                val[x[k]] = 1
            last = ch["visits"][-1]
            k = (last, greedy["assign"][last])
            if k not in e:
                return None
            val[e[k]] = 1
        for f in F:
            if any(v for (i, ff), v in e.items() if ff == f and val.get(v)):
                val[y[f]] = 1
        for v, one in val.items():
            sol.add_var_value(v, one)
        m.add_mip_start(sol)
        return True
    except Exception:
        return None


def solve_multitrip(inst: BenchmarkInstance, time_limit=600):
    loads = list(inst.customers)
    n = len(loads)
    F = [f.id for f in inst.facilities]
    beta = inst.transport_cost_per_km
    phi = inst.vehicle_fixed_cost
    T = inst.max_route_minutes

    def dcf(c, f):
        return inst.distance("c", c, "f", f)

    def dfc(f, c):
        return inst.distance("f", f, "c", c)

    def tcf(c, f):
        return inst.travel_time("c", c, "f", f, 0)

    def tfc(f, c):
        return inst.travel_time("f", f, "c", c, 0)

    m = Model(name=f"mt_{inst.name}")
    m.parameters.timelimit = time_limit
    m.parameters.threads = 4

    # full-day windows make the window constraints integer-redundant, BUT
    # dropping u measurably weakens the LP propagation (tested: gap 1.05% ->
    # 2.08% on NB-L2-n3-s1), so u is kept unconditionally
    full_day = all(abs(c.earliest - inst.day_start) < 1e-9
                   and abs(c.latest - inst.day_end) < 1e-9 for c in loads)
    del full_day  # documented above; retained for the record

    # symmetry breaking: loads at the same site with identical service time
    # and demand are interchangeable; forbid arcs from higher to lower id
    # within each such group (any chain order can be relabeled)
    groups = {}
    for c in loads:
        key = (round(c.x, 6), round(c.y, 6), c.service_time, round(c.demand, 3))
        groups.setdefault(key, []).append(c.id)
    sym_forbidden = set()
    for ids in groups.values():
        ids.sort()
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                sym_forbidden.add((ids[b], ids[a]))

    # ---- arcs with time-window pruning ----
    x = {}
    for i in loads:
        for f in F:
            ready = i.earliest + i.service_time + tcf(i.id, f)
            for j in loads:
                if j.id == i.id:
                    continue
                if (i.id, j.id) in sym_forbidden:
                    continue
                if ready + tfc(f, j.id) <= j.latest + 1e-6:
                    x[(i.id, f, j.id)] = m.binary_var(name=f"x_{i.id}_{f}_{j.id}")
    s = {(f, j.id): m.binary_var(name=f"s_{f}_{j.id}") for f in F for j in loads}
    e = {(i.id, f): m.binary_var(name=f"e_{i.id}_{f}") for i in loads for f in F}
    y = {f: m.binary_var(name=f"y_{f}") for f in F}
    u = {i.id: m.continuous_var(lb=i.earliest, ub=i.latest, name=f"u_{i.id}")
         for i in loads}
    Tm = {i.id: m.continuous_var(lb=0, ub=T, name=f"T_{i.id}") for i in loads}

    out = {i.id: [] for i in loads}
    inn = {j.id: [] for j in loads}
    for (i, f, j), v in x.items():
        out[i].append(v)
        inn[j].append(v)
    for (f, j), v in s.items():
        inn[j].append(v)
    for (i, f), v in e.items():
        out[i].append(v)

    for L in loads:
        m.add_constraint(m.sum(out[L.id]) == 1)
        m.add_constraint(m.sum(inn[L.id]) == 1)

    for (i, f, j), v in x.items():
        li, lj = inst.customer_map[i], inst.customer_map[j]
        step = li.service_time + tcf(i, f) + tfc(f, j)
        # tight per-arc big-M: worst-case violation given var bounds
        M_u = max(0.0, (li.latest + step) - lj.earliest)
        m.add_constraint(u[j] >= u[i] + step - M_u * (1 - v))
        M_T = T + step
        m.add_constraint(Tm[j] >= Tm[i] + step - M_T * (1 - v))
    for (i, f), v in e.items():
        li = inst.customer_map[i]
        m.add_constraint(Tm[i] + li.service_time + tcf(i, f)
                         <= T + (T + li.service_time) * (1 - v))

    for f in F:
        cap = inst.facility_map[f].capacity
        dump_terms = []
        for i in loads:
            terms = [x[(i.id, f, j)] for (ii, ff, j) in x if ii == i.id and ff == f]
            if (i.id, f) in e:
                terms.append(e[(i.id, f)])
            if terms:
                dump_terms.append(i.demand * m.sum(terms))
        if dump_terms:
            m.add_constraint(m.sum(dump_terms) <= cap * y[f])
        thru = [v for (i, ff, j), v in x.items() if ff == f]
        thru += [v for (i, ff), v in e.items() if ff == f]
        thru += [v for (ff, j), v in s.items() if ff == f]
        if thru:
            m.add_constraint(m.sum(thru) <= len(thru) * y[f])

    cost = []
    for (i, f, j), v in x.items():
        cost.append((beta * dcf(i, f) + beta * dfc(f, j)) * v)
    for (f, j), v in s.items():
        cost.append(beta * dfc(f, j) * v)
    for (i, f), v in e.items():
        cost.append(beta * dcf(i, f) * v)
    V = m.sum(s.values())
    m.minimize(m.sum(cost) + phi * V
               + m.sum(inst.facility_map[f].fixed_cost * y[f] for f in F))

    # duty-time lower bound on the fleet. The duty clock runs from the first
    # service start, so each load contributes one out-leg and every load but
    # its chain's first contributes an in-leg. For V chains, at most V in-legs
    # (each <= max one-way travel) fall outside the clocks, hence the valid
    # bound V*(T + maxtau) >= sum(s_i + 2*tau_min,i).
    tau_min = {i.id: min(tfc(f, i.id) for f in F) for i in loads}
    lb_veh = math.ceil(sum(i.service_time + 2 * tau_min[i.id] for i in loads)
                       / (T + max(tau_min.values())) - 1e-9)
    m.add_constraint(V >= lb_veh)

    # warm start from the aligned greedy
    g = greedy_multitrip(inst)
    warm_start_solution(m, inst, g, x, s, e, y, F)

    t0 = time.time()
    sol = m.solve()
    elapsed = time.time() - t0
    if sol is None:
        return {"instance": inst.name, "status": "NO_SOLUTION", "sec": elapsed}

    Vv = sol.get_value(V)
    succ = {}
    dumps = {}
    for (i, f, j), v in x.items():
        if round(sol.get_value(v)) == 1:
            succ[i] = (f, j)
            dumps[i] = f
    for (i, f), v in e.items():
        if round(sol.get_value(v)) == 1:
            dumps[i] = f
    starts = {}
    for (f, j), v in s.items():
        if round(sol.get_value(v)) == 1:
            starts[j] = f
    lens = []
    chains = []
    for st_, base in starts.items():
        cur, ln = st_, 1
        visits = [(st_, dumps[st_])]
        while cur in succ:
            cur = succ[cur][1]
            visits.append((cur, dumps[cur]))
            ln += 1
        lens.append(ln)
        chains.append({"base": base, "visits": visits})
    return {
        "instance": inst.name, "loads": n, "facilities": len(F), "arcs": len(x),
        "status": m.solve_details.status, "objective": round(sol.objective_value, 1),
        "gap_pct": round(getattr(m.solve_details, "mip_relative_gap", -1) * 100, 3),
        "vehicles": int(round(Vv)),
        "open_facilities": sum(round(sol.get_value(y[f])) for f in F),
        "median_chain": sorted(lens)[len(lens) // 2] if lens else 0,
        "max_chain": max(lens) if lens else 0,
        "sec": round(elapsed, 1),
        "chains": chains,
    }


if __name__ == "__main__":
    argv = sys.argv[1:]
    tl, args, k = 600, [], 0
    while k < len(argv):
        a = argv[k]
        if a.startswith("--t"):
            if "=" in a:
                tl = int(a.split("=")[1])
            else:
                tl = int(argv[k + 1])
                k += 1
        else:
            args.append(a)
        k += 1
    for nm in (args or ["NB-L2-n3-s1", "SX-L3-n3-s1"]):
        inst = BenchmarkInstance.load(ROOT / "instances" / f"{nm}.json")
        r = solve_multitrip(inst, time_limit=tl)
        ch = r.pop("chains", None)
        if ch is not None:
            outp = ROOT / "results" / f"mt_ub_{nm}.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(
                {"instance": nm, "objective": r["objective"], "chains": ch}))
        print(json.dumps(r), flush=True)
