#!/usr/bin/env python3
"""Multi-trip ALNS for CDW-LIB (Path-B rebuild).

Solution = list of vehicle chains. Each chain: a base facility, an ordered
list of loads; consecutive loads are linked by "site -> dump -> next site"
legs. Cost semantics match mt_prototype.py exactly:
  beta * (sum of all legs) + phi * (number of chains)
  + sum of fixed costs of facilities that receive any dump.

Feasibility: per-load service windows, per-chain duty <= T_max, per-facility
daily capacity. One load per visit (each trip node ~10 t < Q).
"""
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BenchmarkInstance  # noqa: E402

DESTROY_OPS = ["random", "expensive", "related", "chain"]
REPAIR_OPS = ["greedy", "regret"]


# ---------------------------------------------------------------- solution --
def chain_times(inst, chain):
    """Service starts and duty for a chain; None if infeasible."""
    if not chain["visits"]:
        return (0.0, 0.0, 0.0)
    t0 = None
    t = prev = None
    for k, (load, fac) in enumerate(chain["visits"]):
        if k == 0:
            arrive = inst.day_start + inst.travel_time("f", chain["base"], "c", load, 0)
            t = max(arrive, inst.customer_map[load].earliest)
            t0 = t
        else:
            pl, pf = chain["visits"][k - 1]
            arrive = (t + inst.customer_map[pl].service_time
                      + inst.travel_time("c", pl, "f", pf, 0)
                      + inst.travel_time("f", fac, "c", load, 0))
            t = max(arrive, inst.customer_map[load].earliest)
            if t > inst.customer_map[load].latest + 1e-6:
                return None
        if t > inst.customer_map[load].latest + 1e-6:
            return None
        prev = t
    ll, lf = chain["visits"][-1]
    duty = (prev + inst.customer_map[ll].service_time
            + inst.travel_time("c", ll, "f", lf, 0) - t0)
    if duty > inst.max_route_minutes + 1e-6:
        return None
    return t0, prev, duty


def chain_cost(inst, chain):
    beta = inst.transport_cost_per_km
    if not chain["visits"]:
        return 0.0
    c = beta * inst.distance("f", chain["base"], "c", chain["visits"][0][0])
    for k in range(len(chain["visits"])):
        load, fac = chain["visits"][k]
        c += beta * inst.distance("c", load, "f", fac)
        if k + 1 < len(chain["visits"]):
            nxt = chain["visits"][k + 1][0]
            c += beta * inst.distance("f", fac, "c", nxt)
    return c


def solution_cost(inst, chains):
    facs = {f for ch in chains for _, f in ch["visits"]}
    return (sum(chain_cost(inst, ch) for ch in chains)
            + inst.vehicle_fixed_cost * len(chains)
            + sum(inst.facility_map[f].fixed_cost for f in facs))


def facility_loads(chains):
    loads = {}
    for ch in chains:
        for _, f in ch["visits"]:
            loads[f] = loads.get(f, 0) + 1
    return loads


def facility_tonnage(inst, chains):
    ton = {}
    for ch in chains:
        for load, f in ch["visits"]:
            q = inst.customer_map[load].demand
            ton[f] = ton.get(f, 0.0) + q
    return ton


def feasible(inst, chains):
    ton = facility_tonnage(inst, chains)
    for f, t in ton.items():
        if t > inst.facility_map[f].capacity + 1e-6:
            return False
    served = [l for ch in chains for l, _ in ch["visits"]]
    if sorted(served) != sorted(c.id for c in inst.customers):
        return False
    return all(chain_times(inst, ch) is not None for ch in chains)


# ------------------------------------------------------------------ repair --
def chain_starts(inst, chain):
    """Service starts + duty for a chain; None if infeasible.
    Returns (t0, [t_1..t_k], duty_end)."""
    if not chain["visits"]:
        return (0.0, [], 0.0)
    starts = []
    t = t0 = None
    for k, (load, fac) in enumerate(chain["visits"]):
        c = inst.customer_map[load]
        if k == 0:
            arrive = inst.day_start + inst.travel_time("f", chain["base"], "c", load, 0)
            t = max(arrive, float(c.earliest))
            t0 = t
        else:
            pl, pf = chain["visits"][k - 1]
            arrive = (starts[-1] + inst.customer_map[pl].service_time
                      + inst.travel_time("c", pl, "f", pf, 0)
                      + inst.travel_time("f", fac, "c", load, 0))
            t = max(arrive, float(c.earliest))
        if t > c.latest + 1e-6:
            return None
        starts.append(t)
    ll, lf = chain["visits"][-1]
    duty_end = (starts[-1] + inst.customer_map[ll].service_time
                + inst.travel_time("c", ll, "f", lf, 0))
    if duty_end - t0 > inst.max_route_minutes + 1e-6:
        return None
    return (t0, starts, duty_end)


def _insert_ok(inst, chain, info, pos, load, fac):
    """Feasibility + cost delta of inserting (load, fac) at pos (incremental,
    suffix replay with early exit). Returns (delta_cost, new_info) or None."""
    t0, starts, _ = info
    visits = chain["visits"]
    q = inst.customer_map[load]
    beta = inst.transport_cost_per_km
    # --- cost delta (exact, O(1)) ---
    prev_fac = chain["base"] if pos == 0 else visits[pos - 1][1]
    next_load = visits[pos][0] if pos < len(visits) else None
    d_add = inst.distance("f", prev_fac, "c", load) + inst.distance("c", load, "f", fac)
    d_rem = 0.0
    if pos == 0 and visits:
        d_rem += inst.distance("f", chain["base"], "c", visits[0][0])
    if next_load is not None:
        d_add += inst.distance("f", fac, "c", next_load)
        d_rem += inst.distance("f", prev_fac, "c", next_load)
    elif pos == len(visits) and visits:
        pass
    delta = beta * (d_add - d_rem)
    # --- time replay from pos ---
    new_starts = list(starts)
    t_prev_end = None
    if pos == 0:
        arrive = inst.day_start + inst.travel_time("f", chain["base"], "c", load, 0)
        t = max(arrive, float(q.earliest))
    else:
        pl, pf = visits[pos - 1]
        arrive = (starts[pos - 1] + inst.customer_map[pl].service_time
                  + inst.travel_time("c", pl, "f", pf, 0)
                  + inst.travel_time("f", fac, "c", load, 0))
        t = max(arrive, float(q.earliest))
    if t > q.latest + 1e-6:
        return None
    new_starts.insert(pos, t)
    cur_t, cur_load, cur_fac = t, load, fac
    for k in range(pos, len(visits)):
        nload, nfac = visits[k]
        nc = inst.customer_map[nload]
        arrive = (cur_t + inst.customer_map[cur_load].service_time
                  + inst.travel_time("c", cur_load, "f", cur_fac, 0)
                  + inst.travel_time("f", nfac, "c", nload, 0))
        tk = max(arrive, float(nc.earliest))
        if tk > nc.latest + 1e-6:
            return None
        new_starts[k + 1] = tk
        cur_t, cur_load, cur_fac = tk, nload, nfac
    duty_end = (cur_t + inst.customer_map[cur_load].service_time
                + inst.travel_time("c", cur_load, "f", cur_fac, 0))
    t0_new = new_starts[0]
    if duty_end - t0_new > inst.max_route_minutes + 1e-6:
        return None
    return delta, (t0_new, new_starts, duty_end)


def insertion_candidates(inst, chains, load, ton, limit=None):
    """Feasible (chain, position, facility) insertions, cheapest first
    (incremental evaluation, no candidate restriction)."""
    cands = []
    for ci, ch in enumerate(chains):
        info = chain_starts(inst, ch)
        if info is None:
            continue
        for pos in range(len(ch["visits"]) + 1):
            best = None
            for f in inst.facilities:
                if ton.get(f.id, 0) + inst.customer_map[load].demand > f.capacity + 1e-6:
                    continue
                r = _insert_ok(inst, ch, info, pos, load, f.id)
                if r is None:
                    continue
                if best is None or r[0] < best[0]:
                    best = (r[0], f.id)
            if best is not None:
                cands.append((best[0], ci, pos, best[1]))
        if limit and len(cands) >= limit:
            break
    cands.sort(key=lambda z: z[0])
    return cands


def new_chain_candidates(inst, load, ton):
    out = []
    for f in inst.facilities:
        if ton.get(f.id, 0) + inst.customer_map[load].demand > f.capacity + 1e-6:
            continue
        ch = {"base": f.id, "visits": [(load, f.id)]}
        if chain_times(inst, ch) is not None:
            out.append((inst.vehicle_fixed_cost
                        + 2 * inst.transport_cost_per_km
                        * inst.distance("f", f.id, "c", load), f.id))
    out.sort(key=lambda z: z[0])
    return out


def greedy_repair(inst, chains, removed, ton, rng):
    for load in removed:
        cands = insertion_candidates(inst, chains, load, ton)
        if cands:
            cost, ci, pos, f = cands[0]
            chains[ci]["visits"].insert(pos, (load, f))
            ton[f] = ton.get(f, 0) + inst.customer_map[load].demand
        else:
            nc = new_chain_candidates(inst, load, ton)
            if not nc:
                return None
            chains.append({"base": nc[0][1], "visits": [(load, nc[0][1])]})
            ton[nc[0][1]] = ton.get(nc[0][1], 0) + inst.customer_map[load].demand
    return chains


def regret_repair(inst, chains, removed, ton, rng):
    remaining = list(removed)
    while remaining:
        best_pick = None
        for load in remaining:
            cands = insertion_candidates(inst, chains, load, ton)
            regret = -1e18
            pick = None
            if cands:
                regret = cands[1][0] - cands[0][0] if len(cands) > 1 else 1e9
                pick = ("ins", cands[0])
            else:
                nc = new_chain_candidates(inst, load, ton)
                if not nc:
                    return None
                pick = ("new", nc[0])
                regret = 1e9
            if best_pick is None or regret > best_pick[0]:
                best_pick = (regret, load, pick)
        _, load, pick = best_pick
        if pick[0] == "ins":
            _, ci, pos, f = pick[1]
            chains[ci]["visits"].insert(pos, (load, f))
            ton[f] = ton.get(f, 0) + inst.customer_map[load].demand
        else:
            chains.append({"base": pick[1][1], "visits": [(load, pick[1][1])]})
            ton[pick[1][1]] = ton.get(pick[1][1], 0) + inst.customer_map[load].demand
        remaining.remove(load)
    return chains


# ---------------------------------------------------------------- destroy --
def remove_loads(inst, chains, ids):
    """Remove loads; cascade head-popping keeps every remaining chain
    feasible. Removing a chain's head changes the base leg for the new
    head, which can violate its window even though the old chain was
    feasible; such heads join the removed pool. (Successor arrivals are
    unaffected by removals: each element's arrival depends only on its
    predecessor's service start and its own dump facility.)"""
    gone = set(ids)
    out = []
    for ch in chains:
        visits = [(l, f) for l, f in ch["visits"] if l not in gone]
        ch2 = {"base": ch["base"], "visits": visits}
        while ch2["visits"] and chain_times(inst, ch2) is None:
            gone.add(ch2["visits"][0][0])
            del ch2["visits"][0]
        if ch2["visits"]:
            out.append(ch2)
    return out, sorted(gone)


def destroy_random(inst, chains, q, rng):
    ids = [c.id for c in inst.customers]
    victims = rng.sample(ids, min(q, len(ids)))
    return remove_loads(inst, chains, victims)


def destroy_expensive(inst, chains, q, rng):
    marg = []
    for ci, ch in enumerate(chains):
        full = chain_cost(inst, ch)
        for k, (load, f) in enumerate(ch["visits"]):
            rest = {"base": ch["base"],
                    "visits": ch["visits"][:k] + ch["visits"][k + 1:]}
            red = chain_cost(inst, rest) if rest["visits"] else 0.0
            marg.append((inst.vehicle_fixed_cost if len(ch["visits"]) == 1 else 0.0
                         + full - red, load))
    marg.sort(reverse=True)
    victims = [l for _, l in marg[:q]]
    return remove_loads(inst, chains, victims)


def destroy_related(inst, chains, q, rng):
    sites = {}
    for c in inst.customers:
        sites.setdefault((round(c.x, 6), round(c.y, 6)), []).append(c.id)
    keys = list(sites)
    k = rng.choice(keys)
    pool = sites[k][:]
    while len(pool) < q and len(keys) > 1:
        keys.remove(k)
        k = rng.choice(keys)
        pool += sites[k]
    victims = pool[:q]
    return remove_loads(inst, chains, victims)


def destroy_chain(inst, chains, q, rng):
    idx = rng.randrange(len(chains))
    victims = [l for l, _ in chains[idx]["visits"]]
    return remove_loads(inst, chains, victims)


DESTROY = {"random": destroy_random, "expensive": destroy_expensive,
           "related": destroy_related, "chain": destroy_chain}
REPAIR = {"greedy": greedy_repair, "regret": regret_repair}




# ------------------------------------------------------------- local search --
def _try_relocate(inst, chains, ton):
    """First-improvement relocation with delta evaluation."""
    if ton is None:
        ton = facility_tonnage(inst, chains)
    facs_open = {f for ch in chains for _, f in ch["visits"]}
    fix = {f: inst.facility_map[f].fixed_cost for f in facs_open}
    for ci in range(len(chains)):
        if not chains[ci]["visits"]:
            continue
        for k in range(len(chains[ci]["visits"])):
            load, old_f = chains[ci]["visits"][k]
            ch_i = chains[ci]
            rest_i = ch_i["visits"][:k] + ch_i["visits"][k + 1:]
            cost_rest_i = chain_cost(inst, {"base": ch_i["base"], "visits": rest_i}) if rest_i else 0.0
            cost_old_i = chain_cost(inst, ch_i)
            if not rest_i:
                rest_i = None
            for cj, ch_j in enumerate(chains):
                if cj == ci:
                    continue
                for pos in range(len(ch_j["visits"]) + 1):
                    for f in inst.facilities:
                        q = inst.customer_map[load].demand
                        new_ton_f = ton.get(f.id, 0) + (0 if f.id == old_f else q)
                        if new_ton_f > f.capacity + 1e-6:
                            continue
                        trial_j = ch_j["visits"][:pos] + [(load, f.id)] + ch_j["visits"][pos:]
                        newch = {"base": ch_j["base"], "visits": trial_j}
                        tm = chain_times(inst, newch)
                        if tm is None:
                            continue
                        if rest_i is not None and chain_times(
                                inst, {"base": ch_i["base"], "visits": rest_i}) is None:
                            continue
                        d_chain = (chain_cost(inst, newch) - chain_cost(inst, ch_j)
                                   + cost_rest_i - cost_old_i)
                        d_veh = -inst.vehicle_fixed_cost if rest_i is None else 0.0
                        d_fac = 0.0
                        if old_f not in {fd for _, fd in (rest_i or [])} \
                                and old_f not in {fd for ch in chains if ch is not ch_i for _, fd in ch["visits"]}:
                            d_fac -= inst.facility_map[old_f].fixed_cost
                        if f.id not in facs_open:
                            d_fac += f.fixed_cost
                        if d_chain + d_veh + d_fac < -1e-9:
                            cand = [dict(base=c["base"], visits=list(c["visits"]))
                                    for c in chains]
                            if rest_i is None:
                                cand.pop(ci)
                            else:
                                cand[ci]["visits"] = rest_i
                            # cj index may shift after pop
                            tgt = cand[cj - 1] if rest_i is None and cj > ci else cand[cj]
                            tgt["visits"].insert(pos, (load, f.id))
                            return cand, solution_cost(inst, cand)
    return None, None


def _try_merge(inst, chains):
    """Try merging two chains (concatenate, continuous-check); keep best cut."""
    base_total = solution_cost(inst, chains)
    n = len(chains)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            merged_visits = chains[a]["visits"] + chains[b]["visits"]
            for cut in range(0, len(merged_visits) + 1):
                v1, v2 = merged_visits[:cut], merged_visits[cut:]
                c1 = dict(base=chains[a]["base"], visits=v1)
                c2 = dict(base=chains[b]["base"], visits=v2)
                if chain_times(inst, c1) is None or chain_times(inst, c2) is None:
                    continue
                cand = [dict(base=c["base"], visits=list(c["visits"]))
                        for i, c in enumerate(chains) if i not in (a, b)]
                cand += [c for c in (c1, c2) if c["visits"]]
                tot = solution_cost(inst, cand)
                if tot < base_total - 1e-9:
                    return cand, tot
    return None, None


def polish(inst, chains, rounds=12):
    """Iterated relocate + merge until no improvement."""
    best = [dict(base=c["base"], visits=list(c["visits"])) for c in chains]
    best_cost = solution_cost(inst, best)
    for _ in range(rounds):
        improved = False
        cand, tot = _try_relocate(inst, best, None)
        if cand is not None:
            best, best_cost, improved = cand, tot, True
            continue
        cand, tot = _try_merge(inst, best)
        if cand is not None:
            best, best_cost, improved = cand, tot, True
            continue
        if not improved:
            break
    return best, best_cost

# ------------------------------------------------------------------ solve ---
def initial_solution(inst, rng):
    loads = sorted(inst.customers, key=lambda c: (c.latest, c.earliest, rng.random()))
    chains = []
    ton = {}
    for c in loads:
        placed = False
        cands = insertion_candidates(inst, chains, c.id, ton)
        if cands:
            _, ci, pos, fid = cands[0]
            chains[ci]["visits"].insert(pos, (c.id, fid))
            ton[fid] = ton.get(fid, 0) + c.demand
            placed = True
        if not placed:
            best = None
            for f in inst.facilities:
                if ton.get(f.id, 0) + c.demand > f.capacity + 1e-6:
                    continue
                ch = {"base": f.id, "visits": [(c.id, f.id)]}
                if chain_times(inst, ch) is not None:
                    key = inst.distance("f", f.id, "c", c.id)
                    if best is None or key < best[0]:
                        best = (key, f.id)
            if best is None:
                return None, None
            chains.append({"base": best[1], "visits": [(c.id, best[1])]})
            ton[best[1]] = ton.get(best[1], 0) + c.demand
    return chains, ton


def solve(inst, seed=1, iterations=60, q_fraction=0.15):
    rng = random.Random(seed)
    t0 = time.time()
    chains, ton = initial_solution(inst, rng)
    if chains is None:
        return {"status": "INFEASIBLE_CONSTRUCTION"}
    cur = [dict(base=c["base"], visits=list(c["visits"])) for c in chains]
    best = solution_cost(inst, cur)
    best_sol = [dict(base=c["base"], visits=list(c["visits"])) for c in cur]
    cur_cost = best
    weights = {(d, r): 1.0 for d in DESTROY_OPS for r in REPAIR_OPS}
    usage = {(d, r): 0 for d in DESTROY_OPS for r in REPAIR_OPS}
    temp = max(best * 0.04, 1e-9)

    n = len(inst.customers)
    q = min(60, max(8, int(n * q_fraction)))
    for it in range(iterations):
        pairs = [(d, r) for d in DESTROY_OPS for r in REPAIR_OPS]
        tot = sum(weights.values())
        pick = rng.random() * tot
        acc = 0
        for p in pairs:
            acc += weights[p]
            if acc >= pick:
                break
        d, r = p
        destroyed, victims = DESTROY[d](inst, cur, q, rng)
        ton2 = facility_tonnage(inst, destroyed)
        repaired = REPAIR[r](inst, destroyed, victims, ton2, rng)
        if repaired is None:
            weights[(d, r)] *= 0.5
            continue
        if not feasible(inst, repaired):
            # hard gate: never accept an infeasible incumbent; dump evidence
            # for post-mortem (root cause of the residual writer under audit)
            bad = [i for i, ch in enumerate(repaired)
                   if chain_times(inst, ch) is None]
            try:
                Path("/tmp/mt_alns_bad").mkdir(exist_ok=True)
                (Path("/tmp/mt_alns_bad") / f"{inst.name}_s{seed}_it{it}.json").write_text(
                    json.dumps({"op": [d, r], "bad_chains": bad,
                                "chains": [dict(c) for c in repaired]},
                               ensure_ascii=False))
            except Exception:
                pass
            weights[(d, r)] *= 0.5
            continue
        cost = solution_cost(inst, repaired)
        usage[(d, r)] += 1
        if cost < cur_cost - 1e-9 or rng.random() < math.exp((cur_cost - cost) / temp):
            cur, cur_cost = repaired, cost
            weights[(d, r)] *= 1.1 if cost < cur_cost else 1.0
            if cost < best - 1e-9:
                best = cost
                best_sol = [dict(base=c["base"], visits=list(c["visits"])) for c in repaired]
                weights[(d, r)] *= 1.2
        else:
            weights[(d, r)] *= 0.99
        temp *= 0.9995

    best_sol, best = polish(inst, best_sol)
    return {"status": "OK", "objective": round(best, 1),
            "vehicles": len(best_sol),
            "median_chain": sorted(len(c["visits"]) for c in best_sol)[len(best_sol) // 2],
            "max_chain": max(len(c["visits"]) for c in best_sol),
            "chains": best_sol,
            "sec": round(time.time() - t0, 1),
            "feasible": feasible(inst, best_sol)}


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    for nm in (sys.argv[1:] or ["NB-L2-n3-s1", "NB-L2-n5-s1", "SX-L3-n5-s1"]):
        inst = BenchmarkInstance.load(ROOT / "instances" / f"{nm}.json")
        r = solve(inst, seed=1, iterations=60)
        r["instance"] = nm
        print({k: r[k] for k in ("instance", "status", "objective", "vehicles",
                                 "median_chain", "max_chain", "sec", "feasible")},
              flush=True)
