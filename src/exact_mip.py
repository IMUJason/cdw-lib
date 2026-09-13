#!/usr/bin/env python3
"""CDW-LIB 精确求解 MIP v2。【C2 修复】

v1 缺陷（审稿 C2）：只有全车辆聚合的入/出度，允许"车 A 进入客户、车 B 离开"。
v2 修复：
  1. 逐客户逐车辆流守恒：in_cv(i,v) = out_cv(i,v) = visit(i,v)
     （xs 是入弧；xc 计入/出；xe 是出弧——方向语义修正）
  2. 路线提取器：从弧变量值重建每车路线（xs 起点 → xc 链 → xe 终点）
  3. 独立验证器：对提取路线逐项检查容量/时间窗/时长/设施容量并重算目标，
     与求解器目标比对（不一致即报错）——最优性声明以验证器为准
  4. 实例已设 restricted 全天 + penalty=1 → 行驶时间与出发时刻无关，
     弧时间参数用 departure=0 与启发式一致
时间链：MTZ 式 t[j] >= t[i] + service_i + travel(i,j)（xc 弧），
       起点 t[c] >= day_start + travel(f,c)（xs 弧），
       终点时长约束 t[c] + service_c + travel(c,f) <= day_start + max_route（xe 弧）。
"""
import math
import time
from collections import defaultdict
from pathlib import Path

from docplex.mp.model import Model

from common import BenchmarkInstance


def build_arc_sets(inst):
    start, end, cust = set(), set(), set()
    for f in inst.facilities:
        for c in inst.customers:
            start.add((f.id, c.id))
            end.add((c.id, f.id))
    for c1 in inst.customers:
        for c2 in inst.customers:
            if c1.id != c2.id:
                cust.add((c1.id, c2.id))
    return start, end, cust


def vehicle_limit(inst):
    return len(inst.customers)  # 宽松上限：每客户一车


def solve(inst: BenchmarkInstance, time_limit: int):
    start_arcs, end_arcs, cust_arcs = build_arc_sets(inst)
    vehicles = list(range(vehicle_limit(inst)))

    model = Model(name=f"exact_{inst.name}")
    model.parameters.timelimit = time_limit
    model.parameters.threads = 4

    visit = {(c.id, v): model.binary_var() for c in inst.customers for v in vehicles}
    xs = {(f, c, v): model.binary_var() for (f, c) in start_arcs for v in vehicles}
    xc = {(i, j, v): model.binary_var() for (i, j) in cust_arcs for v in vehicles}
    xe = {(c, f, v): model.binary_var() for (c, f) in end_arcs for v in vehicles}
    used = {v: model.binary_var() for v in vehicles}
    opened = {f.id: model.binary_var() for f in inst.facilities}
    t = {(c.id, v): model.continuous_var(lb=0) for c in inst.customers for v in vehicles}
    load = {(f.id, v): model.continuous_var(lb=0) for f in inst.facilities for v in vehicles}
    BIG_M = inst.day_end + 2000
    total_demand = sum(c.demand for c in inst.customers)

    # ---- 方向语义：xs 入弧，xc 双向，xe 出弧（逐 (客户,车辆) 分组）----
    in_cv, out_cv = defaultdict(list), defaultdict(list)
    for (f, c, v), var in xs.items():
        in_cv[(c, v)].append(var)
    for (i, j, v), var in xc.items():
        out_cv[(i, v)].append(var)
        in_cv[(j, v)].append(var)
    for (c, f, v), var in xe.items():
        out_cv[(c, v)].append(var)

    # ---- C2 核心：逐客户逐车辆流守恒 ----
    for c in inst.customers:
        model.add_constraint(model.sum(visit[(c.id, v)] for v in vehicles) == 1)
        for v in vehicles:
            model.add_constraint(model.sum(in_cv[(c.id, v)]) == visit[(c.id, v)],
                                 ctname=f"inflow_{c.id}_{v}")
            model.add_constraint(model.sum(out_cv[(c.id, v)]) == visit[(c.id, v)],
                                 ctname=f"outflow_{c.id}_{v}")

    for v in vehicles:
        model.add_constraint(
            model.sum(c.demand * visit[(c.id, v)] for c in inst.customers)
            <= inst.vehicle_capacity * used[v])
        model.add_constraint(
            model.sum(xs[(f, c, v)] for (f, c) in start_arcs) == used[v])
        for f in inst.facilities:
            model.add_constraint(
                model.sum(xs[(f.id, c.id, v)] for c in inst.customers)
                == model.sum(xe[(c.id, f.id, v)] for c in inst.customers))
            ss = model.sum(xs[(f.id, c.id, v)] for c in inst.customers)
            sv = model.sum(c.demand * visit[(c.id, v)] for c in inst.customers)
            model.add_constraint(load[(f.id, v)] <= sv)
            model.add_constraint(load[(f.id, v)] >= sv - total_demand * (1 - ss))
            model.add_constraint(load[(f.id, v)] <= total_demand * ss)
    for f in inst.facilities:
        model.add_constraint(model.sum(load[(f.id, v)] for v in vehicles)
                             <= f.capacity * opened[f.id])
    for v in range(len(vehicles) - 1):
        model.add_constraint(used[v] >= used[v + 1])

    for c in inst.customers:
        for v in vehicles:
            model.add_constraint(t[(c.id, v)] >= c.earliest * visit[(c.id, v)])
            model.add_constraint(t[(c.id, v)] <= c.latest + BIG_M * (1 - visit[(c.id, v)]))
    for (f, c, v), var in xs.items():
        tt = inst.travel_time("f", f, "c", c, 0)
        model.add_constraint(t[(c, v)] >= inst.day_start + tt - BIG_M * (1 - var))
    for (i, j, v), var in xc.items():
        model.add_constraint(
            t[(j, v)] >= t[(i, v)] + inst.customer_map[i].service_time
            + inst.travel_time("c", i, "c", j, 0) - BIG_M * (1 - var))
    for (c, f, v), var in xe.items():
        model.add_constraint(
            t[(c, v)] + inst.customer_map[c].service_time
            + inst.travel_time("c", c, "f", f, 0)
            <= inst.day_start + inst.max_route_minutes + BIG_M * (1 - var))

    transport = model.sum(
        inst.distance("f", f, "c", c) * inst.transport_cost_per_km * var
        for (f, c, v), var in xs.items())
    transport += model.sum(
        inst.distance("c", i, "c", j) * inst.transport_cost_per_km * var
        for (i, j, v), var in xc.items())
    transport += model.sum(
        inst.distance("c", c, "f", f) * inst.transport_cost_per_km * var
        for (c, f, v), var in xe.items())
    model.minimize(transport
                   + model.sum(inst.vehicle_fixed_cost * used[v] for v in vehicles)
                   + model.sum(f.fixed_cost * opened[f.id] for f in inst.facilities))

    t0 = time.time()
    sol = model.solve(log_output=False)
    elapsed = time.time() - t0

    res = {"solver": "MIP", "elapsed": elapsed, "objective": None, "gap": None,
           "status": "NO_SOLUTION", "validated": False, "violations": None}
    try:
        res["status"] = str(model.solve_details.status) or "UNKNOWN"
    except Exception:
        pass
    if sol:
        res["objective"] = sol.get_objective_value()
        try:
            res["gap"] = model.solve_details.mip_relative_gap
        except Exception:
            pass
        routes = extract_routes(sol, xs, xc, xe, vehicles, inst)
        check = validate_routes(inst, routes)
        res["violations"] = check["violations"]
        res["validated"] = check["ok"] and abs(
            check["objective"] - sol.get_objective_value()) < 1e-4 * max(1, abs(sol.get_objective_value()))
        res["routes"] = routes
        res["route_count"] = len(routes)
        res["recomputed_objective"] = check["objective"]
    return res


def extract_routes(sol, xs, xc, xe, vehicles, inst):
    """从弧变量重建每车路线：xs → xc 链 → xe。"""
    routes = []
    for v in vehicles:
        # 起点
        start_c = None
        for (f, c, vv), var in xs.items():
            if vv == v and round(sol.get_value(var)) == 1:
                start_c = (f, c)
                break
        if start_c is None:
            continue
        f0, c0 = start_c
        seq = [c0]
        cur = c0
        end_f = None
        for _ in range(len(inst.customers) + 1):
            nxt = None
            for (i, j, vv), var in xc.items():
                if vv == v and i == cur and round(sol.get_value(var)) == 1:
                    nxt = j
                    break
            if nxt is None:
                for (c, f, vv), var in xe.items():
                    if vv == v and c == cur and round(sol.get_value(var)) == 1:
                        end_f = f
                    if end_f is not None:
                        break
                break
            seq.append(nxt)
            cur = nxt
        if end_f is not None and f0 == end_f:
            routes.append({"facility_id": f0, "customers": seq})
    return routes


def validate_routes(inst, routes):
    """独立验证器：逐项检查并重算目标（不信任求解器内部）。"""
    viol = {"vehicle_overload_t": 0.0, "facility_overload_t": 0.0,
            "lateness_min": 0.0, "overtime_min": 0.0, "missing": 0, "duplicate": 0}
    fac_load = defaultdict(float)
    cost = 0.0
    seen = set()
    for r in routes:
        f = inst.facility_map[r["facility_id"]]
        seq = r["customers"]
        load = sum(inst.customer_map[c].demand for c in seq)
        viol["vehicle_overload_t"] += max(load - inst.vehicle_capacity, 0)
        fac_load[f.id] += load
        cur_t = float(inst.day_start)
        dist = 0.0
        prev = ("f", f.id)
        for cid in seq:
            if cid in seen:
                viol["duplicate"] += 1
            seen.add(cid)
            c = inst.customer_map[cid]
            tt = inst.travel_time(prev[0], prev[1], "c", cid, 0)
            dist += inst.distance(prev[0], prev[1], "c", cid)
            cur_t += tt
            cur_t = max(cur_t, float(c.earliest))
            viol["lateness_min"] += max(cur_t - c.latest, 0)
            cur_t += c.service_time
            prev = ("c", cid)
        dist += inst.distance(prev[0], prev[1], "f", f.id)
        cur_t += inst.travel_time(prev[0], prev[1], "f", f.id, 0)
        viol["overtime_min"] += max(cur_t - inst.day_start - inst.max_route_minutes, 0)
        cost += dist * inst.transport_cost_per_km + inst.vehicle_fixed_cost
    for fid, ld in fac_load.items():
        viol["facility_overload_t"] += max(ld - inst.facility_map[fid].capacity, 0)
        cost += inst.facility_map[fid].fixed_cost
    viol["missing"] = len(inst.customers) - len(seen)
    ok = all(v == 0 for v in viol.values())
    return {"ok": ok, "violations": viol, "objective": cost}


def brute_force(inst: BenchmarkInstance):
    """微型实例（≤6 车次）真值穷举：集合划分 × 块内全序 × 设施分配 × 容量检查。"""
    from itertools import permutations, product
    cs = sorted(c.id for c in inst.customers)
    assert len(cs) <= 6
    fac_ids = [f.id for f in inst.facilities]

    def route_cost(f, seq):
        load = sum(inst.customer_map[c].demand for c in seq)
        if load > inst.vehicle_capacity + 1e-9:
            return None
        cur_t = float(inst.day_start)
        dist = 0.0
        prev = ("f", f)
        for cid in seq:
            c = inst.customer_map[cid]
            cur_t += inst.travel_time(prev[0], prev[1], "c", cid, 0)
            dist += inst.distance(prev[0], prev[1], "c", cid)
            cur_t = max(cur_t, float(c.earliest))
            if cur_t > c.latest + 1e-6:
                return None
            cur_t += c.service_time
            prev = ("c", cid)
        dist += inst.distance(prev[0], prev[1], "f", f)
        cur_t += inst.travel_time(prev[0], prev[1], "f", f, 0)
        if cur_t - inst.day_start > inst.max_route_minutes + 1e-6:
            return None
        return dist * inst.transport_cost_per_km + inst.vehicle_fixed_cost

    def set_partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for sub in set_partitions(rest):
            for i in range(len(sub)):
                yield sub[:i] + [[first] + sub[i]] + sub[i+1:]
            yield sub + [[first]]

    best = None
    for blocks in set_partitions(cs):
        for orders in product(*[list(permutations(b)) for b in blocks]):
            for facs in product(fac_ids, repeat=len(blocks)):
                total = 0.0
                fload = defaultdict(float)
                ok = True
                for seq, f in zip(orders, facs):
                    rc = route_cost(f, list(seq))
                    if rc is None:
                        ok = False
                        break
                    total += rc
                    fload[f] += sum(inst.customer_map[c].demand for c in seq)
                if not ok:
                    continue
                if any(fload[f] > inst.facility_map[f].capacity + 1e-6 for f in fload):
                    continue
                total += sum(inst.facility_map[f].fixed_cost for f in set(facs))
                if best is None or total < best:
                    best = total
    return best


def _compositions(n, lo, hi):
    """长度 1..n 的有序正整数组合（切段方案）。"""
    def rec(rem, acc):
        if rem == 0:
            yield list(acc)
            return
        for k in range(1, rem + 1):
            acc.append(k)
            yield from rec(rem - k, acc)
            acc.pop()
    yield from rec(n, [])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    inst = BenchmarkInstance.load(Path(sys.argv[1]))
    res = solve(inst, time_limit=int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    print({k: v for k, v in res.items() if k != "routes"})
