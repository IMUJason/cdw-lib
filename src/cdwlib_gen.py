#!/usr/bin/env python3
"""CDW-LIB 生成器 v2：潜在世界（latent world）设计。【C1/M3 修复，取代 v1（已归档）】

设计原则：
  1. 潜在世界（流 A，seed 决定）与成熟度无关：真实工地（坐标+日需求+区县）、
     真实设施（不可变实体 ID `{ARC}-F{k}`）、车次拆分（≤10t）、服务时长、
     时间窗中心、车队额定载重（ratedload，语义见 02_数据语义核实备忘）。
  2. 成熟度 = 观测函数（流 B 仅影响观测抖动）：
       L3 完整观测 / L2 去时间窗 / L1 坐标→区县质心+车队聚合
     单因子分解族：TW / SP / FL（NB 潜在世界）。
  3. 孪生不变量（--validate 校验）：设施逐字段相同、车次需求/服务/数量相同。
  4. restricted 全天且 penalty=1 → 行驶时间出发时刻无关（与 MIP 时间语义一致）。

校准：D1 日需求=总量/工期（737 池，中位 155t）；D2 容量归一化 1.3×需求（潜在内）；
D3 载重=宁波渣土车 ratedload（中位 16.1t）；D4 L3 窗中心=绍兴运营记录小时剖面±150min
（proxy 假设）；D5 退化=观测函数；D6 车次=ceil(日需求/10)。
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
GPS_RAW = (Path(__file__).resolve().parent.parent.parent / "D1_数据采集" / "raw" /
           "shaoxing" / "205427_gps_snapshot")
OUT = Path(__file__).resolve().parent.parent / "instances"

DAY_START, DAY_END = 9 * 60, 21 * 60 + 30
AVG_SPEED = 30.0
PRESSURE = 1.3
TRIP_T = 10.0


def hour_profile():
    """Hour-of-day histogram driving L3 time-window centers.

    Reads the frozen histogram shipped in data/hour_profile.json, so the
    family regenerates from this package alone. If the raw GPS snapshots are
    present (they are large and are not distributed here), they are used
    instead to rebuild the histogram from source.
    """
    raw = sorted(GPS_RAW.glob("part*.csv"))
    if raw:
        hours = Counter()
        for p in raw:
            df = pd.read_csv(p, usecols=["gxsj"])
            for h in pd.to_datetime(df["gxsj"], errors="coerce").dt.hour.dropna():
                hours[int(h)] += 1
        if hours:
            return sorted(hours.items())
    frozen = FROZEN / "hour_profile.json"
    with open(frozen) as fh:
        return [tuple(x) for x in json.load(fh)["hour_counts"]]


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
        rng = random.Random(seed)          # 流 A：潜在世界唯一随机源
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

        # 设施：一次采样，不可变 ID；容量归一化到 需求×PRESSURE（吨）
        # DQ/WZ 无真实设施池 → 潜在世界内合成（围绕质心，lognormal 容量，流 A）
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

        # 车次：同址拆分（需求 t、服务、窗中心）
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


def observe(world, name, arc, maturity, prof):
    rng = obs_rng(name)                    # 流 B：仅观测抖动
    show_tw = maturity == "L3"
    real_coord = maturity in ("L3", "L2", "TW", "FL")
    vehicle_fleet = maturity in ("L3", "L2", "TW", "SP")

    customers = []
    for i, (si, per, service, twc) in enumerate(world.trips):
        s = world.sites[si]
        if real_coord:
            x, y = s[0], s[1]
        else:
            cx, cy = world.district_centroid[s[3]]
            x, y = cx + rng.uniform(-0.01, 0.01), cy + rng.uniform(-0.01, 0.01)
        if show_tw:
            e, l = max(DAY_START, twc - 150), min(DAY_END, twc + 150)
            if l - e < 120:
                e, l = DAY_START, DAY_END
        else:
            e, l = DAY_START, DAY_END
        customers.append({"id": i, "name": f"{name}-c{i}", "x": round(x, 6),
                          "y": round(y, 6), "demand": round(per, 3),
                          "earliest": int(e), "latest": int(l),
                          "service_time": service, "hotspot": "", "hotspot_weight": 1.0,
                          "source_project": f"{arc}-latent"})

    veh_cap = round(world.payload_t if vehicle_fleet else world.mean_payload_t * 1.25, 2)
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
    fails = checked = 0
    pairs = [(("SX", "L3"), "L2"), (("SX", "L3"), "L1"), (("SX", "L2"), "L1"),
             (("NB", "L2"), "L1"), (("NB", "L2"), "TW"), (("NB", "L2"), "SP"),
             (("NB", "L2"), "FL")]
    for (arc, base), d in pairs:
        for size in SIZE_SEEDS:
            for s in range(1, SIZE_SEEDS[size] + 1):
                bp = OUT / f"{arc}-{base}-n{size}-s{s}.json"
                dp = (OUT / f"{arc}-{d}-n{size}-s{s}.json" if d in ("L1", "L2", "L3")
                      else OUT / f"NBd-{d}-n{size}-s{s}.json")
                if not (bp.exists() and dp.exists()):
                    continue
                b, dd = json.loads(bp.read_text()), json.loads(dp.read_text())
                checked += 1
                ok = (b["facilities"] == dd["facilities"]
                      and [c["demand"] for c in b["customers"]] ==
                          [c["demand"] for c in dd["customers"]]
                      and [c["service_time"] for c in b["customers"]] ==
                          [c["service_time"] for c in dd["customers"]]
                      and len(b["customers"]) == len(dd["customers"]))
                if not ok:
                    fails += 1
                    print(f"[FAIL] {bp.name} vs {dp.name}")
    print(f"配对不变量: {checked} 对, {fails} 失败 {'✅' if fails == 0 else '❌'}")
    return fails == 0


def main():
    prof = build_pools()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()

    worlds = {}

    def world_of(arc, n, s):
        key = (arc, n, s)
        if key not in worlds:
            worlds[key] = LatentWorld(arc, n, M_FAC[n], 1000 * n + s, prof)
        return worlds[key]

    rows = []

    def emit(name, w, arc, mat):
        inst = observe(w, name, arc, mat, prof)
        (OUT / f"{name}.json").write_text(json.dumps(inst, ensure_ascii=False), encoding="utf-8")
        rows.append({"name": name, "archetype": arc, "maturity": mat,
                     "n": len(w.sites), "m": len(w.facilities), "n_trips": len(w.trips)})

    for arc, mat in CORE:
        for size, ns in SIZE_SEEDS.items():
            for s in range(1, ns + 1):
                emit(f"{arc}-{mat}-n{size}-s{s}", world_of(arc, size, s), arc, mat)
    for factor in ["TW", "SP", "FL"]:
        for size in [5, 10]:
            for s in range(1, 4):
                emit(f"NBd-{factor}-n{size}-s{s}", world_of("NB", size, s), "NB", factor)

    pd.DataFrame(rows).to_csv(OUT / "library_manifest.csv", index=False)
    # 容量健全性门（防单位 bug 复发）
    bad = 0
    for r in rows:
        d = json.loads((OUT / f"{r['name']}.json").read_text())
        tot_d = sum(c["demand"] for c in d["customers"])
        tot_c = sum(f["capacity"] for f in d["facilities"])
        if abs(tot_c - PRESSURE * tot_d) > 0.02 * tot_d or min(f["capacity"] for f in d["facilities"]) <= 0:
            bad += 1
            print(f"[CAP-FAIL] {r['name']}: cap {tot_c:.1f} vs demand×{PRESSURE} {PRESSURE*tot_d:.1f}")
    print(f"容量门: {bad} 失败 {'✅' if bad == 0 else '❌'}")
    print(f"v2.2 生成完成: {len(rows)} 实例（核心 {sum(1 for r in rows if r['maturity'] in ('L1','L2','L3'))} + 分解 18）")
    validate()


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
    else:
        main()
