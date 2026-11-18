#!/usr/bin/env python3
"""Full-family multi-trip ALNS runner (131 instances).

- iteration schedule by size class: n3/n5=200, n10=150, n20=60, n30=30
- seeds 1..3 per instance
- 4 parallel workers; results appended row-by-row to
  results/mt_alns_results.csv (resume: existing (instance, seed) pairs
  are skipped)

Usage:
  python3 run_mt_alns.py [--workers=4] [--seeds=3] [--only PREFIX ...]
"""
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = ROOT / "results" / "mt_alns_results.csv"

FIELDS = ["instance", "seed", "iterations", "objective", "vehicles",
          "median_chain", "max_chain", "feasible", "elapsed_s", "ts"]

ITERS_BY_N = {3: 200, 5: 200, 10: 150, 20: 60, 30: 30}


def iter_count(name: str) -> int:
    for n in (30, 20, 10, 5, 3):
        if f"-n{n}-" in name:
            return ITERS_BY_N[n]
    return 60


def job(args):
    name, seed = args
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "mta", ROOT / "src" / "mt_alns.py")
    mta = importlib.util.module_from_spec(spec)
    mta.__name__ = "mta"
    spec.loader.exec_module(mta)
    from common import BenchmarkInstance
    inst = BenchmarkInstance.load(ROOT / "instances" / f"{name}.json")
    iters = iter_count(name)
    t0 = time.time()
    try:
        r = mta.solve(inst, seed=seed, iterations=iters)
    except Exception as exc:  # noqa: BLE001
        return {"instance": name, "seed": seed, "iterations": iters,
                "error": repr(exc), "ts": time.strftime("%F %T")}
    r.pop("chains", None)
    row = {"instance": name, "seed": seed, "iterations": iters,
           "objective": r.get("objective"), "vehicles": r.get("vehicles"),
           "median_chain": r.get("median_chain"),
           "max_chain": r.get("max_chain"),
           "feasible": r.get("feasible"),
           "elapsed_s": round(time.time() - t0, 1),
           "ts": time.strftime("%F %T")}
    return row


def main():
    argv = sys.argv[1:]
    workers, seeds = 4, 3
    only = []
    k = 0
    while k < len(argv):
        a = argv[k]
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
        elif a.startswith("--seeds="):
            seeds = int(a.split("=")[1])
        else:
            only.append(a)
        k += 1

    manifest = ROOT / "instances" / "library_manifest.csv"
    import pandas as pd
    names = pd.read_csv(manifest)["name"].tolist()
    if only:
        names = [n for n in names if any(n.startswith(p) for p in only)]

    done = set()
    if OUT.exists():
        with OUT.open() as fh:
            for rec in csv.DictReader(fh):
                done.add((rec["instance"], rec["seed"]))
    jobs = [(n, s) for n in names for s in range(1, seeds + 1)
            if (n, str(s)) not in done and (n, s) not in done]
    print(f"{len(names)} instances, {len(jobs)} runs to do "
          f"({len(done)} already done), workers={workers}", flush=True)
    if not jobs:
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    fh = OUT.open("a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        w.writeheader()
    t0 = time.time()
    with Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(job, jobs), 1):
            w.writerow(row)
            fh.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {row['instance']} s{row['seed']} "
                      f"obj={row.get('objective')} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
    fh.close()
    print(f"DONE {len(jobs)} runs in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
