#!/usr/bin/env python3
"""Reproduce the aggregate-payload (Q-prime) sweep of Section 7.

Regenerates the instance family at a given aggregate-payload factor in an
isolated workdir (the released family in ./instances is never touched),
then runs the paired value-of-information evaluation on it.

Usage:
  python3 src/qfactor_sweep.py 1.10            # full 80-pair grid
  python3 src/qfactor_sweep.py 1.40 NB L2 FL 5 1   # single pair
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        raise SystemExit(2)
    factor, extra = argv[0], argv[1:]
    py = sys.executable
    with tempfile.TemporaryDirectory(prefix=f"cdwlib-qf{factor}-") as td:
        wd = Path(td)
        shutil.copytree(ROOT / "src", wd / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data", wd / "data")
        (wd / "instances").mkdir()
        subprocess.run([py, "-I", str(wd / "src" / "cdwlib_gen.py"),
                        "--qfactor", factor], check=True, cwd=wd)
        cmd = [py, "-I", str(wd / "src" / "mt_vss.py"),
               f"--qfactor={factor}"] + (extra if extra else ["--all"])
        subprocess.run(cmd, check=True, cwd=wd)
        src = wd / "results" / "vss_mt_results.jsonl"
        if src.exists() and not extra:   # archive only full --all runs
            out = ROOT / "certified" / f"vss_mt_qf{factor.replace('.', '')}.jsonl"
            out.parent.mkdir(exist_ok=True)
            shutil.copy2(src, out)
            print(f"sweep records copied to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
