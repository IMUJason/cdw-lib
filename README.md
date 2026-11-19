# CDW-LIB Minimal Reproducibility Package

This package accompanies the manuscript *From Regulatory Data to Routing Benchmarks:
An Information-Stratified Instance Library for Construction and Demolition Waste
Transportation*. It contains the cleaned input tables, the benchmark instance
family, and the generator and solver stack. It does not contain the manuscript,
figures, figure-generation scripts, or private credentials.

## Contents
- `data/`: seven cleaned tables (pseudonymized, personal/contact fields removed),
  the frozen hour-of-day histogram that drives time-window calibration, and the
  frozen tonnage distribution of the Shaoxing status records (the registry figure
  against which the aggregate-payload factor is anchored in the manuscript).
- `instances/`: 131 benchmark instances (119 core + 12 single-factor variants)
  and the instance manifest.
- `src/`: the latent-world generator (`cdwlib_gen.py`), the multi-trip ALNS
  (`mt_alns.py`), the chain-formulation exact model (`mt_prototype.py`), the
  time-expanded lower-bound relaxation (`mt_lb.py`), the certified-bounds
  battery (`mt_battery.py`), the value-of-information repair evaluation
  (`mt_vss.py`), and the family-wide ALNS runner (`run_mt_alns.py`).

## Reproduction
Python 3.10+ with `numpy` and `pandas`. `docplex` with a local CPLEX 22.1
engine is required for the exact components (`mt_prototype.py`, `mt_lb.py`,
`mt_battery.py`); the ALNS and the repair evaluation are standard-library only.

```bash
# Regenerate the instance family deterministically and re-verify the
# 80 entity-invariance pairs (prints: 80 pairs, 0 failures).
python3 src/cdwlib_gen.py

# Multi-trip ALNS on a micro instance.
python3 src/mt_alns.py SX-L3-n3-s1

# Certified interval [LB, UB] for a micro cell.
python3 src/mt_battery.py --ub=300 --lb=300 SX-L3-n3-s1

# Value-of-information evaluation with explicit repair execution.
python3 src/mt_vss.py NB L2 L1 5 1

# Full-family ALNS with resume (writes results/mt_alns_results.csv).
python3 src/run_mt_alns.py

# Regenerate the family at another aggregate-payload factor (isolated
# workdir; the released instances are never touched).
python3 src/cdwlib_gen.py --qfactor 1.10 --out /tmp/family_qf110

# Reproduce the Section-7 Q-prime sweep (full 80-pair grid per factor).
python3 src/qfactor_sweep.py 1.10

# Time-expanded lower bound at a finer slot width.
/opt/conda/envs/env310/bin/python -I src/mt_lb.py --delta=1 SX-L3-n3-s1
```

The `certified/` directory archives the certified results reported in the
manuscript: the n=3 and n=5 optimality-interval batteries and the
aggregate-payload sweep records at factors 1.10 and 1.40.

## Licensing
Code: MIT. Derived instances and cleaned tables: CC BY 4.0.
