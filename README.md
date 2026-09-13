# CDW-LIB Minimal Reproducibility Package

This package accompanies the manuscript *From Regulatory Data to Routing Benchmarks*.
It contains only the cleaned input tables, benchmark instances, and core solver/generator code.
It does not contain the manuscript, figures, figure-generation scripts, or private credentials.

## Contents
- `data/`: seven cleaned tables. Identifiers are pseudonymized and personal/contact fields are removed.
- `instances/`: 137 benchmark instances and the instance manifest.
- `src/`: latent-world generator, ALNS, learning-augmented ALNS, exact MIP, and common data structures.

## Reproduction
Python 3.10+ is required. `numpy`, `pandas`, and `docplex` are required; CPLEX 22.1 is required for the exact MIP.

```bash
python3 src/cdwlib_gen.py
```
The generator creates the benchmark instances deterministically from the cleaned tables.

## Licensing
Code: MIT. Derived instances and cleaned tables: CC BY 4.0.
