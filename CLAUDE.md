# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`clifford` is a research package for Clifford/geometric algebra, currently focused on fast
multivector inversion. The author is publishing a paper on a 6D inverse and is generalizing
that result to a dimension-independent inverse algorithm for an upcoming conference talk.
This is an early-stage research repo (no CI, no test suite yet) — treat correctness claims
about new inverse algorithms as provisional until verified numerically (e.g. `A * A.inverse()`
reducing to a clean scalar).

## Install / environment

- Python ≥3.10 (dev env uses conda env `clifford-dev`, Python 3.12).
- `pip install -e .` from the repo root. Optional `dev` extra adds `jupyter`/`ipykernel`.
- Runtime deps: `numpy>=1.24`, `numba>=0.58`.
- No linter/formatter is configured yet.

## Package architecture

The package is organized as four cooperating layers (see `clifford/__init__.py` docstring
for the full quick-start):

1. `clifford.context` — global algebra state (dimension, signature, degeneracy, grade table).
   Configure the algebra with `context.Cl(p, q=0, r=0)` or `context.Layout([...])` before
   constructing any multivectors.
2. `clifford.sign_table` — builds/caches the multiplication sign table and generates a
   Numba-jitted fast multiplier for dimensions < 8. Supports the full algebra family
   Cl(p, q, r): `neg_mask` (basis vectors squaring to −1) is corrected into the key arrays
   at setup time; `deg_mask` (null basis vectors) is enforced as a zero-contribution check
   in the inner product loop. All three context entry points (`Cl`, `Layout`, `Initialize`)
   propagate these masks automatically. `sign_table_old.py` is a superseded version kept
   for reference — don't extend it.
3. `clifford.multivector` — the `Accum` class: a multivector as a NumPy array of real
   coefficients indexed by ordinal (binary representation names the basis blade, e.g.
   `5 = 0b101 = e1^e3`). Note: this ordinal ordering differs from the graded lexicographic
   ordering used by the public `clifford` package on PyPI — that's a deliberate choice here,
   not a bug.
4. `clifford.util` — construction/printing/grade-projection/involution helpers and
   dimension-specific inverses.

`clifford.inverse` holds the inverse algorithms: `fls.py` (Faddeev–LeVerrier–Souriau based,
with even/odd-graded variants), `sparse_fls.py`, `newton.py` (Newton–Schulz refinement), and
`euclidean.py` (Euclidean Cl(d, 0), dimensions 6–13).

## Repo hygiene — research scratch vs. real package code

`clifford/inverse/` currently mixes the real, exported algorithms above with leftover
per-dimension experiment scripts and notebooks (e.g. `I8_v2_8D_inverse.py`, `ga4d_inverse.py`,
`self_reverse_6d/7d/8d.py`, `i7_reference.py`, `sri_table_gen.py`, the `.ipynb` files,
`FLS-SRI-to-11D.md`). **The plan is to move this old material to `scratch/inverse/`** (already
gitignored), keeping only the 6–13D Euclidean odd/even optimized inverses once reviewed — this
hasn't happened yet, so don't assume the current file list is final. When adding new exploratory
per-dimension code, prefer dropping it in `scratch/` directly rather than `clifford/inverse/`.

The normal workflow is notebook-first: new inverse algorithms are prototyped in Jupyter, then
the working version is ported into the `clifford/` package proper. Don't promote notebook
exploration into package code unless asked — flag what looks portable instead.

## Conventions

- Commit messages use conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`).
- Docs build with Sphinx: `cd docs && make html`, output at `docs/_build/html/index.html`.
