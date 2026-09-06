# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`clif` is a research package for Clifford/geometric algebra, currently focused on fast
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

The package is organized as four cooperating layers (see `clif/__init__.py` docstring
for the full quick-start):

1. `clif.context` — global algebra state (dimension, signature, degeneracy, grade table).
   Configure the algebra with `context.Cl(p, q=0, r=0)` or `context.Layout([...])` before
   constructing any multivectors.
2. `clif.sign_table` — builds/caches the multiplication sign table and generates a
   Numba-jitted fast multiplier for dimensions < 8. Supports the full algebra family
   Cl(p, q, r): `neg_mask` (basis vectors squaring to −1) is corrected into the key arrays
   at setup time; `deg_mask` (null basis vectors) is enforced as a zero-contribution check
   in the inner product loop. All three context entry points (`Cl`, `Layout`, `Initialize`)
   propagate these masks automatically.
3. `clif.multivector` — the `Accum` class: a multivector as a NumPy array of real
   coefficients indexed by ordinal (binary representation names the basis blade, e.g.
   `5 = 0b101 = e1^e3`). Note: this ordinal ordering differs from the graded lexicographic
   ordering used by the public `clifford` package on PyPI — that's a deliberate choice here,
   not a bug.
4. `clif.util` — construction/printing/grade-projection/involution helpers and
   dimension-specific inverses.

`clif.inverse` holds the inverse algorithms: `fls.py` (Faddeev–LeVerrier–Souriau based,
with even/odd-graded variants), `sparse_fls.py`, `newton.py` (Newton–Schulz refinement), and
`euclidean.py` (Euclidean Cl(d, 0), dimensions 6–13).

## Repo hygiene — research scratch vs. real package code

`clif/inverse/` currently mixes the real, exported algorithms above with leftover
per-dimension experiment scripts and notebooks (e.g. `.ipynb` files, `fls_eigen.py`).
**The plan is to move old material to `scratch/inverse/`** (already gitignored), keeping
only the 6–13D Euclidean odd/even optimized inverses once reviewed — this hasn't happened
yet, so don't assume the current file list is final. When adding new exploratory
per-dimension code, prefer dropping it in `scratch/` directly rather than `clif/inverse/`.

The normal workflow is notebook-first: new inverse algorithms are prototyped in Jupyter, then
the working version is ported into the `clif/` package proper. Don't promote notebook
exploration into package code unless asked — flag what looks portable instead.

## Conventions

- Commit messages use conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`).
- Docs build with Sphinx: `cd docs && make html`, output at `docs/_build/html/index.html`.
