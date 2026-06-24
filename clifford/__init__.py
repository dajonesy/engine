# file: clifford/__init__.py
"""
clifford — a Python package for Clifford algebra.

This package implements Clifford algebra (geometric algebra) with a focus on
explicit ordinal indexing of multivector components, efficient sign-table-based
multiplication, and support for arbitrary metric signature and degeneracy.

Architecture
------------
The package is organized as four cooperating layers:

1. **context** — global algebra state (dimension, signature, degeneracy, grade table).
   Call :func:`context.Initialize`, :func:`context.Layout`, or :func:`context.Cl`
   to configure the algebra before constructing any multivectors.

2. **sign_table** — builds and caches the multiplication sign table for the current
   algebra, and generates a Numba-jitted fast multiplier for dimensions < 8.

3. **multivector** — the :class:`multivector.Accum` class, which implements a
   multivector as a NumPy array of real coefficients indexed by their ordinal
   (a bitmask whose set bits name the basis vectors in the blade).

4. **util** — convenience functions for constructing, printing, and operating on
   multivectors (grade projection, involutions, random multivectors, inverses
   for specific dimensions).

Inverse algorithms live in the :mod:`clifford.inverse` sub-package: :func:`clifford.inverse.fls_inverse`
(Faddeev–LeVerrier–Souriau based, with :func:`even_inverse`/:func:`odd_inverse` variants and
sparse counterparts), :func:`clifford.inverse.euclidean_inverse`, and
:func:`clifford.inverse.newton_schulz` (iterative refinement).

Ordinal indexing convention
---------------------------
An element of the algebra is identified by a non-negative integer whose binary
representation names the basis vectors it contains.  In a 3-dimensional algebra::

    0  = 0b000  scalar          (grade 0)
    1  = 0b001  e1              (grade 1)
    2  = 0b010  e2              (grade 1)
    3  = 0b011  e1^e2           (grade 2)
    4  = 0b100  e3              (grade 1)
    5  = 0b101  e1^e3           (grade 2)
    6  = 0b110  e2^e3           (grade 2)
    7  = 0b111  e1^e2^e3  = I   (grade 3, pseudoscalar)

This ordering differs from the graded lexicographic ordering used in the public
``clifford`` package and is a deliberate design choice of this implementation.

Quick start
-----------
::

    from clifford.context import Cl, Layout
    from clifford.multivector import Accum

    Cl(1, 1, 1)        # algebra with 1 positive, 1 negative, 1 null basis vector
    A = Accum()
    A.random()
    print(A)

    Cl(1, 3)                  # spacetime algebra  Cl(1,3)
    Layout([1, -1, -1, -1])   # equivalent, explicit

Authors
-------
D. A. Jones  (original implementation)
"""

# ---- nice for the developer ---- (maybe)
from clifford import context
from clifford import sign_table
from clifford import multivector
from clifford import util

# ---- all a user should need ----
from clifford.context import Cl,Grade
from clifford.multivector import Accum

__all__ = [
    "context",
    "sign_table",
    "multivector",
    "util",
    "Cl",
    "Grade",
    "Accum"
]