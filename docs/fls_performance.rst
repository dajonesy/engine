FLS Inverse — Sparse Kernel Performance
========================================

The :mod:`clifford.inverse.sparse_fls` module implements the
Faddeev–LeVerrier–Souriau (FLS) inverse using compact pair-table Numba
kernels that operate entirely within the *active subspace* of the FLS
involution.

Algorithm overview
------------------

For a multivector ``A`` in Euclidean ``Cl(d, 0)``, the FLS inverse proceeds:

1. **Involution** — form ``B = invol(A)`` (elementwise sign flip, free).
2. **Enter** — compute ``C = A · B``, a full geometric product whose result
   lies in the active subspace (positive eigenspace of the involution).
3. **Polynomial loop** — iterate ``D ← D · C`` for ``N = 2^(d//2 − 1)``
   steps, with scalar (and pseudoscalar, in odd dimensions) tweaks at each
   step.  This is the expensive part; the sparse kernel runs it compactly.
4. **Exit** — form the adjugate coefficient array from ``D`` and the
   scalar denominator from one more compact product with ``C``.
5. **Reassemble** — ``A⁻¹ = B · (adjugate / det)``, a full geometric product.

Steps 2 and 5 are full-size products and are unavoidable.  Steps 3–4
run in the compact active subspace of dimension ``M ≈ 2^(d−1)``, using a
precomputed pair table.

Active subspace and pair table
-------------------------------

The active subspace is the positive eigenspace of the dimension-specific
FLS involution (reversion composed with a Thue–Morse block pattern).
For each non-scalar output blade in the active subspace, the pair table
records all pairs of active blades whose geometric product lands on that
output, together with the Euclidean product sign
``(−1)^popcount(row_key[a] ∧ b)``.

Because the FLS iterates commute with ``C`` (each is a scalar polynomial
of ``C``), the pair contributions are symmetric:
``sign(a, b) = sign(b, a)`` for all active pairs.  The compact multiply
therefore uses the symmetric combination
``S · (X[p₁] Y[p₂] + X[p₂] Y[p₁])`` with a single stored sign.

Tables are built once per dimension on first call and cached for the
session.  Table construction is ``O(M²)`` in Python; Numba JIT compilation
happens on the first product call.

Benchmark
---------

Random full multivectors, Euclidean ``Cl(d, 0)``, timed after JIT warm-up.
*Dense* is :func:`~clifford.inverse.fls.fls_inverse`;
*Sparse* is :func:`~clifford.inverse.sparse_fls.sparse_fls_inverse`.
Error is ``max |A · A⁻¹ − e₀|`` over all blades.

.. list-table::
   :header-rows: 1
   :widths: 5 8 8 12 12 10 12

   * - dim
     - blades
     - active M
     - max pairs P
     - dense µs
     - sparse µs
     - speedup
   * - 6
     - 64
     - 28
     - 6
     - 71.9
     - 33.5
     - 2.1×
   * - 7
     - 128
     - 56
     - 28
     - 203.9
     - 68.4
     - 3.0×
   * - 8
     - 256
     - 120
     - 28
     - 732.9
     - 230.5
     - 3.2×
   * - 9
     - 512
     - 240
     - 120
     - 3 107.8
     - 836.3
     - 3.7×
   * - 10
     - 1 024
     - 496
     - 120
     - 16 247.6
     - 3 990.6
     - 4.1×
   * - 11
     - 2 048
     - 992
     - 496
     - 72 390.9
     - 17 664.2
     - 4.1×
   * - 12
     - 4 096
     - 2 016
     - 496
     - 525 466.7
     - 102 467.1
     - 5.1×
   * - 13
     - 8 192
     - 4 032
     - 2 016
     - 2 259 116.1
     - 465 760.8
     - 4.9×

At dimensions 12–13 the polynomial accumulates enough floating-point
error to leave residuals of ``O(10⁻⁶)`` to ``O(10⁻³)``.  A two-pass
strategy using :func:`~clifford.inverse.newton.newton_schulz` recovers
full double precision:

.. list-table::
   :header-rows: 1
   :widths: 5 20 12

   * - dim
     - sparse FLS + Newton–Schulz µs
     - final error
   * - 12
     - 200 914.8
     - 9.0 × 10⁻¹⁶
   * - 13
     - 1 014 010.2
     - 1.8 × 10⁻¹⁵

The Newton–Schulz pass converges quadratically from the FLS initial
approximation; two or three iterations are typically sufficient to reach
machine precision.

Usage example
-------------

::

   import clifford.context as Clif
   from clifford.multivector import Accum
   from clifford.inverse.sparse_fls import sparse_fls_inverse
   from clifford.inverse.newton import newton_schulz
   import numpy as np

   Clif.Cl(13)                             # Euclidean Cl(13, 0)
   A = Accum()
   A.Reg = np.random.default_rng(0).standard_normal(1 << 13)

   X0 = sparse_fls_inverse(A)             # fast FLS approximation
   Ai = newton_schulz(A, X0)              # refine to machine precision

   mul = Clif._ActiveTable.fast_mul
   chk = mul(A.Reg, Ai.Reg)
   print(f"error: {max(abs(chk[0]-1), float(np.max(np.abs(chk[1:])))):.2e}")
   # → error: ~1.8e-15
