"""
file: self_reverse_6D.py
2026.02.08  DAJones — initial construction (with the "Self Reverse" notebook).
2026.05.10  Kernel and cache scaffolding completed to match jones.py.

Optimized 6D inverse for self-reverse multivectors
==================================================

Motivation
----------
The Jones inverse for a multivector ``A`` in ``Cl(p, q, r)`` with
``d = p + q + r ≤ 6`` proceeds via the self-reverse intermediate
``B = A · ~A``, computes ``B^{-1}`` by a short polynomial iteration,
and returns ``A^{-1} = ~A · B^{-1}``.  In 5D and 6D, the bulk of the
cost lives in the three multivector products that drive the iteration.

The kernel in :mod:`jones` performs each of those products as a generic
6D multivector multiplication — 64 input slots against 64 input slots,
with the full Cayley-table dispatch.  But the iterates ``B``, ``B² − …``,
``B³ − …`` are *all known a priori to be self-reverse*, so two-thirds
of those input slots and well over half of the output slots are dead
weight.  This module replaces the inner three products (and the final
scalar product that yields the determinant) with a compact form that
operates only on the 28 register slots a self-reverse multivector can
populate, dispatched through a precomputed pair table.

Two facts make this exact rather than approximate:

1. **The self-reverse blade indices in 6D form a 28-dimensional subspace.**
   The reverse acts on a grade-*k* blade by multiplication by
   ``(-1)^{k(k-1)/2}``.  In dimension 6 this evaluates to ``+1`` for
   grades 0, 1, 4, 5 and to ``-1`` for grades 2, 3, 6.  A multivector
   is therefore self-reverse iff its grade-{2,3,6} parts vanish, leaving
   ``1 + 6 + 15 + 6 = 28`` populated register slots out of 64.  The
   array ``I6`` enumerates the corresponding linear indices.

2. **The Jones iteration multiplies ``H`` only by scalar-shifted copies
   of itself.**  Scalars commute with every multivector, so the two
   factors at each step commute, and ``~(XY) = ~Y · ~X = Y · X = X · Y``;
   that is, the product is automatically self-reverse.  The compact
   multiply ``_mul_6`` therefore loses no information when it computes
   only the self-reverse projection of ``X · Y`` — the orthogonal
   complement is identically zero on this path.

The compact multiply itself exploits a separate symmetry: the self-
reverse projection of ``X · Y`` is symmetric under exchange of ``X``
and ``Y``, so for every contributing index pair ``(p1, p2)`` only the
combination ``X[p1]·Y[p2] + X[p2]·Y[p1]`` need be formed.  Each non-
scalar output slot in 6D receives contributions from exactly six such
pairs (see :data:`T6` below), with signs that depend on the algebra
signature.

Cost
----
Per compact multiply, the kernel performs:

  •  28 real muls + 27 adds for the scalar slot (the diagonal sum);
  •  27 non-scalar slots, each requiring 6 pairs × 2 real muls plus
     6 sign multiplications and a small accumulator —
     ≈ 12 real muls per slot, ≈ 324 real muls total;

so ~352 real muls per compact multiply, versus up to 64² = 4096
real-mul candidates in an unstructured 6D product.  The price is
indirect addressing through ``T6`` and ``S6`` — but these are int8
arrays totalling well under a kilobyte and live comfortably in L1.

The full inverse uses three compact multiplies plus one compact
scalar product, then unpacks the 28-slot result back into the 64-slot
output register.

Signature configurability
-------------------------
``S6`` and ``S6_0`` are initialised here for the Euclidean case
``Cl(6, 0, 0)``.  For any other signature ``Cl(p, q, r)`` with
``p + q + r = 6``, call :func:`reinit` with the full 64×64 sign
table for that signature; it rewrites the sign data in place and
invalidates the cached Numba kernel so the next call rebuilds under
the new signs.  Degenerate signatures (``r > 0``) work without change
— the only effect of a zero entry in the sign table is to silently
zero out the corresponding term in the compact multiply.

References
----------
Jones, D. A. — "Self Reverse" research notebook, 2026.
``jones.py`` — the enclosing inverse algorithm.
"""

import numpy as np
from numba import njit

#import clifford.context as Clif
from clifford.multivector import Accum

# ===============================================================
# Index and sign tables for the 6D self-reverse subspace
# ===============================================================

# I6 enumerates the 28 linear indices in the 64-slot register that
# correspond to grade-{0, 1, 4, 5} blades — the self-reverse blades
# in Cl(6).  Order matters: I6[0] = 0 is the scalar slot and is
# special-cased throughout (the "diagonal" scalar product handles it
# separately from the off-diagonal pair sums).
I6 = np.array([ 0, 1, 2, 4, 8, 15, 16, 23, 27, 29, 30, 31, 32, 39, 43, 45, 46,
                47, 51, 53, 54, 55, 57, 58, 59, 60, 61, 62 ], dtype=np.int8)

# S6_0 holds the "diagonal" signs — i.e. the values of ``e_α · e_α``
# for each blade ``e_α`` at index I6[i].  These signs feed the scalar
# part of any compact product: ``<X·Y>_0 = Σ_i S6_0[i] · X[i] · Y[i]``.
# In the Euclidean case every diagonal square is +1; the array is
# rewritten by reinit() for other signatures.
S6_0 = np.array([1] * len(I6), dtype=np.int8)

# T6 is the pair-contribution table for the non-scalar self-reverse
# output slots.  Row ``i`` (zero-indexed) corresponds to output slot
# ``I6[i+1]`` — the scalar slot is excluded, hence the index shift.
# Each row holds six pairs ``(p1, p2)``: the indices in the compact
# 28-slot form whose blade product e_{I6[p1]} · e_{I6[p2]} lands on
# the output blade, up to a sign carried by S6.  Six pairs per output
# is exact in 6D and reflects the multiplicity with which two self-
# reverse blades can combine to give a third self-reverse blade.
T6 = np.array([
     [ [0, 1], [10,11], [16,17], [20,21], [23,24], [25,26] ], #  1.  1  000001
     [ [0, 2], [ 9,11], [15,17], [19,21], [22,24], [25,27] ], #  2.  2  000010
     [ [0, 3], [ 8,11], [14,17], [18,21], [22,26], [23,27] ], #  3.  4  000100
     [ [0, 4], [ 7,11], [13,17], [18,24], [19,26], [20,27] ], #  4.  8  001000
     [ [0, 5], [ 6,11], [12,17], [18,25], [19,23], [20,22] ], #  5. 15  001111
     [ [0, 6], [ 5,11], [13,21], [14,24], [15,26], [16,27] ], #  6. 16  010000
     [ [0, 7], [ 4,11], [12,21], [14,25], [15,23], [16,22] ], #  7. 23  010111
     [ [0, 8], [ 3,11], [12,24], [13,25], [15,20], [16,19] ], #  8. 27  011011
     [ [0, 9], [ 2,11], [12,26], [13,23], [14,20], [16,18] ], #  9. 29  011101
     [ [0,10], [ 1,11], [12,27], [13,22], [14,19], [15,18] ], # 10. 30  011110
     [ [0,11], [ 1,10], [ 2, 9], [ 3, 8], [ 4, 7], [ 5, 6] ], # 11. 31  011111
     [ [0,12], [ 5,17], [ 7,21], [ 8,24], [ 9,26], [10,27] ], # 12. 32  100000
     [ [0,13], [ 4,17], [ 6,21], [ 8,25], [ 9,23], [10,22] ], # 13. 39  100111
     [ [0,14], [ 3,17], [ 6,24], [ 7,25], [ 9,20], [10,19] ], # 14. 43  101011
     [ [0,15], [ 2,17], [ 6,26], [ 7,23], [ 8,20], [10,18] ], # 15. 45  101101
     [ [0,16], [ 1,17], [ 6,27], [ 7,22], [ 8,19], [ 9,18] ], # 16. 46  101110
     [ [0,17], [ 1,16], [ 2,15], [ 3,14], [ 4,13], [ 5,12] ], # 17. 47  101111
     [ [0,18], [ 3,21], [ 4,24], [ 5,25], [ 9,16], [10,15] ], # 18. 51  110011
     [ [0,19], [ 2,21], [ 4,26], [ 5,23], [ 8,16], [10,14] ], # 19. 53  110101
     [ [0,20], [ 1,21], [ 4,27], [ 5,22], [ 8,15], [ 9,14] ], # 20. 54  110110
     [ [0,21], [ 1,20], [ 2,19], [ 3,18], [ 6,13], [ 7,12] ], # 21. 55  110111
     [ [0,22], [ 2,24], [ 3,26], [ 5,20], [ 7,16], [10,13] ], # 22. 57  111001
     [ [0,23], [ 1,24], [ 3,27], [ 5,19], [ 7,15], [ 9,13] ], # 23. 58  111010
     [ [0,24], [ 1,23], [ 2,22], [ 4,18], [ 6,14], [ 8,12] ], # 24. 59  111011
     [ [0,25], [ 1,26], [ 2,27], [ 5,18], [ 7,14], [ 8,13] ], # 25. 60  111100
     [ [0,26], [ 1,25], [ 3,22], [ 4,19], [ 6,15], [ 9,12] ], # 26. 61  111101
     [ [0,27], [ 2,25], [ 3,23], [ 4,20], [ 6,16], [10,12] ]  # 27. 62  111110
], dtype=np.int8)

# S6 carries the {+1, -1} swap-parity signs that accompany each pair
# in T6 for the Euclidean case Cl(6, 0, 0).  Row ``i`` aligns with
# T6 row ``i``: the contribution of pair ``(p1, p2)`` to output slot
# ``I6[i+1]`` is ``S6[i, j] · (X[p1]·Y[p2] + X[p2]·Y[p1])`` — the
# symmetric pair combination, since the self-reverse projection of a
# product is symmetric in its arguments.
#
# For any other signature, reinit() overwrites S6 in place with
# {-1, 0, +1} values pulled from the corresponding full sign table.
# This array is the *base* for that operation and is intended to stay
# untouched at module-load time.
S6 = np.array([
    [ 1,  1,  1,  1,  1,  1] , #  1.  1  000001
    [ 1, -1, -1, -1, -1,  1] , #  2.  2  000010
    [ 1,  1,  1,  1, -1, -1] , #  3.  4  000100
    [ 1, -1, -1,  1,  1,  1] , #  4.  8  001000
    [ 1,  1,  1, -1,  1, -1] , #  5. 15  001111
    [ 1,  1, -1, -1, -1, -1] , #  6. 16  010000
    [ 1, -1,  1,  1, -1,  1] , #  7. 23  010111
    [ 1,  1,  1, -1,  1, -1] , #  8. 27  011011
    [ 1, -1,  1,  1, -1,  1] , #  9. 29  011101
    [ 1,  1,  1, -1,  1, -1] , # 10. 30  011110
    [ 1,  1, -1,  1, -1,  1] , # 11. 31  011111
    [ 1,  1,  1,  1,  1,  1] , # 12. 32  100000
    [ 1, -1, -1, -1,  1, -1] , # 13. 39  100111
    [ 1,  1, -1,  1, -1,  1] , # 14. 43  101011
    [ 1, -1, -1, -1,  1, -1] , # 15. 45  101101
    [ 1,  1, -1,  1, -1,  1] , # 16. 46  101110
    [ 1,  1, -1,  1, -1,  1] , # 17. 47  101111
    [ 1,  1,  1, -1,  1, -1] , # 18. 51  110011
    [ 1, -1,  1,  1, -1,  1] , # 19. 53  110101
    [ 1,  1,  1, -1,  1, -1] , # 20. 54  110110
    [ 1,  1, -1,  1, -1,  1] , # 21. 55  110111
    [ 1, -1, -1, -1,  1, -1] , # 22. 57  111001
    [ 1,  1, -1,  1, -1,  1] , # 23. 58  111010
    [ 1,  1, -1,  1, -1,  1] , # 24. 59  111011
    [ 1,  1,  1, -1,  1, -1] , # 25. 60  111100
    [ 1,  1, -1,  1, -1,  1] , # 26. 61  111101
    [ 1,  1, -1,  1, -1,  1]   # 27. 62  111110
], dtype=np.int8)


def reinit(table: np.ndarray) -> None:
    """Rebuild S6 and S6_0 from a 64×64 sign table for any 6D signature.

    Parameters
    ----------
    table : np.ndarray, shape (64, 64), dtype int8
        Full Clifford-product sign table for ``Cl(p, q, r)`` with
        ``p + q + r = 6``: ``table[i, j]`` is the {-1, 0, +1} sign with
        which ``e_i · e_j`` produces a basis blade (or zero, in the
        degenerate case ``r > 0``).  The kernel reads from exactly the
        entries ``table[I6[i], I6[i]]`` (diagonal) and ``table[I6[p1],
        I6[p2]]`` for the off-diagonal pairs in T6.

    Notes
    -----
    The sign data are modified *in place*, so an already-compiled Numba
    kernel that closes over these arrays would automatically pick up
    the new values on its next call.  We additionally clear the kernel
    cache here so the next ``_get_kernel()`` returns a freshly built
    closure — defensive but free in steady state.
    """
    global _cached_kernel
    for i in range(len(S6)):
        S6_0[i] = table[I6[i]][I6[i]]
        for j in range(len(S6[i])):
            (x, y) = T6[i][j]
            S6[i][j] = table[I6[x]][I6[y]]
    _cached_kernel = None  # force rebuild on next _get_kernel() call


# ===============================================================
# Numba kernel factory
# ===============================================================

def _make_sri6_kernel(I6, T6, S6, S6_0):
    """Factory that closes a Numba kernel over the active index/sign tables.

    The factory pattern mirrors :func:`jones._make_jones_kernel`: the
    closed-over arrays are read by Numba at call time, so in-place
    edits made by :func:`reinit` propagate without recompilation, but
    the closure makes the dependency explicit and pairs cleanly with
    the cache-invalidation step in ``reinit``.

    Parameters
    ----------
    I6 : np.ndarray, shape (28,)
        Linear indices of the self-reverse blades in the full 64-slot
        register.
    T6 : np.ndarray, shape (27, 6, 2)
        Pair-contribution table for the 27 non-scalar output slots.
    S6 : np.ndarray, shape (27, 6)
        Sign multipliers for the off-diagonal pair sums.
    S6_0 : np.ndarray, shape (28,)
        Diagonal sign multipliers for the scalar-part contribution.

    Returns
    -------
    callable
        An ``@njit``-compiled function ``_sri6(H64) -> np.ndarray``
        that returns the inverse of a 6D self-reverse multivector.
    """

    @njit(error_model='numpy')
    def _smul_6(X: np.ndarray, Y: np.ndarray) -> np.float64:
        """Scalar part ``<X·Y>_0`` for self-reverse X, Y (compact form).

        The scalar part of any Clifford product is the diagonal sum
        ``Σ_α (e_α · e_α) X_α Y_α``, with the per-blade signs supplied
        by ``S6_0``.  No off-diagonal pair contributes to the scalar
        slot in the self-reverse subspace.
        """
        a = 0.0
        for i in range(I6.shape[0]):
            a += S6_0[i] * X[i] * Y[i]
        return a

    @njit(error_model='numpy')
    def _mul_6(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Self-reverse projection of ``X · Y`` in compact 28-slot form.

        For the Jones iteration ``X`` and ``Y`` always commute (one is
        a scalar-shifted copy of the other), so the projection in fact
        equals the full product.  The symmetric pair combination
        ``X[p1]·Y[p2] + X[p2]·Y[p1]`` is the natural form of the
        projection and halves the multiplicative work.
        """
        n = I6.shape[0]
        Z = np.zeros(n, dtype=X.dtype)

        # Scalar slot: same diagonal sum as in _smul_6, inlined to spare
        # the function-call overhead on Numba's hottest path.
        a = 0.0
        for i in range(n):
            a += S6_0[i] * X[i] * Y[i]
        Z[0] = a

        # Non-scalar slots: T6 row i feeds Z[i+1] with six pair sums.
        for i in range(S6.shape[0]):
            a = 0.0
            for j in range(S6.shape[1]):
                p1 = T6[i, j, 0]
                p2 = T6[i, j, 1]
                a += S6[i, j] * (X[p1] * Y[p2] + X[p2] * Y[p1])
            Z[i + 1] = a
        return Z

    @njit(error_model='numpy')
    def _sri6(H64: np.ndarray) -> np.ndarray:
        """Inverse of a 6D self-reverse multivector.

        Parameters
        ----------
        H64 : np.ndarray, shape (64,)
            Coefficient array of a self-reverse multivector.  Slots
            outside ``I6`` are not read.

        Returns
        -------
        np.ndarray, shape (64,)
            Coefficient array of ``H^{-1}``, also self-reverse.
            Contains ``inf`` or ``nan`` if ``H`` is singular; the
            public wrapper :func:`sri6_inverse` converts this to a
            ``ValueError``.

        Algorithm
        ---------
        Pack ``H64`` into the compact 28-slot self-reverse form, then
        run the Jones adjugate iteration.  Each step multiplies the
        running accumulator ``S`` by ``H``; before each multiplication
        the scalar slot is rescaled by a fixed coefficient determined
        by the characteristic polynomial of ``H`` in the self-reverse
        subspace.  After three multiplications, ``S`` holds the adju-
        gate of ``H``; a final scalar product with ``H`` yields the
        determinant ``d``, and ``H^{-1} = S / d``.

        Note that throughout the iteration ``S`` and ``H`` commute (S
        only ever differs from a scalar multiple of H by additive
        scalar shifts), so the self-reverse projection performed by
        ``_mul_6`` coincides with the full Clifford product — no
        information is dropped.
        """
        n = I6.shape[0]

        # Pack H64 into compact 28-slot form.
        H = np.empty(n, dtype=H64.dtype)
        for i in range(n):
            H[i] = H64[I6[i]]

        # Jones adjugate iteration on the compact form.
        S = H.copy()
        S[0] *= -3.0           # rescale scalar slot
        S = _mul_6(S, H)       # S ← S · H   (1st multiply)
        S[0] *= -1.0           # rescale scalar slot
        S = _mul_6(S, H)       # S ← S · H   (2nd multiply)
        S[0] *= (-1.0 / 3.0)   # rescale scalar slot
        # S is now the adjugate of H in the self-reverse subspace.

        d = _smul_6(S, H)      # determinant: scalar of S · H
        rd = 1.0 / d           # singular ⇒ inf (numpy error model)

        # Unpack compact result back to full 64-slot form.
        iH64 = np.zeros_like(H64)
        for i in range(n):
            iH64[I6[i]] = rd * S[i]
        return iH64

    return _sri6


# ===============================================================
# Module-level kernel cache
# ===============================================================

_cached_kernel = None


def _get_kernel():
    """Return the cached kernel, building it on first call (or after reinit)."""
    global _cached_kernel
    if _cached_kernel is None:
        _cached_kernel = _make_sri6_kernel(I6, T6, S6, S6_0)
    return _cached_kernel


# ===============================================================
# Public interface
# ===============================================================

def sri6_inverse(H64: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 6D self-reverse multivector.

    Parameters
    ----------
    H64 : np.ndarray
        64-element coefficient array of a self-reverse multivector in
        ``Cl(p, q, r)``, ``p + q + r = 6``.  Only positions indexed
        by :data:`I6` (grades 0, 1, 4, 5) are read; other slots are
        ignored.

    Returns
    -------
    np.ndarray
        64-element coefficient array of ``H^{-1}``, also self-reverse.

    Raises
    ------
    ValueError
        If ``H`` is singular (the scalar determinant vanishes).

    Notes
    -----
    Cost: three compact multivector multiplies plus one compact scalar
    product, each operating on 28-slot arrays.  The full 64-slot
    output array is materialised only once, at the very end, via the
    unpack step in :func:`_sri6`.

    For algebras of dimension ≠ 6 (or for multivectors that are not
    self-reverse), use the general :func:`jones.jones_inverse`
    instead; this kernel makes no attempt to validate either
    precondition.
    """
    kernel = _get_kernel()
    result = Accum()
    result.Reg = kernel(H64.Reg)

    if not np.isfinite(result.Reg).all():
        raise ValueError(
            "sri6_inverse: multivector is singular "
            "(scalar denominator is zero)."
        )

    return result
