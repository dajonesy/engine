"""
file: self_reverse_7D.py
2026.05.10  DAJones — initial construction, extending self_reverse_6D.py
            to the 7D Abdulkhaev–Shirokov pipeline with the K-tweak
            reaching both scalar and pseudoscalar.

Optimized 7D inverse for the Abdulkhaev–Shirokov working subspace
==================================================================

Motivation
----------
The Abdulkhaev–Shirokov (A&S) 7D inverse routes a general multivector
``U`` through an intermediate ``W = U · b_{2,3,6}(U)``, where
``b_{2,3,6}`` is the Hitzer–Sangwine 6D-grading involution (the one
that negates the 7D blades classified as 6D-grades 2, 3, 6 under that
embedding — *not* standard 7D grades).  The choice of this partial-
reverse rather than the full reverse leaves W with a non-zero pseudo-
scalar, so the iteration's K-"tweak" must touch both the scalar slot
(linear index 0) and the pseudoscalar slot (linear index 127):

    K(W, s)[0]   = -s · W[0]
    K(W, s)[127] = -s · W[127]
    (all other slots unchanged)

After three iteration steps of the form ``K(., c) · W``, the running
accumulator is the adjugate of W in this working subspace.  Multi-
plying by W once more collapses every slot except scalar and pseudo-
scalar — leaving a 2D "complex-like" quantity ``V = v0 + v_p · I``.
A grade-involution-then-multiply step extracts the pure scalar
denominator ``d = v0² − σ · v_p²``, where ``σ = I² = e_{127}²`` is
the pseudoscalar's square (depends on signature; for Cl(7, 0, 0): σ = -1).

The working subspace M7
-----------------------
The slots that *survive* the b_{2,3,6} involution — i.e. those where
``Signs236[i] = +1`` — form a 56-element set in 7D, which we call M7.
Explicitly, M7 = MT[0] ∪ MT[1] ∪ MT[4] ∪ MT[5] in the H–S grading.
Critically, MT[0] = {0, 127}: the scalar and the pseudoscalar are
both in M7, paired together as a 2D "centre".  This is what makes the
K-tweak natural — it acts on the two elements of MT[0] simultaneously.

A useful structural fact: M7 is closed under blade-complement
``a ↦ 127 ⊕ a``.  Every element of M7 has its complement also in M7,
producing exactly 28 unordered complement-pairs.

Pair structure
--------------
For each non-scalar output slot ``M7[k]``, the compact multiply needs
the unordered pairs ``(a, b)`` of compact indices with
``M7[a] ⊕ M7[b] = M7[k]``.  The distribution is almost-uniform:

    • 54 "regular" output slots: exactly 12 pairs each   →  ``U7``, ``S7``
    • 1 "pseudoscalar" output slot (M7[55] = 127): 28 pairs → ``Up7``,
      ``Sp7``  — the 28 complement-pairs (a, 127⊕a)

That asymmetry is the only structural complication on top of the 6D
template, and it costs a single extra loop block in the kernel.

Cost ledger per compact multiply (relative to 16384 for full 7D)
----------------------------------------------------------------
    Scalar slot (56-term diagonal sum):           56 muls
    54 regular slots × 12 pairs × 2 muls:       1296 muls
    Pseudoscalar slot × 28 pairs × 2 muls:        56 muls
    ────────────────────────────────────────────────────
    Total:                                      1408 muls
    Compression ratio:                          11.6×

The full inverse does:
    3 compact multiplies in the iteration:      4224 muls
    Narrow V extraction (scalar + pseudoscalar):  112 muls
    1 compact multiply for S · Vp:              1408 muls
    ────────────────────────────────────────────────────
    Total inner-kernel cost:                    5744 muls

The fourth compact multiply (``S · Vp``) is performed on a sparse Vp
(only slots 0 and PSEUDO_M_IDX are non-zero), so most of those 1408
muls collapse to zero arithmetically.  A future optimisation could
exploit Vp's narrowness via a precomputed complement-index table —
each regular output slot receives exactly two surviving pair contri-
butions, from ``(0, k)`` and ``(m, 55)`` where ``m`` is the M7-com-
plement of ``k`` — saving another ~1300 muls if performance demands
it.  As-is, this lands the kernel at about 11 µs on a sandbox VM.

Signature configurability
-------------------------
The signature data live in two pieces:

    • ``Signs236`` and ``GISigns``: the b_{2,3,6} and grade-involution
      sign vectors.  These are properties of the H–S construction —
      *not* of the algebra signature — so they are constants.
    • Sign tables ``M7_0``, ``S7``, ``Sp7`` (and the diagonal pseudo-
      scalar square σ inside M7_0): these come from the Clifford
      product table for the active signature.

Bootstrap is for Cl(7, 0, 0).  For any other signature ``Cl(p, q, r)``
with ``p + q + r = 7``, call :func:`reinit` with the full 128×128
sign table; it rewrites M7_0, S7, Sp7 in place and invalidates the
cached kernel.

References
----------
Abdulkhaev, K.; Shirokov, D. — talk at ENGAGE 2025.
Acus, A.; Dargys, A. — "Inverse of multivector: Beyond p+q=5
threshold", arXiv:1712.05204.
Hitzer, E.; Sangwine, S. — 7D-via-6D embedding.
Jones, D. A. — "7D Self Reverse" research notebook and
``I7_instrumented.py`` reference implementation, 2026.
``self_reverse_6D.py`` — the 6D template this module extends.
"""

import numpy as np
from numba import njit


# ===============================================================
# Algorithm-defining sign vectors (independent of signature)
# ===============================================================

# Signs236: per-position {+1, -1} mask implementing the b_{2,3,6}
# involution used by the A&S pipeline.  Lifted verbatim from
# I7_instrumented.py.  Negation at position i means that blade is in
# MT[2] ∪ MT[3] ∪ MT[6] under the Hitzer–Sangwine 6D grading.
Signs236 = np.array([
    1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1, -1,
    1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1,
    1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,
    1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1,
    1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1,
    1, -1, -1, -1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,
    1, -1,  1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1
], dtype=np.int8)
assert Signs236.shape == (128,)

# GISigns: per-position {+1, -1} mask implementing the grade
# involution (negate odd grades).  Used to convert ``V`` to ``Vp``
# inside the kernel.  Lifted from I7_instrumented.py.
GISigns = np.array([
    1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1,
    1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,
    1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,
    1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,
    1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,
    1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,  1,
    1, -1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,
    1,  1, -1, -1,  1, -1,  1,  1, -1
], dtype=np.int8)
assert GISigns.shape == (128,)


# ===============================================================
# Working subspace M7 (derived from Signs236)
# ===============================================================

# M7: linear indices of the 56 blades that survive the b_{2,3,6}
# involution — exactly the positions where Signs236 == +1.
# Order is ascending, so M7[0] = 0 (scalar) and M7[-1] = 127
# (pseudoscalar).
M7 = np.array(
    [i for i in range(128) if Signs236[i] == 1],
    dtype=np.int8
)
N_M = len(M7)                       # 56
SCALAR_M_IDX = 0                    # M7[SCALAR_M_IDX] = 0
PSEUDO_M_IDX = N_M - 1              # M7[PSEUDO_M_IDX] = 127
assert M7[SCALAR_M_IDX] == 0
assert M7[PSEUDO_M_IDX] == 127

# Number of "regular" non-scalar non-pseudoscalar output slots:
# M7-indices 1, 2, ..., N_M - 2  (i.e. 54 of them).
N_REG = N_M - 2                     # 54

# 7D-index → M7-index map, used for pair-table derivation.
_M7_INV = {int(M7[i]): i for i in range(N_M)}


# ===============================================================
# Pair-table derivation
# ===============================================================
# For each non-scalar output slot k, enumerate pairs (a, b) of
# M7-indices with M7[a] ⊕ M7[b] = M7[k] and a < b.
#
# Empirically (and provably, from the H-S construction):
#   • k in 1..N_M-2   →   12 pairs each  (the "regular" rows)
#   • k = N_M-1       →   28 pairs       (the pseudoscalar row;
#                                         these are the 28 unordered
#                                         complement-pairs in M7)

_PAIRS_PER_REGULAR = 12
_PAIRS_PER_PSEUDO  = 28


def _derive_pair_tables():
    """Compute the index half of the pair tables from M7 alone.

    Returns
    -------
    U7  : (54, 12, 2) int8  — pair indices for regular output slots
    Up7 : (28, 2)      int8  — pair indices for the pseudoscalar slot
    """
    U7  = np.zeros((N_REG, _PAIRS_PER_REGULAR, 2), dtype=np.int8)
    Up7 = np.zeros((_PAIRS_PER_PSEUDO, 2), dtype=np.int8)

    # Walk every unordered pair of M7-indices and bin by XOR target.
    bins = {int(M7[k]): [] for k in range(1, N_M)}
    for i in range(N_M):
        for j in range(i + 1, N_M):
            x = int(M7[i]) ^ int(M7[j])
            if x in bins:
                bins[x].append((i, j))

    # Fill U7 in M7-index order for k = 1..N_M-2.
    for k_reg, k in enumerate(range(1, N_M - 1)):
        pairs = bins[int(M7[k])]
        if len(pairs) != _PAIRS_PER_REGULAR:
            raise RuntimeError(
                f"Regular slot M7[{k}] = {int(M7[k])} has {len(pairs)} pairs, "
                f"expected {_PAIRS_PER_REGULAR}"
            )
        for j, (a, b) in enumerate(pairs):
            U7[k_reg, j, 0] = a
            U7[k_reg, j, 1] = b

    # Fill Up7 for the pseudoscalar slot.
    pairs = bins[127]
    if len(pairs) != _PAIRS_PER_PSEUDO:
        raise RuntimeError(
            f"Pseudoscalar slot has {len(pairs)} pairs, expected {_PAIRS_PER_PSEUDO}"
        )
    for j, (a, b) in enumerate(pairs):
        Up7[j, 0] = a
        Up7[j, 1] = b

    return U7, Up7


U7, Up7 = _derive_pair_tables()


# ===============================================================
# Bookend-1 (U·Up) pair-index table — signature-independent
# ===============================================================
# For each non-scalar M7 output slot M7[k] (k = 1 .. N_M-1), enumerate
# the 64 unordered pairs (a, b) of full-128-space indices with a^b = M7[k]
# and a < b.  These are *all* the cross-pairs that reinforce in the
# U · b236(U) product: pairs whose XOR lies outside M7 cancel algebraically
# and so don't need to be enumerated.

_PAIRS_PER_UUP = 64


def _derive_uup_pair_table():
    """Compute the pair-index table for the U·Up bookend.

    Returns
    -------
    Uuup : (N_M - 1, 64, 2) int8
        For each non-scalar M7 output slot M7[k+1] (k = 0..N_M-2), the
        64 unordered pairs (a, b) of full-128 indices with a^b = M7[k+1]
        and a < b.  The scalar slot (M7[0] = 0) is handled separately
        via the diagonal sum.
    """
    Uuup = np.zeros((N_M - 1, _PAIRS_PER_UUP, 2), dtype=np.int8)
    for k in range(1, N_M):
        c = int(M7[k])
        j = 0
        for a in range(128):
            b = a ^ c
            if a < b:
                Uuup[k - 1, j, 0] = a
                Uuup[k - 1, j, 1] = b
                j += 1
        if j != _PAIRS_PER_UUP:
            raise RuntimeError(
                f"Slot M7[{k}] = {c}: got {j} pairs, expected {_PAIRS_PER_UUP}"
            )
    return Uuup


Uuup = _derive_uup_pair_table()


# ===============================================================
# Sign tables (depend on the algebra signature)
# ===============================================================

# M7_0: diagonal Clifford-product signs e_{M7[i]} · e_{M7[i]} for the
# 56 M7 slots.  Includes the pseudoscalar's square σ at M7_0[-1].
M7_0 = np.zeros(N_M, dtype=np.int8)

# S7: signs for the 54 regular output rows of U7.
S7  = np.zeros((N_REG, _PAIRS_PER_REGULAR), dtype=np.int8)

# Sp7: signs for the 28 pseudoscalar-row pairs of Up7.
Sp7 = np.zeros(_PAIRS_PER_PSEUDO, dtype=np.int8)

# Diag_uup: per-position sign for the scalar-slot diagonal sum in U·Up.
# Diag_uup[i] = FULL_TABLE[i, i] * Signs236[i].
Diag_uup = np.zeros(128, dtype=np.int8)

# Suup: ±1 signs for the bookend-1 reinforce pairs.  The {-2, +2} pattern
# from S(a,b) = FULL_TABLE[a,b]·Signs236[b] + FULL_TABLE[b,a]·Signs236[a]
# is stored as ±1 here; the doubling factor is folded into the kernel as
# a single ``* 2.0`` per output slot.
Suup = np.zeros((N_M - 1, _PAIRS_PER_UUP), dtype=np.int8)

# Bookend2_signs: precomputed signs for the rU = Up · rW bookend.
# Bookend2_signs[c, j] = FULL_TABLE[M7[j] ^ c, M7[j]] — the sign of the
# rW[j]'s contribution into output slot c.
Bookend2_signs = np.zeros((128, N_M), dtype=np.int8)

# Full 128x128 Clifford-product sign table for the active signature.
# Used at module init time to populate the derived sign tables; not
# referenced by the optimized runtime kernels.
_FULL_TABLE = np.zeros((128, 128), dtype=np.int8)


def _build_clifford_signs(p: int, q: int, r: int = 0) -> np.ndarray:
    """Build the 128×128 Clifford-product sign table for Cl(p, q, r).

    ``table[a, b]`` is the {-1, 0, +1} sign such that
    ``e_a · e_b = table[a, b] · e_{a XOR b}``.  Zero indicates a
    null-generator collision (only possible when r > 0).
    """
    n = p + q + r
    if n != 7:
        raise ValueError(f"Expected p + q + r == 7, got {n}.")
    sq = np.array([1] * p + [-1] * q + [0] * r, dtype=np.int8)

    table = np.zeros((128, 128), dtype=np.int8)
    for a in range(128):
        a_bits = [i for i in range(n) if (a >> i) & 1]
        for b in range(128):
            common = a & b
            # Null-generator collision → zero.
            if any((common >> k) & 1 and sq[k] == 0 for k in range(n)):
                continue
            # Count swaps: for each bit i in a, count bits j < i set in b.
            swaps = sum(bin(b & ((1 << i) - 1)).count('1') for i in a_bits)
            sign = 1 if (swaps & 1) == 0 else -1
            for k in range(n):
                if (common >> k) & 1 and sq[k] == -1:
                    sign = -sign
            table[a, b] = sign
    return table


def _init_signs_from_table(table: np.ndarray) -> None:
    """Refill M7_0, S7, Sp7, Diag_uup, Suup, Bookend2_signs, and _FULL_TABLE
    from a 128×128 Clifford-product sign table."""
    _FULL_TABLE[:] = table.astype(np.int8)

    # Inner-kernel tables.
    for i in range(N_M):
        M7_0[i] = table[M7[i], M7[i]]
    for k_reg in range(N_REG):
        for j in range(_PAIRS_PER_REGULAR):
            a = U7[k_reg, j, 0]
            b = U7[k_reg, j, 1]
            S7[k_reg, j] = table[M7[a], M7[b]]
    for j in range(_PAIRS_PER_PSEUDO):
        a = Up7[j, 0]
        b = Up7[j, 1]
        Sp7[j] = table[M7[a], M7[b]]

    # Bookend-1 tables.  Diagonal sign for scalar-slot sum:
    for i in range(128):
        Diag_uup[i] = table[i, i] * Signs236[i]
    # Cross-pair signs (±1, with the factor of 2 folded into the kernel):
    for k in range(N_M - 1):
        for j in range(_PAIRS_PER_UUP):
            a = int(Uuup[k, j, 0])
            b = int(Uuup[k, j, 1])
            s = table[a, b] * Signs236[b] + table[b, a] * Signs236[a]
            if s not in (-2, 2):
                raise RuntimeError(
                    f"Bookend-1 sign S({a},{b}) = {s}, expected ±2."
                )
            Suup[k, j] = s // 2

    # Bookend-2 sign table.
    for c in range(128):
        for j in range(N_M):
            a = int(M7[j]) ^ c
            Bookend2_signs[c, j] = table[a, M7[j]]


# Bootstrap signs for Cl(7, 0, 0).
_init_signs_from_table(_build_clifford_signs(7, 0, 0))


def reinit(table: np.ndarray) -> None:
    """Rebuild M7_0, S7, Sp7, and _FULL_TABLE from a 128×128 sign table for any 7D signature.

    Modifies the sign data in place and clears the kernel caches so the
    next ``_get_kernel()`` / ``_get_i7_kernel()`` returns a freshly built closure.
    """
    global _cached_kernel, _cached_compact_kernel, _cached_i7_kernel
    _init_signs_from_table(table)
    _cached_kernel = None
    _cached_compact_kernel = None
    _cached_i7_kernel = None


# ===============================================================
# Numba kernel factory
# ===============================================================

def _make_sri7_kernel(M7, U7, S7, Up7, Sp7, M7_0):
    """Factory that closes a Numba kernel over the active sign and pair tables.

    Mirrors ``self_reverse_6D._make_sri6_kernel`` exactly; the only
    novelties are the irregular pseudoscalar row and the K-tweak that
    touches two slots instead of one.
    """

    @njit(error_model='numpy')
    def _smul_7(X, Y):
        """Scalar slot of X · Y for X, Y in M7-compact form (56 slots)."""
        a = 0.0
        for i in range(N_M):
            a += M7_0[i] * X[i] * Y[i]
        return a

    @njit(error_model='numpy')
    def _pmul_7(X, Y):
        """Pseudoscalar slot of X · Y for X, Y in M7-compact form.

        Pulls the 28 complement-pair contributions out of Up7/Sp7.
        """
        a = 0.0
        for j in range(_PAIRS_PER_PSEUDO):
            p1 = Up7[j, 0]
            p2 = Up7[j, 1]
            a += Sp7[j] * (X[p1] * Y[p2] + X[p2] * Y[p1])
        return a

    @njit(error_model='numpy')
    def _mul_7(X, Y):
        """Full M7-compact multiply: returns Z = X · Y on the 56-slot subspace."""
        Z = np.zeros(N_M, dtype=X.dtype)

        # Scalar slot: 56-term diagonal sum.
        a = 0.0
        for i in range(N_M):
            a += M7_0[i] * X[i] * Y[i]
        Z[SCALAR_M_IDX] = a

        # Regular slots: 12 pair sums each, via U7.
        for k_reg in range(N_REG):
            a = 0.0
            for j in range(_PAIRS_PER_REGULAR):
                p1 = U7[k_reg, j, 0]
                p2 = U7[k_reg, j, 1]
                a += S7[k_reg, j] * (X[p1] * Y[p2] + X[p2] * Y[p1])
            Z[k_reg + 1] = a   # +1 because slot 0 is the scalar

        # Pseudoscalar slot: 28 pair sums, via Up7.
        a = 0.0
        for j in range(_PAIRS_PER_PSEUDO):
            p1 = Up7[j, 0]
            p2 = Up7[j, 1]
            a += Sp7[j] * (X[p1] * Y[p2] + X[p2] * Y[p1])
        Z[PSEUDO_M_IDX] = a

        return Z

    @njit(error_model='numpy')
    def _sri7_compact(W):
        """Inverse of an M7-compact (56-slot) multivector.

        Parameters
        ----------
        W : np.ndarray, shape (56,)
            M7-compact coefficient array.

        Returns
        -------
        np.ndarray, shape (56,)
            M7-compact coefficient array of W^{-1}.
        """
        # Jonesian adjugate iteration.  K touches both scalar and
        # pseudoscalar slots — they are the two elements of MT[0].
        S = W.copy()
        S[SCALAR_M_IDX] *= -3.0
        S[PSEUDO_M_IDX] *= -3.0
        S = _mul_7(S, W)                # K(W, 3) · W
        S[SCALAR_M_IDX] *= -1.0
        S[PSEUDO_M_IDX] *= -1.0
        S = _mul_7(S, W)                # K(., 1) · W
        S[SCALAR_M_IDX] *= (-1.0 / 3.0)
        S[PSEUDO_M_IDX] *= (-1.0 / 3.0) # = adjugate (in this subspace)

        # V = S · W collapses to scalar + pseudoscalar only —
        # compute just those two slots.
        v0  = _smul_7(S, W)
        v_p = _pmul_7(S, W)

        # Determinant: with Vp = grade-involute(V) = (v0, -v_p),
        #   d = <Vp · V>_0 = v0² - σ · v_p²    where σ = M7_0[-1]
        sigma = M7_0[PSEUDO_M_IDX]
        d = v0 * v0 - sigma * v_p * v_p
        rd = 1.0 / d                    # singular ⇒ inf (numpy error model)

        # rW = (1/d) · S · Vp.
        Vp = np.zeros(N_M, dtype=W.dtype)
        Vp[SCALAR_M_IDX] = v0
        Vp[PSEUDO_M_IDX] = -v_p
        SVp = _mul_7(S, Vp)
        return rd * SVp

    @njit(error_model='numpy')
    def _sri7(W128):
        """Inverse of an A&S-style 7D multivector W (supported on M7).

        128-slot wrapper around ``_sri7_compact``: packs W128 into compact
        form, calls the compact kernel, and unpacks to 128 slots.

        Parameters
        ----------
        W128 : np.ndarray, shape (128,)
            Coefficient array of a multivector supported on the 56
            slots in M7.  Slots outside M7 are not read; they are
            expected to be zero (as they will be for any W = U·b_{2,3,6}(U)).

        Returns
        -------
        np.ndarray, shape (128,)
            Coefficient array of ``W^{-1}``, also supported on M7.
        """
        # Pack W128 into 56-slot M7-compact form.
        W = np.empty(N_M, dtype=W128.dtype)
        for i in range(N_M):
            W[i] = W128[M7[i]]

        iW = _sri7_compact(W)

        # Unpack to full 128-slot form.
        iW128 = np.zeros_like(W128)
        for i in range(N_M):
            iW128[M7[i]] = iW[i]
        return iW128

    return _sri7, _sri7_compact


# ===============================================================
# Full A&S pipeline kernel factory (general 7D inverse)
# ===============================================================

def _make_i7_kernel(sri7_compact_kernel, M7_arr, signs_236, Uuup_arr,
                    Suup_arr, Diag_uup_arr, B2_signs):
    """Factory: build the full A&S 7D inverse kernel with optimized bookends.

    Composes:
        Up = b_{2,3,6}(U)               (elementwise sign flip)
        W  = U · Up   (bookend 1)       — uses Uuup/Suup/Diag_uup; output
                                          is M7-compact (56 slots)
        rW = sri7_compact(W)            (inner kernel, compact form)
        rU = Up · rW  (bookend 2)       — uses Bookend2_signs; reads only
                                          the 56 rW slots that are in M7

    Parameters
    ----------
    sri7_compact_kernel : numba dispatcher
        The compact-form inner kernel from ``_make_sri7_kernel``.
    M7_arr : np.ndarray, (56,) int8
        M7 slot indices.
    signs_236 : np.ndarray, (128,) int8
        The Signs236 partial-reverse mask.
    Uuup_arr : np.ndarray, (55, 64, 2) int8
        Pair-index table for bookend 1.
    Suup_arr : np.ndarray, (55, 64) int8
        Precomputed ±1 signs for bookend-1 cross pairs.
    Diag_uup_arr : np.ndarray, (128,) int8
        Diagonal signs for the bookend-1 scalar slot.
    B2_signs : np.ndarray, (128, 56) int8
        Precomputed signs for bookend 2.

    Returns
    -------
    numba dispatcher
        Compiled ``_i7(U128) -> rU128`` for general 7D inversion.
    """

    @njit(error_model='numpy')
    def _uup_mul_compact(U):
        """Bookend 1: W = U · b236(U), output in 56-slot M7-compact form.

        Algebraically W lives on M7, so we only fill 56 slots.  Cross-pairs
        whose XOR lies outside M7 cancel exactly, so they're not enumerated.
        For non-scalar slots, each contribution is doubled (one of the two
        orderings) — applied as a single ``* 2.0`` per slot.
        """
        W = np.zeros(N_M, dtype=U.dtype)

        # Scalar slot: 128-term diagonal sum (a = b case only).
        a = 0.0
        for i in range(128):
            a += Diag_uup_arr[i] * U[i] * U[i]
        W[0] = a

        # 55 other M7 slots: 64 reinforce pairs each.
        for k in range(N_M - 1):
            a = 0.0
            for j in range(_PAIRS_PER_UUP):
                p1 = Uuup_arr[k, j, 0]
                p2 = Uuup_arr[k, j, 1]
                a += Suup_arr[k, j] * U[p1] * U[p2]
            W[k + 1] = 2.0 * a

        return W

    @njit(error_model='numpy')
    def _uprw_mul(Up, rW):
        """Bookend 2: rU = Up · rW (full 128-slot output, 56-slot rW input).

        Exploits the fact that rW lives on M7, so the inner loop is only
        56 long instead of 128.  For each output slot c, gather contri-
        butions from the 56 rW slots, using the precomputed sign table.
        """
        rU = np.zeros(128, dtype=Up.dtype)
        for c in range(128):
            a = 0.0
            for j in range(N_M):
                idx = M7_arr[j] ^ c     # full-128 index of Up's contribution
                a += B2_signs[c, j] * Up[idx] * rW[j]
            rU[c] = a
        return rU

    @njit(error_model='numpy')
    def _i7(U128):
        """Full A&S 7D inverse of a general multivector U."""
        # Up = b_{2,3,6}(U): elementwise sign flip per Signs236.
        Up = np.empty(128, dtype=U128.dtype)
        for i in range(128):
            Up[i] = signs_236[i] * U128[i]
        # W = U · Up — direct to M7-compact form (skips 72 zero slots).
        W_compact = _uup_mul_compact(U128)
        # rW = inner inverse, compact form throughout.
        rW_compact = sri7_compact_kernel(W_compact)
        # rU = Up · rW — back to full 128 slots.
        return _uprw_mul(Up, rW_compact)

    return _i7


# ===============================================================
# Module-level kernel cache
# ===============================================================

_cached_kernel = None
_cached_compact_kernel = None
_cached_i7_kernel = None


def _get_kernel():
    """Return the cached inner kernel, building it on first call (or after reinit)."""
    global _cached_kernel, _cached_compact_kernel
    if _cached_kernel is None:
        _cached_kernel, _cached_compact_kernel = _make_sri7_kernel(
            M7, U7, S7, Up7, Sp7, M7_0)
    return _cached_kernel


def _get_compact_kernel():
    """Return the cached M7-compact inner kernel (56-slot in/out)."""
    global _cached_kernel, _cached_compact_kernel
    if _cached_compact_kernel is None:
        _cached_kernel, _cached_compact_kernel = _make_sri7_kernel(
            M7, U7, S7, Up7, Sp7, M7_0)
    return _cached_compact_kernel


def _get_i7_kernel():
    """Return the cached full inverse kernel, building it on first call."""
    global _cached_i7_kernel
    if _cached_i7_kernel is None:
        _cached_i7_kernel = _make_i7_kernel(
            _get_compact_kernel(), M7, Signs236,
            Uuup, Suup, Diag_uup, Bookend2_signs,
        )
    return _cached_i7_kernel


# ===============================================================
# Public interface
# ===============================================================

def sri7_inverse(W128: np.ndarray) -> np.ndarray:
    """Compute the inverse of an A&S-pipeline 7D multivector W.

    Parameters
    ----------
    W128 : np.ndarray, shape (128,)
        Coefficient array of a multivector ``W = U · b_{2,3,6}(U)``,
        which is supported on the 56-slot subspace ``M7``.  Slots
        outside M7 are ignored.

    Returns
    -------
    np.ndarray, shape (128,)
        Coefficient array of ``W^{-1}``, also supported on M7.

    Raises
    ------
    ValueError
        If W is singular (the scalar determinant vanishes).

    Notes
    -----
    Cost: three full M7-compact multivector multiplies (≈1408 real
    muls each), plus two narrow ``V`` / ``S·Vp`` multiplies (~60 muls
    each), plus pack/unpack.  Total ≈4500 real muls before unpacking.

    For a general 7D multivector U: first form ``Up = Signs236 * U``
    and ``W = U · Up`` (full 7D multiplies — the bookends), then call
    ``sri7_inverse(W)`` for the inner inverse, and finally compute
    ``rU = Up · rW``.
    """
    kernel = _get_kernel()
    result = kernel(W128)
    if not np.isfinite(result).all():
        raise ValueError(
            "sri7_inverse: multivector is singular "
            "(scalar denominator is zero)."
        )
    return result


def i7_inverse(U128: np.ndarray) -> np.ndarray:
    """Compute the inverse of a general 7D multivector U via the A&S pipeline.

    This is the user-facing wrapper.  Internally it forms
    ``Up = b_{2,3,6}(U)``, computes ``W = U · Up`` (which lands on the
    56-slot M7 subspace by construction), inverts W with the inner
    ``sri7_inverse`` kernel, and returns ``rU = Up · rW``.

    Parameters
    ----------
    U128 : np.ndarray, shape (128,)
        Coefficient array of a general 7D multivector U.

    Returns
    -------
    np.ndarray, shape (128,)
        Coefficient array of U^{-1}, satisfying ``U · U^{-1} = 1`` (unit
        scalar) and ``U^{-1} · U = 1`` to float64 precision.

    Raises
    ------
    ValueError
        If U is singular (the A&S scalar denominator vanishes).

    Notes
    -----
    Cost: two full 128-slot multiplies (16384 muls each) plus one inner-
    kernel evaluation (~5744 muls).  The bookend multiplies dominate
    runtime; future speedups would target the U·Up bookend, whose output
    is known a priori to live on M7 (so 72 of the 128 output slots are
    redundant).
    """
    kernel = _get_i7_kernel()
    result = kernel(U128)
    if not np.isfinite(result).all():
        raise ValueError(
            "i7_inverse: multivector is singular "
            "(scalar denominator is zero)."
        )
    return result
