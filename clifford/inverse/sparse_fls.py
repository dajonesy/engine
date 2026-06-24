# file: clifford/inverse/sparse_fls.py
"""
Sparse FLS inverse using compact pair-table Numba kernels.

For each dimension, the positive eigenspace of the FLS involution (the
"active subspace") is identified once and cached.  A pair table is built
over that subspace using the virtual Euclidean sign formula, and a Numba
kernel is closed over those tables.  The FLS polynomial loop then runs
entirely in the compact active subspace; only two full-size geometric
products appear at the boundaries.

Cost per inverse at dimension d
--------------------------------
Dense:   (N + 2) full products   N = 2^(d//2 - 1)
Sparse:  2 full products  +  N compact products
         compact product cost ≈ M * P * 2 muls
         where M = |active subspace|, P = max pairs per output slot

Tables are built lazily on first call per dimension and cached for the
lifetime of the Python session.
"""

import numpy as np
from numba import njit
import clifford.context as Clif
from clifford.multivector import Accum
from clifford.util import inverse_involution
from clifford.inverse.fls import _make_tweak


_table_cache: dict = {}


# ---------------------------------------------------------------------------
# Sign helper
# ---------------------------------------------------------------------------

def _euc_sign(row_key_a: int, b: int) -> int:
    """Euclidean geometric-product sign: (-1)^popcount(row_keys[a] & b)."""
    return 1 - 2 * (bin(row_key_a & b).count('1') % 2)


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def _build_tables(dim: int) -> dict:
    """Build dimension-specific sparse FLS tables.

    Parameters
    ----------
    dim : int
        Algebra dimension, 6 <= dim <= 13.

    Returns
    -------
    dict
        Keys: I, T, S, S0, counts, tweaks, invol, pseudo_pos, kernel.
    """
    size     = 1 << dim
    invol    = inverse_involution(dim)
    row_keys = Clif._ActiveTable.row_keys

    # Active subspace: positive eigenspace of the FLS involution.
    active   = np.where(invol > 0)[0].astype(np.int32)
    M        = len(active)

    # Reverse map: blade index → compact position (-1 if not active).
    blade_to_pos           = np.full(size, -1, dtype=np.int32)
    blade_to_pos[active]   = np.arange(M, dtype=np.int32)

    # Pair-contribution lists: rows[k] = [(p1, p2, sign), ...] for output active[k].
    rows: list = [[] for _ in range(M)]

    for p1 in range(M - 1):
        a    = int(active[p1])
        rk_a = int(row_keys[a])
        tail  = active[p1 + 1:]
        xors  = tail ^ a                     # XOR with all subsequent active blades
        slots = blade_to_pos[xors]           # compact index for each XOR, or -1
        hits  = np.where(slots >= 0)[0]
        for vi in hits:
            p2   = int(p1 + 1 + vi)
            slot = int(slots[vi])
            b    = int(active[p2])
            s    = _euc_sign(rk_a, b)
            rows[slot].append((p1, p2, s))

    assert int(active[0]) == 0,  "scalar must be first active blade"
    assert rows[0] == [],        "no off-diagonal pairs land on scalar slot"

    max_pairs = max((len(r) for r in rows[1:]), default=0)

    T      = np.zeros((M - 1, max_pairs, 2), dtype=np.int16)
    S      = np.zeros((M - 1, max_pairs),    dtype=np.int8)
    counts = np.zeros(M - 1,                 dtype=np.int32)

    for i in range(M - 1):
        k    = i + 1
        cnt  = len(rows[k])
        counts[i] = cnt
        for j, (p1, p2, s) in enumerate(rows[k]):
            T[i, j, 0] = p1
            T[i, j, 1] = p2
            S[i, j]    = s

    # Diagonal signs: sign(e_a * e_a) → scalar coefficient.  In Euclidean Cl(n,0)
    # this equals (-1)^(k*(k-1)/2) for a grade-k blade; computed via row_keys.
    S0 = np.array(
        [_euc_sign(int(row_keys[a]), int(a)) for a in active],
        dtype=np.int8,
    )

    I_arr      = active                   # int32[M]
    tweaks     = _make_tweak(dim)
    pseudo_pos = int(blade_to_pos[size - 1])   # -1 for even dims (not active)

    kernel = _make_compact_mul(I_arr, T, S, S0, counts)

    return {
        'I':          I_arr,
        'T':          T,
        'S':          S,
        'S0':         S0,
        'counts':     counts,
        'tweaks':     tweaks,
        'invol':      invol,
        'pseudo_pos': pseudo_pos,
        'kernel':     kernel,
    }


def _get_tables(dim: int) -> dict:
    """Return cached tables for dim, building on first call."""
    if dim not in _table_cache:
        _table_cache[dim] = _build_tables(dim)
    return _table_cache[dim]


# ---------------------------------------------------------------------------
# Numba kernel factory
# ---------------------------------------------------------------------------

def _make_compact_mul(
    I_arr:  np.ndarray,
    T:      np.ndarray,
    S:      np.ndarray,
    S0:     np.ndarray,
    counts: np.ndarray,
):
    """Return a jitted compact multiply closed over the active-subspace tables.

    Parameters
    ----------
    I_arr  : int32[M]            compact → full index map; I_arr[0] = 0 (scalar)
    T      : int16[M-1, P, 2]   pair indices (p1, p2) per non-scalar output slot
    S      : int8[M-1, P]       pair signs for symmetric combination
    S0     : int8[M]            diagonal signs: sign(e_a * e_a) for each active blade
    counts : int32[M-1]         valid pairs per non-scalar output slot

    Returns
    -------
    callable
        @njit  compact_mul(X, Y) -> Z,  all arrays of length M (float64).
    """

    @njit
    def compact_mul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        M = I_arr.shape[0]
        Z = np.zeros(M, dtype=np.float64)

        # Scalar slot: diagonal sum  <X·Y>_0 = Σ_i S0[i] * X[i] * Y[i]
        s = 0.0
        for i in range(M):
            s += S0[i] * X[i] * Y[i]
        Z[0] = s

        # Non-scalar slots: symmetric pair sums
        for i in range(M - 1):
            a   = 0.0
            cnt = counts[i]
            for j in range(cnt):
                p1 = T[i, j, 0]
                p2 = T[i, j, 1]
                a += S[i, j] * (X[p1] * Y[p2] + X[p2] * Y[p1])
            Z[i + 1] = a

        return Z

    return compact_mul


# ---------------------------------------------------------------------------
# Sparse FLS inverses
# ---------------------------------------------------------------------------

def sparse_even_inverse(A: Accum) -> Accum:
    """Sparse FLS inverse for even Euclidean dimensions (6, 8, 10, 12).

    The FLS polynomial loop runs entirely in the compact active subspace.
    Only the two bookend geometric products (A·invol(A) and invol(A)·adj)
    are full-size.

    Parameters
    ----------
    A : Accum
        Non-singular multivector in Cl(dim, 0), dim in {6, 8, 10, 12}.

    Returns
    -------
    Accum
        Approximate inverse.  For dim <= 10 residuals are < 1e-8;
        at dim 12 follow with newton_schulz() for full precision.
    """
    dim  = A.dimensions
    tb   = _get_tables(dim)
    mul  = Clif._ActiveTable.fast_mul

    invol  = tb['invol']
    I_arr  = tb['I']
    tweaks = tb['tweaks']
    cmul   = tb['kernel']

    b      = invol * A.Reg          # invol(A): elementwise, full
    c_full = mul(A.Reg, b)          # C = A * invol(A)  (result in active subspace)
    c      = c_full[I_arr]          # pack into compact form

    d = c.copy()
    d[0] *= -tweaks[0]
    for i in range(1, len(tweaks)):
        d    = cmul(d, c)
        d[0] *= -tweaks[i]

    e  = cmul(d, c)
    e0 = e[0]

    d_full        = np.zeros(1 << dim, dtype=np.float64)
    d_full[I_arr] = d
    rc             = (1.0 / e0) * d_full

    result     = Accum()
    result.Reg = mul(b, rc)
    return result


def sparse_odd_inverse(A: Accum) -> Accum:
    """Sparse FLS inverse for odd Euclidean dimensions (7, 9, 11, 13).

    The pseudoscalar slot is tweaked in parallel with the scalar slot at
    each step; both are guaranteed active for odd dims 7-13.

    Parameters
    ----------
    A : Accum
        Non-singular multivector in Cl(dim, 0), dim in {7, 9, 11, 13}.

    Returns
    -------
    Accum
        Approximate inverse.  For dim <= 11 residuals are < 1e-8;
        at dim 13 follow with newton_schulz() for full precision.
    """
    dim        = A.dimensions
    tb         = _get_tables(dim)
    mul        = Clif._ActiveTable.fast_mul

    invol      = tb['invol']
    I_arr      = tb['I']
    tweaks     = tb['tweaks']
    pseudo_pos = tb['pseudo_pos']
    cmul       = tb['kernel']

    b      = invol * A.Reg
    c_full = mul(A.Reg, b)
    c      = c_full[I_arr]

    d = c.copy()
    d[0]          *= -tweaks[0]
    d[pseudo_pos] *= -tweaks[0]
    for i in range(1, len(tweaks)):
        d              = cmul(d, c)
        d[0]          *= -tweaks[i]
        d[pseudo_pos] *= -tweaks[i]

    e               = cmul(d, c)
    ep              = e.copy()
    ep[pseudo_pos] *= -1.0

    epe = cmul(ep, e)           # only epe[0] (scalar) is used
    epd = cmul(ep, d)           # compact adjugate numerator

    size           = 1 << dim
    epd_full       = np.zeros(size, dtype=np.float64)
    epd_full[I_arr] = epd
    rc              = (1.0 / epe[0]) * epd_full

    result     = Accum()
    result.Reg = mul(b, rc)
    return result


def sparse_fls_inverse(A: Accum) -> Accum:
    """Sparse FLS inverse for Euclidean Cl(dim, 0), dim in 6-13.

    Dispatches to sparse_even_inverse or sparse_odd_inverse.  Tables and
    Numba kernels are built on the first call per dimension and cached.
    For dims 12-13, compose with newton_schulz() for full double precision.

    Parameters
    ----------
    A : Accum
        Non-singular multivector.

    Returns
    -------
    Accum
        Multiplicative inverse of A.

    Raises
    ------
    ValueError
        If dimension is outside 6-13.
    """
    dim = A.dimensions
    if dim < 6 or dim > 13:
        raise ValueError(f"sparse_fls_inverse supports dimensions 6-13; got {dim}")
    return sparse_even_inverse(A) if dim % 2 == 0 else sparse_odd_inverse(A)
