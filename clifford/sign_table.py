# file: clifford/sign_table.py
"""
Euclidean keyed virtual sign table for Cl(n, 0).

The geometric product sign table for a pure Euclidean algebra has the
structure of a Walsh-Hadamard matrix: every row and every column is a
Hadamard row identified by a single integer key.  This module computes
those key arrays at setup time — one key per basis blade for rows and
columns separately — and provides Numba-jitted multipliers that
reconstruct sign rows on the fly rather than storing the full table.

Memory cost  2 * 2^n integers  (vs 4^n bytes for a full int8 table)
Setup cost   O(n * 2^n)
Per-product  O(2^n) per sign-row reconstruction inside the inner loop

Key arrays
----------
For Euclidean Cl(n), the sign table satisfies

    sign[i, j]  =  Hadamard[ row_keys[i],  j ]
                =  Hadamard[ col_keys[j],  i ]

where Hadamard is the Sylvester Walsh-Hadamard matrix of order 2^n.

Row key basis  (for each basis vector e_p, p = 0 .. n-1):
    v_key[p] = pseudo XOR ((1 << (p+1)) - 1)
where pseudo = (1 << n) - 1.

Column key basis:
    c_key[p] = (1 << p) - 1

Both key arrays are built by XOR-ing the appropriate basis keys for
each set bit in the blade index i.  See the notebook
scratch/"Signtable -- None vs Keyed vs Tabled.ipynb" for derivation
and validation.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Key-array construction (runs once at context initialisation)
# ---------------------------------------------------------------------------

def euclidean_row_keys(dim: int) -> np.ndarray:
    """Row keys for the Euclidean Cl(dim) sign table.

    row_keys[i] is the Hadamard index k such that
    Hadamard[k, j] == sign(e_i * e_j) for all j.

    Parameters
    ----------
    dim : int

    Returns
    -------
    numpy.ndarray, shape (2**dim,), dtype int32
    """
    pseudo = (1 << dim) - 1
    v_keys = [pseudo ^ ((1 << (p + 1)) - 1) for p in range(dim)]
    size = 1 << dim
    keys = np.zeros(size, dtype=np.int32)
    for i in range(size):
        k = 0
        for p in range(dim):
            if i & (1 << p):
                k ^= v_keys[p]
        keys[i] = k
    return keys


def euclidean_col_keys(dim: int) -> np.ndarray:
    """Column keys for the Euclidean Cl(dim) sign table.

    col_keys[j] is the Hadamard index k such that
    Hadamard[k, i] == sign(e_i * e_j) for all i.

    Parameters
    ----------
    dim : int

    Returns
    -------
    numpy.ndarray, shape (2**dim,), dtype int32
    """
    c_keys = [(1 << p) - 1 for p in range(dim)]
    size = 1 << dim
    keys = np.zeros(size, dtype=np.int32)
    for j in range(size):
        k = 0
        for p in range(dim):
            if j & (1 << p):
                k ^= c_keys[p]
        keys[j] = k
    return keys


# ---------------------------------------------------------------------------
# Hadamard row reconstruction (Numba, called inside jitted multipliers)
# ---------------------------------------------------------------------------

@njit(inline='always')
def _row_from_key(key: int, signs: np.ndarray) -> None:
    """Fill signs[] with the Walsh-Hadamard row indexed by key.

    Uses the Sylvester construction: each successive bit of key controls
    whether the next block is copied (+1) or negated (-1).

    Parameters
    ----------
    key : int
        Hadamard row index.
    signs : numpy.ndarray
        Pre-allocated int8 buffer; length must equal 2^dim.
    """
    signs[0] = 1
    m = 1
    k = key
    size = len(signs)
    while m < size:
        if k & 1:
            for j in range(m):
                signs[m + j] = -signs[j]
        else:
            for j in range(m):
                signs[m + j] = signs[j]
        m <<= 1
        k >>= 1


# ---------------------------------------------------------------------------
# Multiplier factories
# ---------------------------------------------------------------------------

def make_row_multiplier(row_keys: np.ndarray):
    """Return a jitted geometric-product function, row-keyed.

    Iterates over components of the left operand u.  Preferred when u
    is dense or when the standard left-to-right product order is needed.

    Parameters
    ----------
    row_keys : numpy.ndarray
        From euclidean_row_keys(dim).

    Returns
    -------
    callable  (u: float64[::1], v: float64[::1]) -> float64[::1]
    """
    @njit
    def multiply(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = len(u)
        res = np.zeros(n, dtype=np.float64)
        signs = np.empty(n, dtype=np.int8)
        for i in range(n):
            if u[i] == 0.0:
                continue
            _row_from_key(row_keys[i], signs)
            for j in range(n):
                res[i ^ j] += u[i] * signs[j] * v[j]
        return res
    return multiply


def make_col_multiplier(col_keys: np.ndarray):
    """Return a jitted geometric-product function, column-keyed.

    Iterates over components of the right operand v.  Preferred when v
    is sparse, avoiding sign-row reconstruction for zero-coefficient blades.

    Parameters
    ----------
    col_keys : numpy.ndarray
        From euclidean_col_keys(dim).

    Returns
    -------
    callable  (u: float64[::1], v: float64[::1]) -> float64[::1]
    """
    @njit
    def multiply_col(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = len(u)
        res = np.zeros(n, dtype=np.float64)
        signs = np.empty(n, dtype=np.int8)
        for j in range(n):
            if v[j] == 0.0:
                continue
            _row_from_key(col_keys[j], signs)
            for i in range(n):
                res[i ^ j] += signs[i] * u[i] * v[j]
        return res
    return multiply_col


# ---------------------------------------------------------------------------
# SignTable class
# ---------------------------------------------------------------------------

class SignTable:
    """Euclidean keyed virtual sign table for Cl(dim, 0).

    Instantiated by clifford.context.Initialize; user code does not
    normally construct this directly.

    Attributes
    ----------
    dim : int
    size : int            2**dim
    row_keys : ndarray    shape (size,), dtype int32
    col_keys : ndarray    shape (size,), dtype int32
    fast_mul : callable   jitted row-keyed geometric product
    fast_mul_col : callable   jitted column-keyed geometric product
    """

    def __init__(self, dim: int) -> None:
        self.dim          = dim
        self.size         = 1 << dim
        self.row_keys     = euclidean_row_keys(dim)
        self.col_keys     = euclidean_col_keys(dim)
        self.fast_mul     = make_row_multiplier(self.row_keys)
        self.fast_mul_col = make_col_multiplier(self.col_keys)
