# file: clifford/sign_table.py
"""
Multiplication sign table and Numba-jitted fast multiplier.

The geometric product of two basis blades ``e_I`` and ``e_J`` (identified by
their ordinal bitmasks ``I`` and ``J``) is always a scalar multiple of a
single basis blade ``e_{I XOR J}``.  The scalar is either +1, −1, or 0
(for degenerate algebras).  This module pre-computes those scalars into a
two-dimensional NumPy array and generates a Numba-jitted function that
uses the table to multiply arbitrary multivectors efficiently.

Sign computation
----------------
The sign of ``e_I * e_J`` is determined by two factors:

1. **Metric sign** — the product of the squares of all basis vectors shared
   by ``I`` and ``J`` (i.e. in ``I AND J``).  A shared vector with negative
   signature contributes a factor of −1; a degenerate shared vector makes the
   whole product zero.

2. **Swap count** — the number of adjacent transpositions needed to move the
   vectors of ``J`` past those of ``I`` when writing the concatenated product
   in canonical (ascending) order.  Each transposition contributes a factor of
   −1 via the anti-commutativity of basis vectors.

The combined sign is ``(-1) ** (metric_sign_count + swap_count)``.

Dimensions limit
----------------
For dimensions < 8 the full ``2^d × 2^d`` sign table fits comfortably in
memory (at most 128 × 128 = 16 384 bytes as ``int8``), and the Numba
multiplier is generated at initialisation time.  For dimensions ≥ 8 the
table and fast multiplier are left as ``None``; a different strategy (e.g.
on-the-fly sign computation) must be supplied by the caller.
"""

import numpy as np
from numba import njit
import clifford.context as Clif


# ---------------------------------------------------------------------------
# Sign computation (pure Python, runs once at initialisation)
# ---------------------------------------------------------------------------

def _swap_count(bits_left: int, bits_right: int) -> int:
    """Count the number of anti-commutation swaps for ``e_I * e_J``.

    When concatenating the basis vectors of ``e_J`` to the right of those of
    ``e_I``, each vector of ``e_J`` must step past the vectors of ``e_I`` that
    sit to its right in the canonical ordering.  Each such step is one swap and
    contributes a sign change.

    Parameters
    ----------
    bits_left : int
        Ordinal bitmask of the left blade ``e_I``.
    bits_right : int
        Ordinal bitmask of the right blade ``e_J``.

    Returns
    -------
    int
        Total number of swaps required.
    """
    n_swap = 0
    n_jump = Clif.Grade[bits_right]   # number of vectors still to be placed
    mask   = 1
    while n_jump != 0:
        if mask & bits_right:
            n_jump -= 1
        if mask & bits_left:
            n_swap += n_jump
        mask <<= 1
    return n_swap


def _fill_table(size: int, degenerate: int, signature: int) -> np.ndarray:
    """Build the full sign table as an ``int8`` NumPy array.

    Parameters
    ----------
    size : int
        Side length of the square table; equals ``2 ** dimensions``.
    degenerate : int
        Degeneracy bitmask from :mod:`clifford.context`.
    signature : int
        Signature bitmask from :mod:`clifford.context`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(size, size)`` with dtype ``int8``.
        Entry ``[i, j]`` is +1, −1, or 0.
    """
    table = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            common = i & j
            if common & degenerate:
                # A degenerate basis vector appears in both blades → product is 0
                v = 0
            else:
                # Count negative-signature shared vectors
                n = Clif.Grade[common & signature]
                # Add anti-commutation swaps
                n += _swap_count(i, j)
                v = -1 if (n & 1) else 1
            table[i][j] = v
    return table


# ---------------------------------------------------------------------------
# SignTable class
# ---------------------------------------------------------------------------

class SignTable:
    """Pre-computed multiplication sign table for the current algebra context.

    An instance is created automatically by :func:`clifford.context.Initialize`
    and stored in :data:`clifford.context._ActiveTable`.  User code does not
    normally need to instantiate this class directly.

    Attributes
    ----------
    dimensions : int
        Algebra dimension at the time of construction.
    size : int
        Number of basis blades; equals ``2 ** dimensions``.
    signature : int
        Signature bitmask at the time of construction.
    degenerate : int
        Degeneracy bitmask at the time of construction.
    table : numpy.ndarray or None
        The ``(size, size)`` int8 sign table.  ``None`` for dimensions ≥ 8.
    fast_mul : callable or None
        Numba-jitted function ``(u_reg, v_reg) -> result_reg`` implementing
        multivector multiplication.  ``None`` for dimensions ≥ 8.
    """

    def __init__(self) -> None:
        self.dimensions = Clif.dimensions
        self.size       = 1 << Clif.dimensions
        self.signature  = Clif.signature
        self.degenerate = Clif.degenerate

        if self.dimensions < 8:
            self.table    = _fill_table(self.size, self.degenerate, self.signature)
            self.fast_mul = self._create_multiplier()
        else:
            self.table    = None
            self.fast_mul = None

    def _create_multiplier(self):
        """Generate a Numba-jitted multiplier closed over the sign table.

        Returns
        -------
        callable
            A function ``multiply(u_reg, v_reg) -> numpy.ndarray`` that
            computes the geometric product of two multivectors given as
            coefficient arrays.

        Notes
        -----
        The table is captured by reference into the Numba closure at the time
        this method is called.  Subsequent calls to :func:`Initialize` create
        a *new* :class:`SignTable` instance and a new closure; any
        :class:`~clifford.multivector.Accum` objects still holding a reference
        to the old table will continue to use the old multiplier.
        """
        table = self.table   # captured by the closure below

        @njit
        def multiply(u_reg: np.ndarray, v_reg: np.ndarray) -> np.ndarray:
            """Geometric product of two multivector coefficient arrays.

            Parameters
            ----------
            u_reg : numpy.ndarray
                Coefficient array of the left multivector.
            v_reg : numpy.ndarray
                Coefficient array of the right multivector.

            Returns
            -------
            numpy.ndarray
                Coefficient array of the product, same shape as inputs.
            """
            res = np.zeros_like(u_reg)
            for i in range(len(u_reg)):
                if u_reg[i] == 0.0:
                    continue
                row = table[i]          # contiguous row — cache-friendly
                for j in range(len(v_reg)):
                    res[i ^ j] += u_reg[i] * (row[j] * v_reg[j])
            return res

        return multiply
