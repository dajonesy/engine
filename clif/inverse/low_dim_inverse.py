# file: clif/inverse/low_dim_inverse.py
r"""Closed-form inverse for $d = 0 \dots 3$, by Clifford conjugation.

For $d \le 3$, multiplying a multivector by its Clifford conjugate
$\bar{A}$ lands in the center of the algebra, in every signature:

    d = 0, 1, 2    A Abar = alpha              (a scalar)
    d = 3          A Abar = alpha + beta I     (scalar + pseudoscalar)

so the inverse needs no product at all beyond those central components:

    d  = 2    A^{-1} = Abar / alpha
    d  = 3    A^{-1} = Abar (alpha - beta I) / (alpha^2 - I^2 beta^2)

At d = 0 and 1 the algebra is its own center, so no product is formed at
all: d = 0 is 1/a, and d = 1 writes A = a + b I (I = e_1) and applies the
d = 3 center formula directly, A^{-1} = (a - b I) / (a^2 - I^2 b^2).
These two run in plain Python, with no compiled kernel.

Clifford conjugation is the grade sign $(-1)^{k(k+1)/2}$: $+,-,-,+$ for
grades $0,1,2,3$.  It is the tau-involution at these dimensions (see the
paper's table of pair-less dimensions).

Cost: $2^d$ multiplies for alpha, $2^d$ for beta and $2^{d+1}$ for the
final scaling at $d = 3$; no table beyond one $4 \times 2^d$ sign array.
At these sizes the call itself (Python and Numba dispatch, a few hundred
nanoseconds) costs more than the arithmetic (tens of nanoseconds).
Degenerate signatures are supported: a null generator zeroes the signs
that square it, and a multivector with no inverse raises ValueError.

Usage
-----
    inv = LowDimInverse(d, sig)       # 0 <= d <= 3, sig entries -1, 0, 1
    B = inv.inverse(A)                # A, B: ordinal arrays of length 2**d

or, using the registered context's signature,

    from clif.inverse.low_dim_inverse import inverse
    B = inverse(A)
"""
import numpy as np

try:
    from numba import njit
except ImportError:                                    # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]

from .sparse_fls import blade_sign

_CONJ = (1.0, -1.0, -1.0, 1.0)                         # by grade 0..3


# Sign table T, one row per sign array, length 2**d:
#   T[0] = cs   Clifford-conjugation sign of e_k
#   T[1] = sq   sign of e_k e_k              (0 if e_k holds a null generator)
#   T[2] = sp   sign of e_k e_{k^PS}         (d = 3 only)
#   T[3] = si   sign of e_k I                (d = 3 only)
# Packed into one array so each call passes few arguments: at these sizes
# Numba's per-argument dispatch cost exceeds the arithmetic.

@njit(cache=True)
def _inverse_le2(A, T):
    """d <= 2: Abar / alpha. Returns (out, alpha); alpha == 0: not invertible."""
    out = np.zeros(A.shape[0])
    alpha = 0.0
    for k in range(A.shape[0]):
        alpha += T[0, k] * T[1, k] * A[k] * A[k]
    if alpha != 0.0:
        r = 1.0 / alpha
        for k in range(A.shape[0]):
            out[k] = T[0, k] * A[k] * r
    return out, alpha


@njit(cache=True)
def _inverse_3(A, T, I2):
    """d = 3: Abar (alpha - beta I) / det. Returns (out, det)."""
    out = np.zeros(8)
    alpha = 0.0
    beta = 0.0
    for k in range(8):
        alpha += T[0, k] * T[1, k] * A[k] * A[k]               # e_k e_k
        beta += T[2, k] * A[k] * T[0, k ^ 7] * A[k ^ 7]        # e_k e_{k^7}
    det = alpha * alpha - I2 * beta * beta
    if det != 0.0:
        a = alpha / det
        b = -beta / det
        for k in range(8):
            out[k] = a * T[0, k] * A[k]
        for k in range(8):                     # Abar I: e_k I = si[k] e_{k^7}
            out[k ^ 7] += b * T[3, k] * T[0, k] * A[k]
    return out, det


class LowDimInverse:
    """Closed-form inverse for Cl(sig), 0 <= d <= 3."""

    def __init__(self, d, sig):
        """Same call shape as TauInverse(d, sig), so tau_api can dispatch to it."""
        sig = [int(s) for s in sig]
        assert len(sig) == d
        assert 0 <= d <= 3, "LowDimInverse covers d = 0..3"
        assert all(s in (-1, 0, 1) for s in sig), "sig entries must be -1, 0, 1"
        self.d, self.sig, self.n = d, sig, 1 << d
        n, PS = self.n, self.n - 1
        self.T = np.array([
            [_CONJ[bin(k).count("1")] for k in range(n)],
            [float(blade_sign(k, k, sig)) for k in range(n)],
            [float(blade_sign(k, k ^ PS, sig)) for k in range(n)],
            [float(blade_sign(k, PS, sig)) for k in range(n)],
        ])
        self.I2 = float(blade_sign(PS, PS, sig))

    def inverse(self, A):
        """Inverse of an ordinal float64 array A (length 2**d). Returns a new array."""
        d = self.d
        # d = 0, 1: Python floats (.item / .tolist), not NumPy scalars, which
        # are several times slower to do arithmetic on.
        if d == 0:                                     # a scalar
            a = A.item()
            if a == 0.0:
                raise ValueError("multivector is not invertible")
            return np.array([1.0 / a])
        if d == 1:                                     # A = a + b I is its own center
            a, b = A.tolist()
            det = a * a - self.I2 * b * b
            if det == 0.0:
                raise ValueError("multivector is not invertible")
            return np.array([a / det, -b / det])
        if d == 3:
            out, det = _inverse_3(A, self.T, self.I2)
        else:
            out, det = _inverse_le2(A, self.T)
        if det == 0.0:
            raise ValueError("multivector is not invertible")
        return out


_CACHE = {}


def inverse(A, sig=None):
    """One-shot inverse; sig defaults to the registered context's signature."""
    if sig is None:
        import clif.context as Clif                    # lazy: avoids an import cycle
        sig = Clif.get_signature()
    key = tuple(int(s) for s in sig)
    inv = _CACHE.get(key)
    if inv is None:
        inv = _CACHE[key] = LowDimInverse(len(key), key)
    return inv.inverse(np.asarray(A, dtype=np.float64))
