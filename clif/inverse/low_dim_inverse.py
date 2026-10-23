# file: clif/inverse/low_dim_inverse.py
"""
Closed-form inverses for dimensions <= 3 (Hitzer & Sangwine).

3 < dimensions < 12 are handled by tau_api.py
so these processes are here to complete support for low dimensions 


Operates on ordinal coefficient arrays, matching the calling convention
of the tau path so dispatch can bind either one interchangeably.
"""

import numpy as np
import clif.context as Clif

SMALL = 1e-12


def _conjugate(A):
    """Clifford conjugation: negate grades 1 and 2 mod 4."""
    g = Clif.Grade & 3
    return np.where((g == 1) | (g == 2), -A, A)


def _reverse(A):
    """Reversion: negate grades 2 and 3 mod 4."""
    return np.where((Clif.Grade & 3) > 1, -A, A)


def inverse(A):
    """Return the inverse of ordinal coefficient array A, for d <= 3.

    Raises
    ------
    ValueError
        If the multivector is not invertible (vanishing denominator).
    """
    d = Clif.dimensions
    mul = Clif._ActiveTable.fast_mul
    scale = float(np.dot(A, A))

    if d == 0:
        den = A[0]
        deg = 1
        num = np.ones(1)
    elif d < 3:
        num = _conjugate(A)          # d=1: grade involution coincides
        H = mul(A, num)
        den, deg = H[0], 2
    else:
        C = _conjugate(A)
        H = mul(A, C)                # Study number: scalar + pseudoscalar
        Hr = _reverse(H)
        num = mul(C, Hr)
        den, deg = mul(H, Hr)[0], 4

    if abs(den) < SMALL * max(scale ** (deg / 2), 1.0):
        raise ValueError(
            f"multivector is not invertible in {Clif.get_signature()}"
        )
    return num / den