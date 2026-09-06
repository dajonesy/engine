# file: tests/test_roundtrip.py
"""
Independent roundtrip verification of the tau-path inverse.

_full_mul is an independently-written geometric product that shares no code
with the engine's sign-table multiplier.  Using it to verify A * A^-1 = 1
gives confidence that the inverse is correct, not just self-consistent.
"""
import numpy as np
import pytest

from clif.inverse.tau_api import inverse


def _full_mul(A, B, sig):
    """Naive geometric product over ordinal-indexed coefficient arrays."""
    d = len(sig)
    n = 1 << d
    C = np.zeros(n)
    idx = np.arange(n)
    for a in range(n):
        if A[a] == 0.0:
            continue
        s = np.ones(n)
        sw = np.zeros(n, dtype=np.int64)
        for i in range(d):
            sw += ((idx >> i) & 1) * bin(a >> (i + 1)).count("1")
            if sig[i] != 1 and (a >> i) & 1:
                s *= np.where((idx >> i) & 1, sig[i], 1.0)
        s *= np.where(sw & 1, -1.0, 1.0)
        C[a ^ idx] += s * A[a] * B
    return C


@pytest.mark.parametrize("sig", [
    [1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, -1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
])
def test_roundtrip(sig):
    """A * A^-1 = 1 to near machine precision, verified with independent multiplier."""
    d = len(sig)
    A = np.random.default_rng(42).standard_normal(1 << d)
    rA = inverse(A, sig)
    ident = _full_mul(A, rA, sig)
    err = max(abs(ident[0] - 1.0), float(np.abs(ident[1:]).max()))
    assert err < 1e-10, f"Cl({sig.count(1)},{sig.count(-1)}) d={d}: residual {err:.2e}"
