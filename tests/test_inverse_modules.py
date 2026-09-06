# file: tests/test_inverse_modules.py
"""
Smoke tests and failure-path tests for the carried-along inverse modules.

Section 5: euclidean, fls, newton — each verified to produce a correct inverse.
Section 8: designed failure paths — ValueError for a non-invertible multivector,
           NotImplementedError above d = 11.
"""
import numpy as np
import pytest

import clif.context as Clif
from clif.multivector import Accum
from clif.inverse.fls import fls_inverse, even_inverse, odd_inverse
from clif.inverse.euclidean import euclidean_inverse
from clif.inverse.newton import newton_schulz
from clif.inverse.low_dim_inverse import inverse as low_dim_inverse
from clif.inverse import model_inverse as MI


def _residual(A: Accum, rA: Accum) -> float:
    P = (A * rA).Reg
    return max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))))


# ---------------------------------------------------------------------------
# Section 5 smoke tests
# ---------------------------------------------------------------------------

_FLS_TOL = {6: 1e-8, 7: 1e-8, 8: 1e-8, 9: 1e-8, 10: 1e-5, 11: 1e-5}

@pytest.mark.parametrize("d", [6, 7, 8, 9, 10, 11])
def test_fls_inverse_smoke(d):
    Clif.Cl(d)
    A = Accum(np.random.default_rng(10 + d).standard_normal(1 << d))
    rA = fls_inverse(A)
    assert _residual(A, rA) < _FLS_TOL[d]


@pytest.mark.parametrize("d", [6, 7, 8, 9, 10, 11])
def test_euclidean_inverse_smoke(d):
    Clif.Cl(d)
    A = Accum(np.random.default_rng(20 + d).standard_normal(1 << d))
    rA = euclidean_inverse(A)
    assert _residual(A, rA) < 1e-8


@pytest.mark.parametrize("d", [6, 7, 8])
def test_newton_schulz_smoke(d):
    """Newton-Schulz refines an FLS result and does not regress it."""
    Clif.Cl(d)
    A = Accum(np.random.default_rng(30 + d).standard_normal(1 << d))
    X0 = fls_inverse(A)
    res_before = _residual(A, X0)
    Xr = newton_schulz(A, X0)
    res_after = _residual(A, Xr)
    assert res_after <= res_before + 1e-14   # must not make things worse
    assert res_after < 1e-12


# ---------------------------------------------------------------------------
# Section 8 failure-path tests
# ---------------------------------------------------------------------------

def test_low_dim_inverse_raises_on_non_invertible():
    """A degenerate signature produces a genuinely non-invertible multivector."""
    Clif.Cl(2, 0, 1)          # one null generator — e3^2 = 0
    # e3 alone: coefficient at ordinal 4 (0b100)
    A = np.zeros(1 << 3)
    A[4] = 1.0                # purely the null basis vector
    with pytest.raises(ValueError, match="not invertible"):
        low_dim_inverse(A)


def test_accum_inverse_raises_above_11d():
    """Accum.inverse() must raise NotImplementedError for d > 11."""
    Clif.Cl(12)
    A = Accum(np.random.default_rng(99).standard_normal(1 << 12))
    with pytest.raises(NotImplementedError):
        A.inverse()
