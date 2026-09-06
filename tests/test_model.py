# file: tests/test_model.py
"""
Certification of the reference model (clif.inverse.model_inverse).

Run from the engine root:  pytest tests/test_model.py -v

Checks, per dimension 4-11:
  * the hand-written involution tables equal the closed form
    (reversion x Thue-Morse character on the top ceil((d-6)/2) bits)
  * the tweak tables equal the FLS ladder n/k - 1, n = 2^(d//2 - 1)
  * closure: C = A*tau(A) vanishes off Fix(tau)
  * central collapse: E = D*C is scalar (even d) or scalar+pseudoscalar (odd d)
  * the inverse itself: ||A * mv_inv(A) - 1|| at machine precision
    (conditioning-limited at d = 10, 11)
"""
import numpy as np
import pytest

import clif.context as Clif
from clif.multivector import Accum
from clif.util import inverse_involution
from clif.inverse.fls import _make_tweak
from clif.inverse import model_inverse as MI

DIMS = list(range(4, 12))

# residual tolerance: machine precision through d=9, conditioning after
TOL = {4: 1e-12, 5: 1e-12, 6: 1e-11, 7: 1e-11, 8: 1e-9, 9: 1e-9,
       10: 1e-5, 11: 1e-5}


def _register(d: int) -> None:
    """Register a Euclidean context Cl(d, 0) with the engine."""
    if hasattr(Clif, "Cl"):
        Clif.Cl(d)
    else:                       # mock harness fallback
        Clif.set_dim(d)


@pytest.mark.parametrize("d", DIMS)
def test_involution_matches_closed_form(d):
    tab = np.array(MI.Involution[d - 4], dtype=np.float64)
    assert np.array_equal(tab, inverse_involution(d))


@pytest.mark.parametrize("d", DIMS)
def test_tweaks_match_fls_ladder(d):
    tw = np.array(MI.Tweak[d - 4], dtype=np.float64)
    cf = _make_tweak(d)
    assert len(tw) == len(cf)
    assert np.allclose(tw, cf, rtol=0, atol=1e-15)


@pytest.mark.parametrize("d", DIMS)
def test_closure(d):
    """C = A*tau(A) stays in Fix(tau): zero leakage off the pd domain."""
    _register(d)
    rng = np.random.default_rng(100 + d)
    tau = inverse_involution(d)
    A = Accum(rng.standard_normal(1 << d))
    C = (A * Accum(tau * A.Reg)).Reg
    leak = np.max(np.abs(C[tau < 0])) / np.max(np.abs(C))
    assert leak < 1e-13


@pytest.mark.parametrize("d", DIMS)
def test_central_collapse(d):
    """E = D*C lands on the center: scalar (even d), scalar+pseudo (odd d)."""
    _register(d)
    rng = np.random.default_rng(200 + d)
    size = 1 << d
    A = Accum(rng.standard_normal(size))

    I = np.array(MI.Involution[d - 4], dtype=np.float64)
    B = Accum(I * A.Reg)
    C = A * B
    T = MI.Tweak[d - 4]
    D = C.copy(); D.Reg[0] *= -T[0]
    if d & 1:
        D.Reg[size - 1] *= -T[0]
    for i in range(1, len(T)):
        D *= C
        D.Reg[0] *= -T[i]
        if d & 1:
            D.Reg[size - 1] *= -T[i]
    E = (D * C).Reg

    mask = np.ones(size, dtype=bool)
    mask[0] = False
    if d & 1:
        mask[size - 1] = False
    off_center = np.max(np.abs(E[mask])) / np.abs(E[0])
    assert off_center < TOL[d]


@pytest.mark.parametrize("d", DIMS)
def test_inverse_residual(d):
    _register(d)
    rng = np.random.default_rng(300 + d)
    A = Accum(rng.standard_normal(1 << d))
    rA = MI.inverse(A)
    P = (A * rA).Reg
    res = max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))))
    assert res < TOL[d]
