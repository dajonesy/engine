# file: tests/test_sparse_fls.py
"""
Certification of the production sparse FLS inverse
(clif.inverse.sparse_fls / TauInverse) against the reference model.

Run from the engine root:  pytest tests/test_sparse_fls.py -v

Checks, per dimension 4-11:
  * structural: TauInverse builds without error; all tabulated pair signs
    match blade_sign(a, b, sig) (the same invariant the old commutation
    certificate verified, now read from EvenTable.TA/TB/S directly)
  * rectangularity: TA/TB/S have the expected shape
  * pseudoscalar machinery for odd dims: I2, pssign, slot complement closure
  * production output equals the model oracle to conditioning level
  * inverse residual ||A * inv(A) - 1||
  * dimension guard: AssertionError outside 4-11 (TauInverse range)
  * optional: current version equals archived sparse_fls_v2 (skipped if
    the v2 module is absent)
"""
import numpy as np
import pytest

import clif.context as Clif
from clif.multivector import Accum
from clif.inverse import model_inverse as MI
from clif.inverse.sparse_fls import (
    TauInverse, tau_slots, blade_sign, EvenTable
)
from clif.inverse.tau_api import get_inverter

try:
    from clif.inverse import sparse_fls_v2 as SFV2
except ImportError:
    SFV2 = None

DIMS = list(range(4, 12))
ODD  = [d for d in DIMS if d % 2 == 1]

TOL = {4: 1e-12, 5: 1e-12, 6: 1e-11, 7: 1e-11, 8: 1e-9, 9: 1e-9, 10: 1e-5, 11: 1e-5}

_EUC = [1] * 11   # signature list for Euclidean algebras, indexed by d


def _register(d: int) -> None:
    Clif.Cl(d)


def _inverter(d: int) -> TauInverse:
    sig = _EUC[:d]
    return TauInverse(d, sig)


# ---------------------------------------------------------------------------
# Structural: pair signs match blade_sign
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("d", DIMS)
def test_commutation_certificate(d):
    """All tabulated pairs carry the correct sign: S[r,j] == blade_sign(a,b,sig)."""
    sig = _EUC[:d]
    ti = _inverter(d)
    de = d - 1 if ti.odd else d
    et = ti.table                    # EvenTable for even dimension de
    slots = et.slots
    for r in range(et.TA.shape[0]):
        for j in range(et.pairs):
            a = slots[et.TA[r, j]]
            b = slots[et.TB[r, j]]
            s_ab = blade_sign(a, b, sig[:de])
            s_ba = blade_sign(b, a, sig[:de])
            assert s_ab == s_ba, f"anticommuting pair at d={d} ({a:#x},{b:#x})"
            assert et.S[r, j] == s_ab, f"sign mismatch at d={d} row {r} pair {j}"


@pytest.mark.parametrize("d", DIMS)
def test_table_rectangular(d):
    """TA/TB/S are rectangular: all rows have the same pair count."""
    ti = _inverter(d)
    et = ti.table
    n = len(et.slots)
    assert et.TA.shape == (n - 1, et.pairs)
    assert et.TB.shape == (n - 1, et.pairs)
    assert et.S.shape  == (n - 1, et.pairs)


@pytest.mark.parametrize("d", ODD)
def test_pseudoscalar_machinery(d):
    """Odd dims: slot set = even slots + complements; I2 and pssign are consistent."""
    sig = _EUC[:d]
    ti = _inverter(d)
    PS = (1 << d) - 1

    # slot set must equal tau_slots(d)
    assert set(ti.slots) == set(tau_slots(d))

    # complement closure: every slot m has m ^ PS also in slots
    slot_set = set(ti.slots)
    for m in ti.slots:
        assert (m ^ PS) in slot_set, f"d={d}: slot {m:#x} missing complement"

    # I2 = sign of e_PS^2 in this signature
    assert ti.I2 == blade_sign(PS, PS, sig)

    # pssign[i] = sign of e_{slots[i]} * e_PS in the even sub-algebra
    even_slots = ti.table.slots
    for i, m in enumerate(even_slots):
        assert ti.pssign[i] == blade_sign(m, PS, sig)


@pytest.mark.parametrize("d", DIMS)
def test_matches_model_oracle(d):
    _register(d)
    rng = np.random.default_rng(400 + d)
    A = Accum(rng.standard_normal(1 << d))
    r_model = MI.inverse(A).Reg
    r_prod = get_inverter(Clif.get_signature()).inverse(A.Reg)
    diff = np.max(np.abs(r_model - r_prod)) / np.max(np.abs(r_model))
    assert diff < TOL[d]


@pytest.mark.parametrize("d", DIMS)
def test_inverse_residual(d):
    _register(d)
    rng = np.random.default_rng(500 + d)
    A = Accum(rng.standard_normal(1 << d))
    rA_reg = get_inverter(Clif.get_signature()).inverse(A.Reg)
    rA = Accum(rA_reg)
    P = (A * rA).Reg
    res = max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))))
    assert res < TOL[d]


@pytest.mark.parametrize("d", [3, 12])
def test_dimension_guard(d):
    """TauInverse raises AssertionError for d outside 4-11."""
    sig = _EUC[:d]
    with pytest.raises(AssertionError):
        TauInverse(d, sig)


V2_DIMS = [d for d in DIMS if d >= 6]

@pytest.mark.skipif(SFV2 is None, reason="sparse_fls_v2 not present")
@pytest.mark.parametrize("d", V2_DIMS)
def test_v2_v3_equivalence(d):
    _register(d)
    sig = Clif.get_signature()
    rng = np.random.default_rng(600 + d)
    A = Accum(rng.standard_normal(1 << d))
    r2 = SFV2.TauInverse(d, sig).inverse(A.Reg)
    r3 = get_inverter(sig).inverse(A.Reg)
    diff = np.max(np.abs(r2 - r3)) / np.max(np.abs(r2))
    assert diff < TOL[d]
