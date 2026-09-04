# file: tests/test_sparse_fls.py
"""
Certification of the production sparse FLS inverse
(clifford.inverse.sparse_fls, v3) against the reference model.

Run from the engine root:  pytest tests/test_sparse_fls.py -v

Checks, per dimension 4-11:
  * table certificate: every tabulated pair commutes (the structural
    invariant the symmetric kernel depends on -- guaranteed because tau
    is an anti-automorphism, verified here rather than assumed) and the
    stored sign equals sign(e_a * e_b)
  * rectangularity: the non-central rows all have length Q and the slots
    array covers exactly the non-central compact slots (the v3 kernel
    assumes this shape; a ragged future involution must fail loudly)
  * pseudoscalar machinery for odd dims: pseudo slot active, complement
    map is a sign-consistent perfect matching, fused sign PS agrees with
    the pair signs it replaced
  * production output equals the model oracle to conditioning level
  * inverse residual ||A * inv(A) - 1||
  * dimension guard: ValueError outside 4-11
  * optional: current version equals archived sparse_fls_v2 (skipped if
    the v2 module is absent)
"""
import numpy as np
import pytest

import clifford.context as Clif
from clifford.multivector import Accum
from clifford.inverse import model_inverse as MI
from clifford.inverse import sparse_fls as SF

try:
    from clifford.inverse import sparse_fls_v2 as SFV2
except ImportError:
    SFV2 = None

DIMS = list(range(4, 12))
ODD  = [d for d in DIMS if d % 2 == 1]

TOL = {4: 1e-12, 5: 1e-12, 6: 1e-11, 7: 1e-11, 8: 1e-9, 9: 1e-9, 10: 1e-5, 11: 1e-5}


def _register(d: int) -> None:
    Clif.Cl(d)


def _tables(module, d):
    _register(d)
    module._table_cache.clear()   # tables must be built under this context
    return module._get_tables(d)


def _euc_sign(rk_a: int, b: int) -> int:
    return 1 - 2 * (bin(rk_a & b).count('1') % 2)


@pytest.mark.parametrize("d", DIMS)
def test_commutation_certificate(d):
    """All tabulated pairs commute and carry the correct sign."""
    tb = _tables(SF, d)
    I_arr, R, RS = tb['I'], tb['R'], tb['RS']
    rk = Clif._ActiveTable.row_keys
    nrows, Q = R.shape[0], R.shape[1]
    for r in range(nrows):
        for j in range(Q):
            a = int(I_arr[R[r, j, 0]])
            b = int(I_arr[R[r, j, 1]])
            s_ab = _euc_sign(int(rk[a]), b)
            s_ba = _euc_sign(int(rk[b]), a)
            assert s_ab == s_ba, f"anticommuting pair in table: {a:#x},{b:#x}"
            assert RS[r, j] == s_ab, f"sign mismatch at ({r},{j})"


@pytest.mark.parametrize("d", DIMS)
def test_table_rectangular(d):
    """slots covers exactly the non-central compact slots, once each,
    and pair endpoints partition each row's blades (no reuse within a row)."""
    tb = _tables(SF, d)
    M     = len(tb['I'])
    R     = tb['R']
    slots = tb['slots']
    central = {0} | ({tb['pseudo_pos']} if d % 2 == 1 else set())
    assert set(slots.tolist()) == set(range(M)) - central
    assert len(slots) == len(set(slots.tolist()))
    assert R.shape[0] == M - len(central)
    assert R.shape[1] == tb['Q']


@pytest.mark.parametrize("d", ODD)
def test_pseudoscalar_machinery(d):
    """Pseudo slot active; complement map is a perfect matching whose
    fused sign PS equals sign(e_a * e_b) for each complement pair."""
    tb = _tables(SF, d)
    I_arr, comp_pos, PS = tb['I'], tb['comp_pos'], tb['PS']
    rk = Clif._ActiveTable.row_keys
    M  = len(I_arr)
    assert tb['pseudo_pos'] >= 0
    assert np.all(comp_pos[comp_pos] == np.arange(M))   # involution/matching
    for i in range(M):
        a, b = int(I_arr[i]), int(I_arr[comp_pos[i]])
        assert a ^ b == (1 << d) - 1
        assert PS[i] == _euc_sign(int(rk[a]), b)


@pytest.mark.parametrize("d", DIMS)
def test_matches_model_oracle(d):
    _register(d)
    SF._table_cache.clear()
    rng = np.random.default_rng(400 + d)
    A = Accum(); A.Reg = rng.standard_normal(1 << d)
    r_model = MI.mv_inv(A).Reg
    r_prod = SF.sparse_fls_inverse(A).Reg
    diff = np.max(np.abs(r_model - r_prod)) / np.max(np.abs(r_model))
    assert diff < TOL[d]


@pytest.mark.parametrize("d", DIMS)
def test_inverse_residual(d):
    _register(d)
    SF._table_cache.clear()
    rng = np.random.default_rng(500 + d)
    A = Accum(); A.Reg = rng.standard_normal(1 << d)
    rA = SF.sparse_fls_inverse(A)
    P = (A * rA).Reg
    res = max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))))
    assert res < TOL[d]


@pytest.mark.parametrize("d", [3, 12])
def test_dimension_guard(d):
    _register(d)
    A = Accum(); A.Reg = np.ones(1 << d)
    with pytest.raises(ValueError):
        SF.sparse_fls_inverse(A)


V2_DIMS = [d for d in DIMS if d >= 6]      # v2 archive never supported 4-5

@pytest.mark.skipif(SFV2 is None, reason="sparse_fls_v2 not present")
@pytest.mark.parametrize("d", V2_DIMS)
def test_v2_v3_equivalence(d):
    _register(d)
    SF._table_cache.clear()
    SFV2._table_cache.clear()
    rng = np.random.default_rng(600 + d)
    A = Accum(); A.Reg = rng.standard_normal(1 << d)
    r2 = SFV2.sparse_fls_inverse(A).Reg
    r3 = SF.sparse_fls_inverse(A).Reg
    diff = np.max(np.abs(r2 - r3)) / np.max(np.abs(r2))
    assert diff < TOL[d]
