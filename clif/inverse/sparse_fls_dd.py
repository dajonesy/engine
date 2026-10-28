#file: clif/inverse/sparse_fls_dd.py
"""Double-double (dd) tau-poly for sparse_fls v7 -- even and odd d, 4..15.

A dd number is an unevaluated pair (hi, lo) of float64 with |lo| <= ulp(hi)/2,
carrying about 106 bits of significand with the float64 exponent range.

Scope
-----
The sweep runs in dd. Step down and step up stay in float64 but are computed
by table-free kernels (signs from bit operations at run time), so nothing of
size n * width is stored: memory is O(n + tau table).

    C (float64)  --sweep-->  D (float64), E0, EPS (float64)

The sweep enters dd at its first multivector product and rounds back to
float64 on return; the composite-centre division for odd d is v7's, in
float64. Tweaks are formed in dd from the integer pair (N-1-k, k+1), so
values such as 1/3 carry no float64 rounding into the ladder.

RANGE GUARD: the ladder forms powers of C up to C**N, and the odd centre
squares E0; at d=15 that overflows the float64 exponent range (dd has the
same range). The ladder is linear in D, and the centre division is
homogeneous in (D, E0, EPS), so each sweep rescales D by an exact power of
two after every product (_renorm). The inverse is unchanged and no rounding
is added.

Layouts (as in v7)
------------------
    even d            EvenTable(d)
    odd d, fast path  EvenTable(d-1), Z = u + w I, four even products
    odd d, native     OddTable(d), when sig[:d-1] has a null generator

sparse_fls.py (v7) is not modified. DDTauInverse re-implements
TauInverse.__init__ without its d <= 11 limit, builds the tau tables in row
chunks with v7's own _pair_rows (identical arrays, bounded peak memory), and
overrides sweep, bookend1 (step down) and bookend2 (step up). Everything
else -- inverse(), mul(), the odd centre -- is inherited.
"""

# The dd row kernels and the step down / step up always run thread-parallel:
# dd is used from d = 12 (and measured at d = 10, 11), where every product
# has hundreds to thousands of rows, and parallel execution was 2.6-4.1x
# faster at every d measured. Set False only to reproduce serial timings.
PARALLEL = True

import math
import numpy as np
from numba import njit, prange

try:
    from .sparse_fls import (TauInverse, EvenTable, OddTable, tau_signs,
                             tau_slots, blade_sign, blade_signs, _pair_rows)
except ImportError:                                    # flat layout for tests
    from sparse_fls import (TauInverse, EvenTable, OddTable, tau_signs,
                            tau_slots, blade_sign, blade_signs, _pair_rows)

# ------------------------------------------------- error-free transforms ---

@njit(inline='always')
def two_sum(a, b):                                     # a + b = s + e exactly
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)

@njit(inline='always')
def quick_two_sum(a, b):                               # requires |a| >= |b|
    s = a + b
    return s, b - (s - a)

@njit(inline='always')
def split(a):                                          # Dekker, 2**27 + 1
    t = 134217729.0 * a
    hi = t - (t - a)
    return hi, a - hi

@njit(inline='always')
def two_prod(a, b):                                    # a * b = p + e exactly
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl

# ------------------------------------------------------- dd arithmetic ---

@njit(inline='always')
def dd_add(ah, al, bh, bl):
    """Accurate dd sum (both words two_summed): safe under cancellation."""
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e += t
    s, e = quick_two_sum(s, e)
    e += f
    return quick_two_sum(s, e)

@njit(inline='always')
def dd_mul(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e += ah * bl + al * bh
    return quick_two_sum(p, e)

@njit(inline='always')
def dd_div(ah, al, bh, bl):
    q1 = ah / bh
    ph, pl = dd_mul(q1, 0.0, bh, bl)
    rh, rl = dd_add(ah, al, -ph, -pl)
    q2 = rh / bh
    ph, pl = dd_mul(q2, 0.0, bh, bl)
    rh, rl = dd_add(rh, rl, -ph, -pl)
    q3 = rh / bh
    q1, q2 = quick_two_sum(q1, q2)
    return dd_add(q1, q2, q3, 0.0)

# ------------------------------------------------------------ dd kernels ---

@njit(cache=True)
def dd_mul_even_scalar(uh, ul, vh, vl, S0):
    """Scalar component of u v only -- the diagonal row. dd in, dd out."""
    ah = 0.0
    al = 0.0
    for k in range(S0.shape[0]):
        s = S0[k]
        if s == 0:
            continue
        ph, pl = dd_mul(uh[k], ul[k], vh[k], vl[k])
        if s < 0:
            ph, pl = -ph, -pl
        ah, al = dd_add(ah, al, ph, pl)
    return ah, al


@njit(cache=True,parallel=PARALLEL)
def dd_mul_even(uh, ul, vh, vl, S0, TA, TB, S, outh, outl):
    """mul_even with u, v, out in dd. Same tables, same row order."""
    outh[0], outl[0] = dd_mul_even_scalar(uh, ul, vh, vl, S0)
    for r in prange(TA.shape[0]):
        ah = 0.0
        al = 0.0
        for k in range(TA.shape[1]):
            s = S[r, k]
            if s == 0:                                 # degenerate term
                continue
            a = TA[r, k]
            b = TB[r, k]
            ph, pl = dd_mul(uh[a], ul[a], vh[b], vl[b])
            qh, ql = dd_mul(uh[b], ul[b], vh[a], vl[a])
            ph, pl = dd_add(ph, pl, qh, ql)            # one sign, two terms
            if s < 0:
                ph, pl = -ph, -pl
            ah, al = dd_add(ah, al, ph, pl)
        outh[r + 1] = ah
        outl[r + 1] = al



@njit(inline='always')
def dd_tweak(N, k):
    """Tweak k = (N-1-k)/(k+1) in dd, negated: the scalar is scaled by -T_k."""
    th, tl = dd_div(float(N - 1 - k), 0.0, float(k + 1), 0.0)
    return -th, -tl

# ------------------------------------------------- dd kernels, odd d ---

@njit(cache=True)
def dd_mul_odd(Xh, Xl, Yh, Yl, Se, I2, S0, TA, TB, S, outh, outl,
               t1h, t1l, t2h, t2l, t3h, t3l):
    """(u1 + w1 I)(u2 + w2 I) in dd using the even table. Four even products.

    out must not alias X or Y.
    """
    u1h, u1l, w1h, w1l = Xh[:Se], Xl[:Se], Xh[Se:], Xl[Se:]
    u2h, u2l, w2h, w2l = Yh[:Se], Yl[:Se], Yh[Se:], Yl[Se:]
    dd_mul_even(u1h, u1l, u2h, u2l, S0, TA, TB, S, t1h, t1l)       # u1 u2
    dd_mul_even(w1h, w1l, w2h, w2l, S0, TA, TB, S, t2h, t2l)       # w1 w2
    for i in range(Se):
        outh[i], outl[i] = dd_add(t1h[i], t1l[i], I2 * t2h[i], I2 * t2l[i])
    dd_mul_even(u1h, u1l, w2h, w2l, S0, TA, TB, S, t1h, t1l)       # u1 w2
    dd_mul_even(w1h, w1l, u2h, u2l, S0, TA, TB, S, t3h, t3l)       # w1 u2
    for i in range(Se):
        outh[Se + i], outl[Se + i] = dd_add(t1h[i], t1l[i], t3h[i], t3l[i])


@njit(cache=True)
def dd_centre_odd(Xh, Xl, Yh, Yl, Se, I2, S0):
    """Scalar and pseudoscalar of an odd product, fast layout, in dd."""
    ah, al = dd_mul_even_scalar(Xh[:Se], Xl[:Se], Yh[:Se], Yl[:Se], S0)
    bh, bl = dd_mul_even_scalar(Xh[Se:], Xl[Se:], Yh[Se:], Yl[Se:], S0)
    e0h, e0l = dd_add(ah, al, I2 * bh, I2 * bl)
    ah, al = dd_mul_even_scalar(Xh[:Se], Xl[:Se], Yh[Se:], Yl[Se:], S0)
    bh, bl = dd_mul_even_scalar(Xh[Se:], Xl[Se:], Yh[:Se], Yl[:Se], S0)
    eph, epl = dd_add(ah, al, bh, bl)
    return e0h, e0l, eph, epl


@njit(cache=True,parallel=PARALLEL)
def dd_mul_oddn(Xh, Xl, Yh, Yl, S0, PSG, comp, pp, RO, TA, TB, S, outh, outl):
    """mul_oddn (native odd layout) in dd."""
    ah = 0.0
    al = 0.0
    for k in range(S0.shape[0]):                       # scalar: diagonal
        s = S0[k]
        if s == 0:
            continue
        ph, pl = dd_mul(Xh[k], Xl[k], Yh[k], Yl[k])
        ah, al = dd_add(ah, al, s * ph, s * pl)
    outh[0], outl[0] = ah, al
    ah = 0.0
    al = 0.0
    for k in range(PSG.shape[0]):                      # pseudoscalar: complement
        j = comp[k]
        ph, pl = dd_mul(Xh[k], Xl[k], Yh[j], Yl[j])
        s = PSG[k]
        ah, al = dd_add(ah, al, s * ph, s * pl)
    outh[pp], outl[pp] = ah, al
    for r in prange(TA.shape[0]):
        ah = 0.0
        al = 0.0
        for k in range(TA.shape[1]):
            s = S[r, k]
            if s == 0:
                continue
            a = TA[r, k]
            b = TB[r, k]
            ph, pl = dd_mul(Xh[a], Xl[a], Yh[b], Yl[b])
            qh, ql = dd_mul(Xh[b], Xl[b], Yh[a], Yl[a])
            ph, pl = dd_add(ph, pl, qh, ql)
            if s < 0:
                ph, pl = -ph, -pl
            ah, al = dd_add(ah, al, ph, pl)
        outh[RO[r]], outl[RO[r]] = ah, al


@njit(cache=True)
def dd_centre_oddn(Xh, Xl, Yh, Yl, S0, PSG, comp):
    """Scalar and pseudoscalar of a native odd product, in dd."""
    e0h = 0.0
    e0l = 0.0
    eph = 0.0
    epl = 0.0
    for k in range(S0.shape[0]):
        s = S0[k]
        if s != 0:
            ph, pl = dd_mul(Xh[k], Xl[k], Yh[k], Yl[k])
            e0h, e0l = dd_add(e0h, e0l, s * ph, s * pl)
        j = comp[k]
        ph, pl = dd_mul(Xh[k], Xl[k], Yh[j], Yl[j])
        s = PSG[k]
        eph, epl = dd_add(eph, epl, s * ph, s * pl)
    return e0h, e0l, eph, epl

# ------------------------------------------------------------ dd sweeps ---
# Each: C float64 in; (D, E0, EPS) float64 out. P is the pseudoscalar
# tau-index (unused at even d).

@njit(inline='always')
def _renorm(Dh, Dl):
    """Scale D by the power of two that brings max|D| into [0.5, 1): exact."""
    m = 0.0
    for i in range(Dh.shape[0]):
        v = abs(Dh[i])
        if v > m:
            m = v
    if m > 0.0:
        f = math.ldexp(1.0, -math.frexp(m)[1])
        for i in range(Dh.shape[0]):
            Dh[i] *= f
            Dl[i] *= f


@njit(inline='always')
def _round(Dh, Dl):
    D = np.empty(Dh.shape[0])
    for i in range(Dh.shape[0]):
        D[i] = Dh[i] + Dl[i]                           # step back to float64
    return D


@njit(cache=True)
def dd_sweep_even(C, N, S0, TA, TB, S):
    w = C.shape[0]
    Ch, Cl = C.copy(), np.zeros(w)
    Dh, Dl = C.copy(), np.zeros(w)
    Wh, Wl = np.zeros(w), np.zeros(w)
    th, tl = dd_tweak(N, 0)
    Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
    for k in range(1, N - 1):
        dd_mul_even(Dh, Dl, Ch, Cl, S0, TA, TB, S, Wh, Wl)   # first pass: dd
        Dh, Wh = Wh, Dh
        Dl, Wl = Wl, Dl
        _renorm(Dh, Dl)
        th, tl = dd_tweak(N, k)
        Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
    eh, el = dd_mul_even_scalar(Dh, Dl, Ch, Cl, S0)
    return _round(Dh, Dl), eh + el, 0.0


@njit(cache=True)
def dd_sweep_odd(C, N, Se, I2, S0, TA, TB, S):
    w = C.shape[0]
    P = Se
    Ch, Cl = C.copy(), np.zeros(w)
    Dh, Dl = C.copy(), np.zeros(w)
    Wh, Wl = np.zeros(w), np.zeros(w)
    t1h, t1l = np.zeros(Se), np.zeros(Se)
    t2h, t2l = np.zeros(Se), np.zeros(Se)
    t3h, t3l = np.zeros(Se), np.zeros(Se)
    th, tl = dd_tweak(N, 0)
    Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
    Dh[P], Dl[P] = dd_mul(Dh[P], Dl[P], th, tl)
    for k in range(1, N - 1):
        dd_mul_odd(Dh, Dl, Ch, Cl, Se, I2, S0, TA, TB, S, Wh, Wl,
                   t1h, t1l, t2h, t2l, t3h, t3l)
        Dh, Wh = Wh, Dh
        Dl, Wl = Wl, Dl
        _renorm(Dh, Dl)
        th, tl = dd_tweak(N, k)
        Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
        Dh[P], Dl[P] = dd_mul(Dh[P], Dl[P], th, tl)
    e0h, e0l, eph, epl = dd_centre_odd(Dh, Dl, Ch, Cl, Se, I2, S0)
    return _round(Dh, Dl), e0h + e0l, eph + epl


@njit(cache=True)
def dd_sweep_native(C, N, S0, PSG, comp, pp, RO, TA, TB, S):
    w = C.shape[0]
    P = pp
    Ch, Cl = C.copy(), np.zeros(w)
    Dh, Dl = C.copy(), np.zeros(w)
    Wh, Wl = np.zeros(w), np.zeros(w)
    th, tl = dd_tweak(N, 0)
    Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
    Dh[P], Dl[P] = dd_mul(Dh[P], Dl[P], th, tl)
    for k in range(1, N - 1):
        dd_mul_oddn(Dh, Dl, Ch, Cl, S0, PSG, comp, pp, RO, TA, TB, S, Wh, Wl)
        Dh, Wh = Wh, Dh
        Dl, Wl = Wl, Dl
        _renorm(Dh, Dl)
        th, tl = dd_tweak(N, k)
        Dh[0], Dl[0] = dd_mul(Dh[0], Dl[0], th, tl)
        Dh[P], Dl[P] = dd_mul(Dh[P], Dl[P], th, tl)
    e0h, e0l, eph, epl = dd_centre_oddn(Dh, Dl, Ch, Cl, S0, PSG, comp)
    return _round(Dh, Dl), e0h + e0l, eph + epl

# ------------------------------------------- table-free step down / up ---
@njit(inline='always')
def parity(x):
    x ^= x >> 32
    x ^= x >> 16
    x ^= x >> 8
    x ^= x >> 4
    x ^= x >> 2
    x ^= x >> 1
    return x & 1


@njit(inline='always')
def sign_of(a, b, neg, nul):
    """blade_sign(a, b, sig) from bit operations: 0 or +/-1 (float64).

    Swap parity: bit i of t is the XOR of the bits of a above i, so the
    swap count's parity is parity(b & t). Metric: parity(a & b & neg).
    """
    if a & b & nul:
        return 0.0
    t = a >> 1
    t ^= t >> 1
    t ^= t >> 2
    t ^= t >> 4
    t ^= t >> 8
    t ^= t >> 16
    return -1.0 if parity((b & t) ^ (a & b & neg)) else 1.0



@njit(cache=True,parallel=PARALLEL)
def step_down_kernel(A, tau, slots, sg, neg, nul, C):
    """C = A tau(A) on tau-domain outputs. Gather by output; no term list.

    slots[i] is the index stored at tau-index i; sg[i] relates the stored
    value to the full coefficient (pssign on the fast odd path, else 1).
    Output c != 0: pairs (a, a^c) with a < a^c, i.e. a has the top bit h of
    c clear; enumerate them by inserting a 0 at bit h. Each pair counts
    twice (C is tau-symmetric). n/2 multiplies per output.
    """
    n = A.shape[0]
    acc0 = 0.0
    for a in range(n):                                 # scalar: diagonal
        acc0 += sign_of(a, a, neg, nul) * tau[a] * A[a] * A[a]
    C[0] = acc0
    half = n >> 1
    for i in prange(1, slots.shape[0]):
        c = slots[i]
        h = 0
        while (c >> (h + 1)) != 0:
            h += 1
        lo = (1 << h) - 1
        acc = 0.0
        for j in range(half):
            a = ((j >> h) << (h + 1)) | (j & lo)
            b = a ^ c
            acc += sign_of(a, b, neg, nul) * tau[b] * A[a] * A[b]
        C[i] = 2.0 * sg[i] * acc


@njit(cache=True,parallel=PARALLEL)
def step_up_kernel(A, rC, tau, slots, sg, neg, nul, R):
    """R = tau(A) rC over all outputs. Gather by output o: a = o ^ c."""
    n = A.shape[0]
    for o in prange(n):
        acc = 0.0
        for i in range(slots.shape[0]):
            c = slots[i]
            a = o ^ c
            acc += sign_of(a, c, neg, nul) * tau[a] * A[a] * (sg[i] * rC[i])
        R[o] = acc

# --------------------------------------------------- chunked tau tables ---

def build_even_table(d, sig, chunk=256):
    """EvenTable built in row chunks via v7's _pair_rows: same arrays."""
    T = EvenTable.__new__(EvenTable)
    T.d, T.sig = d, list(sig)
    T.slots = tau_slots(d)
    n = len(T.slots)
    sl = np.asarray(T.slots, dtype=np.int64)
    T.S0 = blade_signs(sl, sl, sig)
    T.TA, T.TB, T.S, T.pairs = _chunked_rows(T.slots, range(1, n), sig, chunk)
    return T


def build_odd_table(d, sig, chunk=256):
    """OddTable built in row chunks via v7's _pair_rows: same arrays."""
    T = OddTable.__new__(OddTable)
    T.d, T.sig = d, list(sig)
    T.slots = tau_slots(d)
    n = len(T.slots)
    PS = (1 << d) - 1
    assert PS in T.slots, "odd tau-domain must contain the pseudoscalar"
    T.pp = T.slots.index(PS)
    sl = np.asarray(T.slots, dtype=np.int64)
    posf = np.full(1 << d, -1, dtype=np.int64)
    posf[sl] = np.arange(n)
    T.S0 = blade_signs(sl, sl, sig)
    T.comp = posf[sl ^ PS].astype(np.int32)
    assert np.all(T.comp >= 0), "odd tau-domain not closed under complement"
    T.PSG = blade_signs(sl, sl ^ PS, sig)
    T.IS = blade_signs(sl, PS, sig)
    assert np.all(T.PSG != 0), "complement pairs are disjoint; PSG never 0"
    ro = [i for i in range(1, n) if i != T.pp]
    T.RO = np.array(ro, dtype=np.int32)
    T.TA, T.TB, T.S, T.pairs = _chunked_rows(T.slots, ro, sig, chunk)
    return T


def _chunked_rows(slots, rows, sig, chunk):
    rows = list(rows)
    TA, TB, S, P = [], [], [], set()
    for r0 in range(0, len(rows), chunk):
        ta, tb, s, p = _pair_rows(slots, rows[r0:r0 + chunk], sig)
        TA.append(ta); TB.append(tb); S.append(s); P.add(p)
    assert len(P) == 1, "not rectangular"
    return np.concatenate(TA), np.concatenate(TB), np.concatenate(S), P.pop()

# ------------------------------------------------------------- inverter ---

class DDTauInverse(TauInverse):
    """TauInverse with the dd sweep and table-free step down / step up.

    4 <= d <= 15, even and odd, same layouts and signature rules as v7.
    """

    def __init__(self, d, sig, native=None):
        assert 4 <= d <= 15, "supported range is d = 4..15"
        assert len(sig) == d
        sig = list(sig)
        assert all(s in (-1, 0, 1) for s in sig), "sig entries must be -1, 0, 1"
        self.d, self.sig, self.odd = d, sig, bool(d & 1)
        self.n = 1 << d
        self.par = PARALLEL           # float64 products inherited from v8 (mul, polish)
        self.tau = tau_signs(d).astype(np.float64)
        self.N = 1 << (d // 2 - 1)
        self.tweak = [(self.N - 1 - k) / (k + 1) for k in range(self.N - 1)]
        PS = self.n - 1
        if native is None:
            native = self.odd and any(s == 0 for s in sig[:d - 1])
        assert not (native and not self.odd), "native layout is odd-d only"
        self.native = bool(native)

        if self.native:
            self.table = build_odd_table(d, sig)
            self.slots = self.table.slots
            self.width = len(self.slots)
            self.Se = self.width // 2
            self.pidx = self.table.pp
            self.I2 = float(blade_sign(PS, PS, sig))
            sg = np.ones(self.width)
        else:
            de = d - 1 if self.odd else d
            # SIGNATURE CARE: odd d builds its table from the LOW generators.
            self.table = build_even_table(de, sig[:de])
            self.Se = len(self.table.slots)
            self.pidx = self.Se
            if self.odd:
                self.I2 = float(blade_sign(PS, PS, sig))
                self.pssign = blade_signs(
                    np.asarray(self.table.slots, dtype=np.int64), PS,
                    sig).astype(np.float64)
                self.slots = self.table.slots + [m ^ PS for m in self.table.slots]
                self.width = 2 * self.Se
                assert set(self.slots) == set(tau_slots(d)), \
                    "odd tau-domain is not the even domain plus its complement"
                sg = np.concatenate([np.ones(self.Se), self.pssign])
            else:
                self.I2 = 0.0
                self.slots = self.table.slots
                self.width = self.Se
                sg = np.ones(self.width)
        self.pos = {m: i for i, m in enumerate(self.slots)}
        self._slots = np.asarray(self.slots, dtype=np.int64)
        self._sg = sg
        self._neg = sum(1 << i for i, s in enumerate(sig) if s < 0)
        self._nul = sum(1 << i for i, s in enumerate(sig) if s == 0)
        self._scratch = [np.zeros(self.width) for _ in range(3)]
        self._t = [np.zeros(self.Se) for _ in range(3)]
        self._C = np.zeros(self.width)
        self._R = np.zeros(self.n)

    def sweep(self, C):
        """Same contract as TauInverse.sweep: returns (D, E0, EPS)."""
        T = self.table
        if self.native:
            return dd_sweep_native(C, self.N, T.S0, T.PSG, T.comp, T.pp, T.RO,
                                   T.TA, T.TB, T.S)
        if self.odd:
            return dd_sweep_odd(C, self.N, self.Se, self.I2, T.S0, T.TA, T.TB,
                                T.S)
        return dd_sweep_even(C, self.N, T.S0, T.TA, T.TB, T.S)

    def bookend1(self, A):
        """Step down: C = A tau(A) on the tau-domain."""
        step_down_kernel(A, self.tau, self._slots, self._sg, self._neg,
                         self._nul, self._C)
        return self._C

    def bookend2(self, A, rC):
        """Step up: A^-1 = tau(A) rC over all 2**d outputs."""
        step_up_kernel(A, rC, self.tau, self._slots, self._sg, self._neg,
                       self._nul, self._R)
        return self._R

    step_down, step_up = bookend1, bookend2

    def cost(self):
        """Real multiplication counts; step down/up from the kernels' loops."""
        M = self.table.real_mults()
        if self.native:
            per_mul, tail = M, 2 * self.width
        else:
            per_mul = 4 * M if self.odd else M
            tail = 4 * self.Se if self.odd else len(self.table.slots)
        sweep = (self.N - 2) * per_mul + tail
        down = self.n + (self.width - 1) * (self.n // 2)
        up = self.n * self.width
        return dict(table_bytes=self.table.nbytes(), mults_per_mul=per_mul,
                    sweep=sweep, step_down=down, step_up=up,
                    total=sweep + down + up)
