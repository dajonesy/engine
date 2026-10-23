#file: clif/inverse/sparse_fls.py   (v7)
""" tau-domain FLS inverse on even tables, with native odd tables for
degenerate odd signatures.

v7 changes
----------
  * Table and bookend construction vectorised via blade_signs() (numpy over
    whole rows); arrays are identical to v6's, term order included, so
    results are bit-for-bit unchanged. Build time at d=11 drops from ~8 s.
    blade_sign() is kept as the scalar reference and porting target.

v6 changes
----------
  * Degenerate signatures Cl(p, q, r) are supported by the inverse.
  * Odd d keeps the shared even table (u + w I) as a FAST PATH only when
    sig[:d-1] has no null generator. Otherwise it builds an OddTable over
    the full odd tau-domain (the v3 layout: scalar row by S0, pseudoscalar
    row by complement matching with a directly computed sign PSG, the rest
    rectangular). See "Why the fast path fails under degeneracy" below.
  * A vanishing denominator raises ZeroDivisionError instead of returning
    inf/nan.

Usage
-----
    inv = TauInverse(d, sig)
        Builds the tables for this dimension and signature; do it once and
        reuse `inv`. This is an order 4**d algorithm. `inv` holds
        preallocated scratch and is NOT thread-safe.

    B = inv.inverse(A)
        A and B are full multivectors of length 2**d, indexed by index.
        Optional: inverse(A, polish_steps=k) -- see the caveat under polish().

    inv.mul(X, Y, out)
        Lower-level tau-domain product. X, Y, out have length inv.width and
        are tau-multivectors, NOT full multivectors. See the closure caveat
        below before using it.

Vocabulary
----------
    index           position in the full 2**d array. An index encodes a
                    blade: bit i set means generator e_i is present, so
                    0b0101 is e_1 e_3. Only blade_sign() and the involution
                    templates test those bits; nothing else does.
    tau-involution  the sign template tau[], one entry per index
      template
    tau-domain      the indexes n with tau[n] > 0
    tau-element     a blade in the tau-domain
    tau-index       position in the compressed array; also called a slot.
      (= slot)      `slots` maps tau-index -> index.

Architecture
------------
Tables are built for EVEN dimensions (4, 6, 8, 10), plus an OddTable only for
odd signatures with a null generator in sig[:d-1] (see v6 changes). On the
fast path an odd dimension d reuses the table of d-1 by writing every
tau-element as u + w*I with u, w in the even (d-1) domain. I is the
pseudoscalar for dimension d:

    (u1 + w1 I)(u2 + w2 I) = (u1 u2 + I^2 w1 w2) + (u1 w2 + w1 u2) I

Odd elements are stored as one array Z of length 2*Se, where Se is the number
of even slots: Z[:Se] = u, Z[Se:] = w. The scalar component is then Z[0] and
the pseudoscalar component is Z[Se] -- the latter because slots[0] == 0, so
the complement half opens at 0 ^ PS == PS. The odd composite-centre step is
therefore a scaling of two entries: no pseudo_pos lookup, no parity keying.

CLOSURE CAVEAT: the tau-domain is NOT closed under multiplication. For two
arbitrary tau-elements the true product has components outside the domain,
and mul() silently discards them. The algorithm is safe because it only ever
multiplies by C = A tau(A) and its own powers, which commute and remain
tau-symmetric. Do not use mul() as a general product.

Tables
------
    slots       index of each tau-slot, ascending; slots[0] == 0
    pairs       number of (a, b) pairs per non-scalar row, a != b
    S0[i]       sign of the square of slots[i]        (len(slots),)
    TA, TB      tau-indexes of the two factors        (len(slots)-1, pairs)
    S           sign shared by (a, b) and (b, a)      (len(slots)-1, pairs)

Row r of TA/TB/S produces output tau-index r+1. The scalar output is separate
because a ^ b == 0 forces b == a, so every scalar-producing pair is a
diagonal square. S0 holds the signs for these products. Conversely rows 1.. have
c != 0, hence a != b throughout, so each pair has a distinct partner (b, a)
carrying the SAME sign; only one of the two is stored and the kernel folds
them as S * (u[a] v[b] + u[b] v[a]). Real multiplications per product:

    len(slots) + 2 * pairs * (len(slots) - 1)

The tables are organised by OUTPUT ROW so the kernel accumulates in a
register and stores once per row. A dense slots x slots form was measured
and is about 2x slower even at identical multiply count, because it must
scatter-accumulate into out[] instead.

Signature discipline (the sharp edge of this design)
----------------------------------------------------
  * sig[] is an array of signatures for each dimension {-1,0,1}
  * For odd d the shared table is built from sig[:d-1] ONLY. The top
    generator never touches it.
  * sig[d-1] enters in exactly two free places:
        I2      = blade_sign(PS, PS, sig)    one global +/-1 (or 0)
        pssign  = blade_sign(m, PS, sig)     per-slot +/-1, a free permutation
  * Mixed signatures change table signs, never table shape.
  * Degenerate metrics: zero-signed terms are kept in place with sign 0 and
    skipped at runtime, so rows stay rectangular. Even d needs nothing more.

Why the fast path fails under degeneracy (odd d)
------------------------------------------------
Complement closure -- the odd domain is the even domain plus its
complements -- is a SHAPE property and holds in every metric. The fast path
realises it MULTIPLICATIVELY, storing A[m ^ PS] = pssign[m] * w[m] with
pssign[m] = sign(e_m I). When slot m contains a null generator, e_m I = 0,
pssign[m] = 0, and that complement coefficient is written as zero by
bookend1 before the sweep begins. A single null placed LAST never enters
sig[:d-1], so the fast path still holds there; any other degenerate odd
signature loses coefficients (e.g. 240 of 496 at sig = [1]*9 + [0]*2).
The OddTable stores the complement coefficients directly and signs each
complement pair by the pure swap sign, which is never zero because the
pair is disjoint. This is the v3 construction.

ACCURACY NOTE: for exactly one signature per dimension -- (7,2) at d=9,
(8,2) at d=10, (8,3) at d=11 -- blade_sign(a,a,sig)*tau[a] is +1 for EVERY
index a, so the scalar part of C = A tau(A) is sum(A[a]**2), i.e. tau is the
conjugation adapted to that metric. There C[0] dominates the rest of C by
roughly sqrt(2**d), and the sweep loses several orders of accuracy to
cancellation: median residual ~1e-2 at d=10 and d=11 against ~1e-10 for
every other signature at the same d. Use polish() with care there (it is
divergent in that regime) or work at higher precision.

Porting notes (C++ and friends)
-------------------------------
  * bin(x).count("1") is popcount; rev[3 & popcount] is a mod-4 lookup of
    the grade-k reversion sign (-1)**(k(k-1)/2).
  * blade_sign's swap counter accumulates masked popcounts and only its
    PARITY is used, so XOR-folding is equivalent. At d = 11 it stays
    below 64.
  * `pos` below is a build-time dict standing in for int32 pos[1 << d]
    initialised to -1, then pos[slots[i]] = i; `c in pos` is `pos[c] >= 0`.
  * Index arrays are int32, sign arrays int8. Both are load-bearing.
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                    # pragma: no cover
    HAVE_NUMBA = False                                 # kernels ~100x slower
    def njit(*a, **k):
        return (lambda f: f) if not a else a[0]

# ----------------------------------------------------------------- domain ---

def _invo_fix(d):
    """Thue-Morse sign layer over indexes; tau = reversion * fix.

    CONVENTION WARNING: the two branches extend the generator set at
    OPPOSITE ENDS, and the resulting templates are related by a relabelling
    of generators.
      even -> odd: appends the new generator as the HIGH bit; blades
                   containing it flip sign.  Index n + 2**(dd-1) <- -fix[n].
      odd  -> even: inserts the new generator at BIT 0, shifting every
                   existing generator up one, contributing no sign.
                   Indexes 2n and 2n+1 <- fix[n].
    Only the odd step's high-bit convention is relied on downstream (by the
    sig[:d-1] slice and the PS complement in TauInverse).
    """
    if d == 4:
        return [1] * 16                                # standalone template
    fix = [1] * 32                                     # invo_5 seed
    for dd in range(6, d + 1):
        if dd & 1:
            fix = fix + [-s for s in fix]              # even -> odd: complement
        else:
            fix = [s for s in fix for _ in (0, 1)]     # odd -> even: double
    return fix


def tau_signs(d):
    """tau[n] = rev[popcount(n) mod 4] * fix[n], over indexes n < 2**d.

    rev = (1, 1, -1, -1) is the grade-k reversion sign (-1)**(k(k-1)/2), so
    tau is reversion composed with the Thue-Morse layer from _invo_fix.
    """
    fix = _invo_fix(d)
    rev = (1, 1, -1, -1)
    return np.array([rev[3 & bin(n).count("1")] * fix[n] for n in range(1 << d)],
                    dtype=np.int64)


def tau_slots(d):
    """The tau-domain: indexes n with tau[n] > 0, ascending.

    slots[0] == 0 always. For odd d the set is closed under complement and
    contains the pseudoscalar; for even d it is neither. Both facts are
    load-bearing and are asserted where they are used.
    """
    t = tau_signs(d)
    return [m for m in range(1 << d) if t[m] > 0]


def blade_sign(a, b, sig):
    """Sign of e_a e_b -> e_{a^b}; 0 if a null generator is squared."""
    sw = 0
    for i in range(len(sig)):                          # swap parity
        if (b >> i) & 1:
            sw += bin(a >> (i + 1)).count("1")         # set bits of a above i
    s = -1 if sw & 1 else 1
    for i, si in enumerate(sig):                       # metric contribution
        if (a >> i) & 1 and (b >> i) & 1:
            s *= si                                    # si == 0 propagates
    return s


def _masks(sig):
    neg = sum(1 << i for i, s in enumerate(sig) if s < 0)
    nul = sum(1 << i for i, s in enumerate(sig) if s == 0)
    return neg, nul


def blade_signs(a, b, sig):
    """blade_sign over integer arrays a, b (broadcast), returning int8.

    Same arithmetic as blade_sign: swap parity from masked popcounts, metric
    parity from a & b & negmask, zero where a & b touches a null generator.
    """
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    d = len(sig)
    pop = _POP[:1 << d] if d <= _POPBITS else \
        np.array([bin(i).count("1") for i in range(1 << d)], dtype=np.int64)
    neg, nul = _masks(sig)
    a, b = np.broadcast_arrays(a, b)
    sw = np.zeros(a.shape, dtype=np.int64)
    for i in range(d):
        sw += ((b >> i) & 1) * pop[a >> (i + 1)]
    par = (sw + pop[a & b & neg]) & 1
    return np.where((a & b & nul) != 0, 0, 1 - 2 * par).astype(np.int8)


_POPBITS = 11
_POP = np.array([bin(i).count("1") for i in range(1 << _POPBITS)], dtype=np.int64)

# ------------------------------------------------------------ even tables ---

def _pair_rows(slots, rows, sig):
    """Rectangular pair table for the output tau-indexes in `rows`.

    For each output c = slots[i], i in rows, keep the pairs (a, b) of
    tau-elements with a ^ b == c and a < b, in ascending a -- the v6 order.
    Returns TA, TB (int32 tau-indexes), S (int8 signs), pairs.
    """
    sl = np.asarray(slots, dtype=np.int64)
    posf = np.full(1 << len(sig), -1, dtype=np.int64)
    posf[sl] = np.arange(len(sl))
    rows = np.asarray(rows, dtype=np.int64)
    a = np.broadcast_to(sl[None, :], (len(rows), len(sl)))
    b = a ^ sl[rows][:, None]
    pb = posf[b]
    keep = (pb >= 0) & (a < b)                     # keep one of (a,b),(b,a)
    counts = keep.sum(axis=1)
    pairs = int(counts[0])
    assert np.all(counts == pairs), "not rectangular"
    ri, ci = np.nonzero(keep)                      # row-major = v6 order
    A, B = a[ri, ci], b[ri, ci]
    S = blade_signs(A, B, sig)
    assert np.array_equal(S, blade_signs(B, A, sig)), \
        "pair (a,b)/(b,a) sign mismatch: symmetric fold invalid"
    sh = (len(rows), pairs)
    return (ci.astype(np.int32).reshape(sh),       # pos[a] == ci
            pb[ri, ci].astype(np.int32).reshape(sh),
            S.reshape(sh), pairs)


class EvenTable:
    """Rectangular tau-domain multiplication table for an even dimension.

    See the module Tables section. No counts[] array, no padding, no
    ragged rows: S0 carries signs for the scalar output, TA/TB/S carry the rest.
    """

    def __init__(self, d, sig):
        assert d % 2 == 0, "EvenTable is for even dimensions"
        assert len(sig) == d
        self.d, self.sig = d, list(sig)
        self.slots = tau_slots(d)
        n = len(self.slots)
        assert self.slots[0] == 0, "slot 0 must be the scalar blade"
        sl = np.asarray(self.slots, dtype=np.int64)

        # scalar output row: a ^ b == 0 forces b == a, so this is the squares
        self.S0 = blade_signs(sl, sl, sig)
        self.TA, self.TB, self.S, self.pairs = _pair_rows(
            self.slots, range(1, n), sig)

    def nbytes(self):
        return sum(x.nbytes for x in (self.S0, self.TA, self.TB, self.S))

    def real_mults(self):
        return len(self.slots) + 2 * self.pairs * (len(self.slots) - 1)


class OddTable:
    """Rectangular tau-domain table for an odd dimension, built natively.

    The two central outputs are perfect matchings of the domain and are
    handled outside the rectangular rows:
        scalar        a ^ a = 0        sign S0[i]  = sign(e_a e_a)
        pseudoscalar  a ^ (a^PS) = PS  sign PSG[i] = sign(e_a e_{a^PS})
    PSG is computed DIRECTLY on the disjoint pair, never as S0 * IS: that
    fusion expresses a nonzero sign as a product of two metric-carrying
    signs and fails as 0 * 0 under degeneracy.
        IS[i]  = sign(e_a I), may be 0: a genuine product, used by the
                 centre multiply (a + b I) X.
        comp[i] = tau-index of slots[i] ^ PS.
    Every other output row has the same length `pairs` (asserted).
    """

    def __init__(self, d, sig):
        assert d % 2 == 1, "OddTable is for odd dimensions"
        assert len(sig) == d
        self.d, self.sig = d, list(sig)
        self.slots = tau_slots(d)
        n = len(self.slots)
        assert self.slots[0] == 0, "slot 0 must be the scalar blade"
        PS = (1 << d) - 1
        assert PS in self.slots, "odd tau-domain must contain the pseudoscalar"
        self.pp = self.slots.index(PS)
        sl = np.asarray(self.slots, dtype=np.int64)
        posf = np.full(1 << d, -1, dtype=np.int64)
        posf[sl] = np.arange(n)

        self.S0 = blade_signs(sl, sl, sig)
        self.comp = posf[sl ^ PS].astype(np.int32)
        assert np.all(self.comp >= 0), "odd tau-domain not closed under complement"
        self.PSG = blade_signs(sl, sl ^ PS, sig)
        self.IS = blade_signs(sl, PS, sig)
        assert np.all(self.PSG != 0), "complement pairs are disjoint; PSG never 0"
        if all(s != 0 for s in sig):
            assert np.array_equal(self.PSG, self.S0 * self.IS), \
                "non-degenerate fusion identity PSG = S0 * IS broken"

        ro = [i for i in range(1, n) if i != self.pp]  # central rows excluded
        self.RO = np.array(ro, dtype=np.int32)
        self.TA, self.TB, self.S, self.pairs = _pair_rows(self.slots, ro, sig)

    def nbytes(self):
        return sum(x.nbytes for x in (self.S0, self.PSG, self.IS, self.comp,
                                      self.RO, self.TA, self.TB, self.S))

    def real_mults(self):
        return 2 * len(self.slots) + 2 * self.pairs * self.TA.shape[0]

# ---------------------------------------------------------------- kernels ---

@njit(cache=True)
def mul_even(u, v, S0, TA, TB, S, out):
    acc = 0.0
    for k in range(S0.shape[0]):                       # scalar row: diagonal
        s = S0[k]
        if s > 0:
            acc += u[k] * v[k]
        elif s < 0:
            acc -= u[k] * v[k]
    out[0] = acc
    for r in range(TA.shape[0]):
        acc = 0.0
        for k in range(TA.shape[1]):
            s = S[r, k]
            if s == 0:                                 # degenerate term
                continue
            a = TA[r, k]
            b = TB[r, k]
            p = u[a] * v[b] + u[b] * v[a]              # one sign, two terms
            if s > 0:
                acc += p
            else:
                acc -= p
        out[r + 1] = acc


@njit(cache=True)
def mul_even_scalar(u, v, S0):
    """Scalar component of u v only -- the diagonal row. len(slots) mults."""
    acc = 0.0
    for k in range(S0.shape[0]):
        s = S0[k]
        if s > 0:
            acc += u[k] * v[k]
        elif s < 0:
            acc -= u[k] * v[k]
    return acc


@njit(cache=True)
def mul_odd(X, Y, Se, I2, S0, TA, TB, S, out, t1, t2, t3):
    """(u1 + w1 I)(u2 + w2 I) using only the even table. Four even products.

    out must not alias X or Y: u1/w1/u2/w2 are views, not copies.
    """
    u1, w1 = X[:Se], X[Se:]
    u2, w2 = Y[:Se], Y[Se:]
    mul_even(u1, u2, S0, TA, TB, S, t1)                # u1 u2
    mul_even(w1, w2, S0, TA, TB, S, t2)                # w1 w2
    for i in range(Se):
        out[i] = t1[i] + I2 * t2[i]
    mul_even(u1, w2, S0, TA, TB, S, t1)                # u1 w2
    mul_even(w1, u2, S0, TA, TB, S, t3)                # w1 u2
    for i in range(Se):
        out[Se + i] = t1[i] + t3[i]


@njit(cache=True)
def centre_odd(X, Y, Se, I2, S0):
    """Central part (scalar, pseudoscalar) of an odd product. 4*Se mults.

    The full product would compute 4*real_mults() and discard all but these
    two components.
    """
    u1, w1 = X[:Se], X[Se:]
    u2, w2 = Y[:Se], Y[Se:]
    e0 = mul_even_scalar(u1, u2, S0) + I2 * mul_even_scalar(w1, w2, S0)
    eps = mul_even_scalar(u1, w2, S0) + mul_even_scalar(w1, u2, S0)
    return e0, eps


@njit(cache=True)
def centre_mul(a, b, I2, X, Se, out):
    """(a + b I) X for an odd tau-element X = u + w I. 2*width mults.

    I is central at odd d, so this is a pair of scalings, not a product.
    """
    ib = I2 * b
    for i in range(Se):
        out[i] = a * X[i] + ib * X[Se + i]
        out[Se + i] = a * X[Se + i] + b * X[i]


@njit(cache=True)
def coo_apply(a, b, o, s, A, X, out):
    """Scatter-accumulate a flat term list: out[o[k]] += s[k] A[a[k]] X[b[k]].

    Replaces np.add.at, which cannot be jitted and is ~5x slower here.
    """
    for i in range(out.shape[0]):
        out[i] = 0.0
    for k in range(a.shape[0]):
        out[o[k]] += s[k] * A[a[k]] * X[b[k]]


@njit(cache=True)
def mul_oddn(X, Y, S0, PSG, comp, pp, RO, TA, TB, S, out):
    """Native odd tau-domain product. 2M + 2*pairs*rows mults."""
    acc = 0.0
    for k in range(S0.shape[0]):                       # scalar: diagonal
        s = S0[k]
        if s != 0:
            acc += s * X[k] * Y[k]
    out[0] = acc
    acc = 0.0
    for k in range(PSG.shape[0]):                      # pseudoscalar: complement
        acc += PSG[k] * X[k] * Y[comp[k]]
    out[pp] = acc
    for r in range(TA.shape[0]):
        acc = 0.0
        for k in range(TA.shape[1]):
            s = S[r, k]
            if s == 0:                                 # degenerate term
                continue
            a = TA[r, k]
            b = TB[r, k]
            p = X[a] * Y[b] + X[b] * Y[a]
            if s > 0:
                acc += p
            else:
                acc -= p
        out[RO[r]] = acc


@njit(cache=True)
def centre_oddn(X, Y, S0, PSG, comp):
    """Scalar and pseudoscalar components of a native odd product. 2M mults."""
    e0 = 0.0
    eps = 0.0
    for k in range(S0.shape[0]):
        e0 += S0[k] * X[k] * Y[k]
        eps += PSG[k] * X[k] * Y[comp[k]]
    return e0, eps


@njit(cache=True)
def centre_muln(a, b, IS, comp, X, out):
    """(a + b I) X in the native layout. X I is a signed complement
    permutation with sign IS, which is 0 where e_m I vanishes."""
    for i in range(X.shape[0]):
        out[i] = a * X[i]
    for i in range(X.shape[0]):
        out[comp[i]] += b * IS[i] * X[i]

# ---------------------------------------------------------------- driver ----

_TWEAK = {
    4:  [1], 5:  [1],
    6:  [3, 1, 1 / 3], 7: [3, 1, 1 / 3],
    8:  [7, 3, 5 / 3, 1, 3 / 5, 1 / 3, 1 / 7],
    9:  [7, 3, 5 / 3, 1, 3 / 5, 1 / 3, 1 / 7],
    10: [15, 7, 13 / 3, 3, 11 / 5, 5 / 3, 9 / 7, 1,
         7 / 9, 3 / 5, 5 / 11, 1 / 3, 3 / 13, 1 / 7, 1 / 15],
    11: [15, 7, 13 / 3, 3, 11 / 5, 5 / 3, 9 / 7, 1,
         7 / 9, 3 / 5, 5 / 11, 1 / 3, 3 / 13, 1 / 7, 1 / 15],
}


class TauInverse:
    """Compiled tau-domain inverse for Cl(sig), 4 <= d <= 11."""

    def __init__(self, d, sig, native=None):
        """native: None chooses automatically (OddTable only when an odd
        signature has a null generator in sig[:d-1]); True/False forces
        the odd layout, e.g. to cross-validate the two paths."""
        assert 4 <= d <= 11, "supported range is d = 4..11 (model oracle range)"
        assert len(sig) == d
        sig = list(sig)
        assert all(s in (-1, 0, 1) for s in sig), "sig entries must be -1, 0, 1"
        self.d, self.sig, self.odd = d, sig, bool(d & 1)
        self.n = 1 << d
        self.tau = tau_signs(d).astype(np.float64)
        self.tweak = _TWEAK[d]
        PS = self.n - 1

        if native is None:
            native = self.odd and any(s == 0 for s in sig[:d - 1])
        assert not (native and not self.odd), "native layout is odd-d only"
        self.native = bool(native)

        if self.native:
            self.table = OddTable(d, sig)
            self.slots = self.table.slots
            self.width = len(self.slots)
            self.Se = self.width // 2
            self.pidx = self.table.pp                 # pseudoscalar tau-index
            self.I2 = float(blade_sign(PS, PS, sig))
            self.pos = {m: i for i, m in enumerate(self.slots)}
            self._build_bookends()
            self._scratch = [np.zeros(self.width) for _ in range(3)]
            self._t = []
            self._C = np.zeros(self.width)
            self._R = np.zeros(self.n)
            return

        de = d - 1 if self.odd else d
        # SIGNATURE CARE: odd d builds its table from the LOW generators only.
        self.table = EvenTable(de, sig[:de])
        self.Se = len(self.table.slots)
        self.pidx = self.Se                           # Z[Se] is the pseudoscalar

        if self.odd:
            PS = self.n - 1
            self.I2 = float(blade_sign(PS, PS, sig))
            # free signed permutation: e_m I = pssign[m] * e_{m^PS}
            self.pssign = np.array([blade_sign(m, PS, sig)
                                    for m in self.table.slots], dtype=np.float64)
            self.slots = self.table.slots + [m ^ PS for m in self.table.slots]
            self.width = 2 * self.Se
            # complement closure -- the whole odd path rests on this
            assert set(self.slots) == set(tau_slots(d)), \
                "odd tau-domain is not the even domain plus its complement"
        else:
            self.I2 = 0.0
            self.slots = self.table.slots
            self.width = self.Se
        self.pos = {m: i for i, m in enumerate(self.slots)}
        self._build_bookends()
        self._scratch = [np.zeros(self.width) for _ in range(3)]
        self._t = [np.zeros(self.Se) for _ in range(3)]
        self._C = np.zeros(self.width)
        self._R = np.zeros(self.n)

    # ---- bookends -------------------------------------------------------
    def _build_bookends(self):
        """B1: C = A tau(A) on domain outputs, symmetric pairs (one multiply,
        doubling is an add).  B2: A^-1 = tau(A) C^-1 over full outputs.

        Both are flat term lists with no shape to preserve, so zero-signed
        terms are DROPPED here -- unlike EvenTable, which keeps them in place
        to hold its rows rectangular.
        """
        n, sig, tau = self.n, self.sig, self.tau
        sl = np.asarray(self.slots, dtype=np.int64)
        posf = np.full(n, -1, dtype=np.int64)       # the int32 pos[] of the
        posf[sl] = np.arange(len(sl))               # porting notes
        sgnf = np.zeros(n)
        sgnf[sl] = [self._sgn(m) for m in self.slots]

        # B1: pairs a <= b in row-major order -- the diagonal square first,
        # then b > a ascending -- restricted to domain outputs c = a ^ b.
        A, B = np.triu_indices(n)
        C = A ^ B
        keep = posf[C] >= 0
        A, B, C = A[keep], B[keep], C[keep]
        s = blade_signs(A, B, sig) * tau[B]          # tau[b] == tau[a] on diag
        nz = s != 0
        A, B, C, s = A[nz], B[nz], C[nz], s[nz]
        self.B1 = (A.astype(np.int32), B.astype(np.int32),
                   posf[C].astype(np.int32))
        self.B1S = np.ascontiguousarray(
            s * sgnf[C] * (1.0 + (A != B)))          # folds the free x2

        # B2: tau(A) component a (outer) times C^-1 component c (inner).
        A = np.repeat(np.arange(n, dtype=np.int64), len(sl))
        Cc = np.tile(sl, n)
        s = blade_signs(A, Cc, sig) * tau[A]
        nz = s != 0
        A, Cc, s = A[nz], Cc[nz], s[nz]
        self.B2 = (A.astype(np.int32), posf[Cc].astype(np.int32),
                   (A ^ Cc).astype(np.int32))
        self.B2S = np.ascontiguousarray(s * sgnf[Cc])

    def _store(self, blade):
        """tau-index of a domain blade in the working representation."""
        return self.pos[blade]

    def _sgn(self, blade):
        """Sign relating stored coefficient to full-array coefficient.

        In the odd representation the complement half stores w, where
        A[m^PS] = pssign[m] * w[m]; the low half stores u directly.
        """
        if not self.odd or self.native:
            return 1.0
        i = self.pos[blade]
        return 1.0 if i < self.Se else self.pssign[i - self.Se]

    # ---- products -------------------------------------------------------
    def mul(self, X, Y, out):
        """tau-domain product. See the CLOSURE CAVEAT in the module docstring.

        X, Y, out have length self.width; out must not alias X or Y.
        """
        T = self.table
        if self.native:
            mul_oddn(X, Y, T.S0, T.PSG, T.comp, T.pp, T.RO, T.TA, T.TB, T.S,
                     out)
        elif self.odd:
            mul_odd(X, Y, self.Se, self.I2, T.S0, T.TA, T.TB, T.S,
                    out, self._t[0], self._t[1], self._t[2])
        else:
            mul_even(X, Y, T.S0, T.TA, T.TB, T.S, out)
        return out

    def bookend1(self, A):
        a, b, o = self.B1
        coo_apply(a, b, o, self.B1S, A, A, self._C)
        return self._C

    def bookend2(self, A, rC):
        a, b, o = self.B2
        coo_apply(a, b, o, self.B2S, A, rC, self._R)
        return self._R

    # ---- the sweep ------------------------------------------------------
    def sweep(self, C):
        """FLS tweak chain over the tau-domain.

        Returns (D, E0, EPS) where D is the adjugate and E = D C is central:
        E0 is its scalar component and EPS its pseudoscalar component (0 at
        even d). Only the central components of the final product are
        computed -- the rest are provably zero and were previously
        calculated and discarded.
        """
        D, W = self._scratch[0], self._scratch[1]
        D[:] = C
        T, Se, P = self.tweak, self.Se, self.pidx
        D[0] *= -T[0]
        if self.odd:
            D[P] *= -T[0]
        for t in T[1:]:
            self.mul(D, C, W)
            D, W = W, D
            D[0] *= -t
            if self.odd:
                D[P] *= -t
        if self.native:
            tb = self.table
            e0, eps = centre_oddn(D, C, tb.S0, tb.PSG, tb.comp)
        elif self.odd:
            e0, eps = centre_odd(D, C, Se, self.I2, self.table.S0)
        else:
            e0, eps = mul_even_scalar(D, C, self.table.S0), 0.0
        return D, e0, eps

    def polish(self, C, rC, steps=2):
        """Newton-Schulz refinement inside the tau-domain.

        CAVEAT: this converges only when max|1 - C rC| < 1. In the
        adapted-conjugation signatures noted in the module docstring it does
        not, and iterating makes the residual worse by orders of magnitude.
        This routine therefore measures the residual first and returns rC
        unchanged rather than diverging.
        """
        W, V = np.zeros(self.width), np.zeros(self.width)
        rC = rC.copy()
        for _ in range(steps):
            self.mul(C, rC, W)          # E = C rC
            W *= -1.0
            W[0] += 1.0                 # 1 - E
            if np.abs(W).max() >= 1.0:  # outside the convergence radius
                break
            W[0] += 1.0                 # 2 - E
            self.mul(rC, W, V)
            rC = V.copy()
        return rC

    def inverse(self, A, polish_steps=0):
        """Inverse of a full multivector A (length 2**d). Returns length 2**d.

        The returned array is internal scratch and is overwritten by the next
        call; copy it if you need to keep it.
        """
        C = self.bookend1(A)
        D, e0, eps = self.sweep(C)
        if self.odd:
            # composite centre: denominator is E' E with E' = E0 - EPS I
            det = e0 * e0 - self.I2 * eps * eps
            if det == 0.0:
                raise ZeroDivisionError("multivector is not invertible")
            rC = self._scratch[2]
            if self.native:
                centre_muln(e0, -eps, self.table.IS, self.table.comp, D, rC)
            else:
                centre_mul(e0, -eps, self.I2, D, self.Se, rC)
            rC /= det
        else:
            if e0 == 0.0:
                raise ZeroDivisionError("multivector is not invertible")
            rC = D / e0
        if polish_steps:
            rC = self.polish(C, rC, polish_steps)
        return self.bookend2(A, rC)

    # ---- reporting ------------------------------------------------------
    def cost(self):
        """Real multiplication counts. Matches the paper's table."""
        M = self.table.real_mults()
        if self.native:
            per_mul, tail = M, 2 * self.width
        else:
            per_mul = 4 * M if self.odd else M
            tail = 4 * self.Se if self.odd else len(self.table.slots)
        sweep = (len(self.tweak) - 1) * per_mul + tail
        centre = (2 * self.width + 4) if self.odd else 0
        return dict(table_bytes=self.table.nbytes(), mults_per_mul=per_mul,
                    sweep=sweep, centre=centre,
                    bookend1=len(self.B1S), bookend2=len(self.B2S),
                    total=sweep + centre + len(self.B1S) + len(self.B2S))
