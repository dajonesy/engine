#file: clif/inverse/sparse_fls.py
""" tau-domain FLS inverse on even tables.

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
Tables are built for EVEN dimensions only (4, 6, 8, 10). An odd dimension d
reuses the table of d-1 by writing every tau-element as u + w*I with u, w in
the even (d-1) domain. I is the pseudoscalar for dimension d:

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
    skipped at runtime, so rows stay rectangular. The INVERSE refuses them
    (build-time assert); the PRODUCT supports them. That path is reachable
    only via EvenTable/mul_even directly.

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

# ------------------------------------------------------------ even tables ---

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
        pos = {m: i for i, m in enumerate(self.slots)}      # build-time only

        # scalar output row: a ^ b == 0 forces b == a, so this is the squares
        self.S0 = np.array([blade_sign(m, m, sig) for m in self.slots],
                           dtype=np.int8)

        rows = []
        for i in range(1, n):
            c = self.slots[i]
            r = []
            for a in self.slots:
                b = a ^ c
                if b not in pos or a >= b:              # keep one of (a,b),(b,a)
                    continue
                s = blade_sign(a, b, sig)
                assert s == blade_sign(b, a, sig), \
                    "pair (a,b)/(b,a) sign mismatch: symmetric fold invalid"
                r.append((pos[a], pos[b], s))
            rows.append(r)
        self.pairs = len(rows[0]) if rows else 0
        assert all(len(r) == self.pairs for r in rows), "not rectangular"

        sh = (n - 1, self.pairs)
        self.TA = np.array([[k[0] for k in r] for r in rows],
                           dtype=np.int32).reshape(sh)
        self.TB = np.array([[k[1] for k in r] for r in rows],
                           dtype=np.int32).reshape(sh)
        self.S = np.array([[k[2] for k in r] for r in rows],
                          dtype=np.int8).reshape(sh)

    def nbytes(self):
        return sum(x.nbytes for x in (self.S0, self.TA, self.TB, self.S))

    def real_mults(self):
        return len(self.slots) + 2 * self.pairs * (len(self.slots) - 1)

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

    def __init__(self, d, sig):
        assert 4 <= d <= 11, "supported range is d = 4..11 (model oracle range)"
        assert len(sig) == d
        sig = list(sig)
        assert all(s != 0 for s in sig), (
            "degenerate signature: the tau-domain inverse is undefined "
            "(needs radical split + Neumann correction)")
        self.d, self.sig, self.odd = d, sig, bool(d & 1)
        self.n = 1 << d
        self.tau = tau_signs(d).astype(np.float64)
        self.tweak = _TWEAK[d]

        de = d - 1 if self.odd else d
        # SIGNATURE CARE: odd d builds its table from the LOW generators only.
        self.table = EvenTable(de, sig[:de])
        self.Se = len(self.table.slots)

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
        dom = set(self.slots)
        b1a, b1b, b1o, b1s, b1d = [], [], [], [], []
        for a in range(n):
            s = blade_sign(a, a, sig) * tau[a]
            if s:
                b1a.append(a); b1b.append(a); b1o.append(self._store(0))
                b1s.append(s); b1d.append(0.0)
            for b in range(a + 1, n):
                c = a ^ b
                if c not in dom:
                    continue
                s = blade_sign(a, b, sig) * tau[b]
                if s == 0:
                    continue
                b1a.append(a); b1b.append(b); b1o.append(self._store(c))
                b1s.append(s * self._sgn(c)); b1d.append(1.0)
        self.B1 = tuple(np.array(x, dtype=np.int32) for x in (b1a, b1b, b1o))
        self.B1S = np.ascontiguousarray(
            np.array(b1s) * (1.0 + np.array(b1d)))       # folds the free x2
        b2a, b2b, b2o, b2s = [], [], [], []
        for a in range(n):                                # tau(A) component
            for c in self.slots:                          # C^-1 component
                s = blade_sign(a, c, sig) * tau[a]
                if s == 0:
                    continue
                b2a.append(a); b2b.append(self._store(c)); b2o.append(a ^ c)
                b2s.append(s * self._sgn(c))
        self.B2 = tuple(np.array(x, dtype=np.int32) for x in (b2a, b2b, b2o))
        self.B2S = np.ascontiguousarray(np.array(b2s, dtype=np.float64))

    def _store(self, blade):
        """tau-index of a domain blade in the working representation."""
        return self.pos[blade]

    def _sgn(self, blade):
        """Sign relating stored coefficient to full-array coefficient.

        In the odd representation the complement half stores w, where
        A[m^PS] = pssign[m] * w[m]; the low half stores u directly.
        """
        if not self.odd:
            return 1.0
        i = self.pos[blade]
        return 1.0 if i < self.Se else self.pssign[i - self.Se]

    # ---- products -------------------------------------------------------
    def mul(self, X, Y, out):
        """tau-domain product. See the CLOSURE CAVEAT in the module docstring.

        X, Y, out have length self.width; out must not alias X or Y.
        """
        T = self.table
        if self.odd:
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
        T, Se = self.tweak, self.Se
        D[0] *= -T[0]
        if self.odd:
            D[Se] *= -T[0]
        for t in T[1:]:
            self.mul(D, C, W)
            D, W = W, D
            D[0] *= -t
            if self.odd:
                D[Se] *= -t
        if self.odd:
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
            rC = self._scratch[2]
            centre_mul(e0, -eps, self.I2, D, self.Se, rC)
            rC /= det
        else:
            rC = D / e0
        if polish_steps:
            rC = self.polish(C, rC, polish_steps)
        return self.bookend2(A, rC)

    # ---- reporting ------------------------------------------------------
    def cost(self):
        """Real multiplication counts. Matches the paper's table."""
        M = self.table.real_mults()
        per_mul = 4 * M if self.odd else M
        tail = 4 * self.Se if self.odd else len(self.table.slots)
        sweep = (len(self.tweak) - 1) * per_mul + tail
        centre = (2 * self.width + 4) if self.odd else 0
        return dict(table_bytes=self.table.nbytes(), mults_per_mul=per_mul,
                    sweep=sweep, centre=centre,
                    bookend1=len(self.B1S), bookend2=len(self.B2S),
                    total=sweep + centre + len(self.B1S) + len(self.B2S))
