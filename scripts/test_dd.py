"""Self-contained test of sparse_fls_dd (even and odd d).

Residual max|A A^-1 - 1| is evaluated by an independent dense product in dd,
so the check does not use the tau tables at all.

    python test_dd.py 10 11 12 13 14 15
    python test_dd.py 9:null    (a null generator inside sig[:d-1] -> native odd layout)
"""
import sys, time, timeit, resource
import numpy as np
from numba import njit
from clif.inverse.sparse_fls import TauInverse
from clif.inverse.sparse_fls_dd import DDTauInverse, dd_add, dd_mul


class F64TauInverse(DDTauInverse):
    """Same construction as DDTauInverse, float64 sweep (baseline past d=11)."""
    sweep = TauInverse.sweep


@njit(inline='always')
def _sign(a, b, neg, nul):
    if a & b & nul:
        return 0.0
    sw = 0
    x = a >> 1
    while x:
        y = x & b
        while y:
            sw += 1
            y &= y - 1
        x >>= 1
    y = a & b & neg
    while y:
        sw += 1
        y &= y - 1
    return -1.0 if sw & 1 else 1.0


@njit(cache=True)
def residual_dd(A, X, neg, nul):
    n = A.shape[0]
    Rh = np.zeros(n)
    Rl = np.zeros(n)
    for a in range(n):
        if A[a] == 0.0:
            continue
        for b in range(n):
            sg = _sign(a, b, neg, nul)
            if sg == 0.0:
                continue
            ph, pl = dd_mul(A[a], 0.0, X[b], 0.0)
            if sg < 0.0:
                ph, pl = -ph, -pl
            c = a ^ b
            Rh[c], Rl[c] = dd_add(Rh[c], Rl[c], ph, pl)
    Rh[0], Rl[0] = dd_add(Rh[0], Rl[0], -1.0, 0.0)
    m = 0.0
    for i in range(n):
        v = abs(Rh[i] + Rl[i])
        if v != v:                                     # NaN: report it
            return v
        if v > m:
            m = v
    return m


def run(d, sig=None, seed=0):
    sig = sig or [1] * d
    neg = sum(1 << i for i, s in enumerate(sig) if s < 0)
    nul = sum(1 << i for i, s in enumerate(sig) if s == 0)
    A = np.random.default_rng(seed).standard_normal(1 << d)
    t = time.perf_counter()
    dd = DDTauInverse(d, sig)
    build = time.perf_counter() - t
    f64 = F64TauInverse.__new__(F64TauInverse)
    f64.__dict__.update(dd.__dict__)                   # share tables
    f64._scratch = [np.zeros(dd.width) for _ in range(3)]
    f64._t = [np.zeros(dd.Se) for _ in range(3)]
    f64._R = np.zeros(dd.n); f64._C = np.zeros(dd.width)
    out = {}
    pairs = [("dd", dd)] if d >= 14 else [("f64", f64), ("dd", dd)]
    for name, inv in pairs:
        X = inv.inverse(A).copy()
        reps = 3 if d >= 12 else 20
        t = min(timeit.repeat(lambda: inv.inverse(A), number=1, repeat=reps))
        out[name] = (residual_dd(A, X, neg, nul), t)
    if d <= 11:                        # step down/up vs v7 term lists
        ref = TauInverse(d, sig)
        rC = np.random.default_rng(seed + 1).standard_normal(dd.width)
        c1, c2 = ref.bookend1(A).copy(), dd.bookend1(A).copy()
        r1, r2 = ref.bookend2(A, rC).copy(), dd.bookend2(A, rC).copy()
        assert abs(c1 - c2).max() <= 1e-13 * abs(c1).max(), "step down != v7"
        assert abs(r1 - r2).max() <= 1e-13 * abs(r1).max(), "step up != v7"
    return build, out


def parse(tok):
    """'9' -> (9, Euclidean); '9:null' -> (9, a null generator at e_2)."""
    d, _, kind = tok.partition(":")
    d = int(d)
    sig = [1] * d
    if kind == "null":
        sig[2] = 0
    return d, sig, tok


if __name__ == "__main__":
    toks = sys.argv[1:] or ["10", "11", "12", "13", "14", "15"]
    print(f"| {'d':>4} | {'N':>4} | {'build s':>8} | {'f64 resid':>9} | {'dd resid':>9}"
          f" | {'f64 ms':>9} | {'dd ms':>9} | {'ratio':>5} | {'peak MB':>7} |")
    print("|------|------|----------|-----------|-----------|-----------|---------|-------|---------|")
    for tok in toks:
        d, sig, lab = parse(tok)
        b, o = run(d, sig)
        rd, td = o["dd"]
        if "f64" in o:
            rf, tf = o["f64"]
            fr, ft, ra = f"{rf:9.2e}", f"{tf*1e3:9.3f}", f"{td/tf:5.3f}"
        else:
            fr, ft, ra = f"{'  --  ':>9}", f"{'  --  ':>9}", f"{'--':>5}"
        print(f"| {lab:>4} | {1 << (d // 2 - 1):>4} | {b:8.3f} | {fr}"
              f" | {rd:9.2e} | {ft} | {td*1e3:9.3f} | {ra} |"
              f" {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:7.0f} |")
