# file: scripts/benchmark_inverse.py
"""End-to-end inverse benchmark per dimension, for the AGACSE proceedings.

Every case inverts the same multivector in Cl(d, 0) and is timed at the
level a user would call it.  Each row reports the best per-call time and
the residual max|A A^-1 - 1|, the residual always computed with clif's own
product so that every case is measured the same way.

Cases
-----
    model        clif.inverse.model_inverse.inverse(A)      reference oracle
    tau-FLS      A.inverse()                                 float64 d<=11, dd d>=12
    tau-FLS+NS   newton_schulz(A, A.inverse())               one polish on top
    native FLS   clif.inverse.fls.fls_inverse(A)             classical Shirokov
    pygae        layout.MultiVector(...).inv()               pygae clifford (LAPACK)
    model/pygae  _model_on_pygae(x)                          the model, written with
                                                             pygae's product only
    kingdon      kingdon_inverse.shirokov_inverse(x)         off by default; kingdon's
                                                             own inv() fails for d >= 5

model/pygae is the portability case: the same tau-FLS algorithm as the
model, using nothing but pygae's geometric product, scalar arithmetic, and
the tau-involution template permuted into pygae's component order.  It is
timed against pygae's own inverse in the same package.

Converting A into pygae's or kingdon's representation happens outside the
timed call, as it would for a user already working in that package.  The
first call of every case is an untimed warm-up (Numba compilation, table
builds, kingdon code generation).  Signature is Euclidean, so pygae's and
kingdon's generator order agrees with clif's.

Usage
-----
    cd ~/project/engine
    python scripts/benchmark_inverse.py

or in a notebook

    from scripts.benchmark_inverse import time_inverse
    rows = time_inverse(dims=range(4, 12))

kingdon is off by default; time_inverse(cases=CASES + ["kingdon"]) adds it.
It is slow from d = 9 (about 3 s per call at d = 9, 18 s at d = 10 and
roughly 2.5 min at d = 11).
"""
import math
import sys
import timeit
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # engine root

import clif.context as Clif
from clif.multivector import Accum
from clif.inverse import model_inverse as MI
from clif.inverse.fls import fls_inverse
from clif.inverse.newton import newton_schulz
from clif.inverse.sparse_fls import tau_signs

try:
    import kingdon
    try:
        from clif.inverse.kingdon_inverse import shirokov_inverse as k_inverse
    except ImportError:
        from kingdon_inverse import shirokov_inverse as k_inverse
except ImportError:
    kingdon = None

try:
    import clifford as cf               # pygae
    if not hasattr(cf, "Cl"):
        cf = None
except ImportError:
    cf = None


CASES = ["model", "tau-FLS", "tau-FLS+NS", "native FLS", "pygae", "model/pygae"]


# ------------------------------------------------------------------ cases ---
# Each builder takes (d, A) and returns a zero-argument callable that runs
# the inverse, plus a function turning that call's result into a clif
# coefficient array.  Setup outside the callable is not timed.

def _model(d, A):
    return (lambda: MI.inverse(A)), (lambda r: r.Reg.copy())


def _tau(d, A):
    return (lambda: A.inverse()), (lambda r: r.Reg.copy())


def _tau_ns(d, A):
    return (lambda: newton_schulz(A, A.inverse())), (lambda r: r.Reg.copy())


def _native(d, A):
    return (lambda: fls_inverse(A)), (lambda r: r.Reg.copy())


def _pygae_setup(d, A):
    """pygae layout, clif index of each pygae position, A as a pygae MultiVector."""
    layout, _ = cf.Cl(d)
    # pygae position i holds the blade whose generators are bladeTupList[i]
    base = min(t[0] for t in layout.bladeTupList if len(t) == 1)   # lowest generator label
    perm = np.array([sum(1 << (k - base) for k in t) for t in layout.bladeTupList])
    x = layout.MultiVector(value=A.Reg[perm])

    def back(r):
        X = np.empty(1 << d)
        X[perm] = r.value
        return X
    return layout, perm, x, back


def _pygae(d, A):
    layout, perm, x, back = _pygae_setup(d, A)
    return (lambda: x.inv()), back


def _model_on_pygae(d, A):
    r"""The model's tau-FLS, written with pygae's geometric product only.

    Step down $C = A\,\tau(A)$; ladder $D \leftarrow D C$ with the scalar
    (and, at odd $d$, the pseudoscalar) scaled by $-T_k$, $T_k = n/k - 1$,
    $n = 2^{\lfloor d/2 \rfloor - 1}$; final step $E = D C$, central; step up
    $A^{-1} = \tau(A)\, D\, E^{-1}$.  Full products: $n + 2$ at even $d$,
    $n + 3$ at odd $d$ (the central $E^{-1}$ is applied as one product).
    """
    layout, perm, x, back = _pygae_setup(d, A)
    tau = tau_signs(d).astype(np.float64)[perm]          # template in pygae order
    n = 1 << (d // 2 - 1)
    tweak = [-(n / k - 1.0) for k in range(1, n)]
    odd = bool(d & 1)
    ps = int(np.flatnonzero(perm == (1 << d) - 1)[0])    # pygae position of I
    e_0 = np.zeros(1 << d); e_0[0] = 1.0                 # unit scalar
    e_I = np.zeros(1 << d); e_I[ps] = 1.0                # unit pseudoscalar
    I = layout.MultiVector(value=e_I)
    I2 = (I * I).value[0]
    MV = layout.MultiVector

    def run():
        B = MV(value=tau * x.value)                      # tau(A)
        C = x * B                                        # step down
        D = MV(value=C.value.copy())
        D.value[0] *= tweak[0]
        if odd:
            D.value[ps] *= tweak[0]
        for t in tweak[1:]:
            D = D * C
            D.value[0] *= t
            if odd:
                D.value[ps] *= t
        E = D * C                                        # central
        e0 = E.value[0]
        if not odd:
            return (B * D) * (1.0 / e0)                  # step up
        eps = E.value[ps]
        det = e0 * e0 - I2 * eps * eps
        Einv = MV(value=(e0 * e_0 - eps * e_I) / det)
        return (B * D) * Einv                            # step up
    return run, back


def _kingdon(d, A):
    alg = kingdon.Algebra(d, 0)
    x = alg.multivector(values=A.Reg.copy(), keys=tuple(range(1 << d)))

    def back(r):
        X = np.zeros(1 << d)
        for k, v in zip(r.keys(), r.values()):
            X[k] = v
        return X
    return (lambda: k_inverse(x)), back


BUILDERS = {"model": _model, "tau-FLS": _tau, "tau-FLS+NS": _tau_ns,
            "native FLS": _native, "pygae": _pygae, "model/pygae": _model_on_pygae,
            "kingdon": _kingdon}


# ---------------------------------------------------------------- helpers ---

def _residual(A, X):
    P = (A * Accum(X)).Reg
    return max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))))


def _best_time(f):
    """Best per-call time: batches sized to ~0.2 s, best of 5.

    A call slower than 1 s is timed once.
    """
    t0 = perf_counter()
    f()
    one = perf_counter() - t0
    if one > 1.0:
        return one
    number = max(1, int(0.2 / one))
    return min(timeit.repeat(f, number=number, repeat=5)) / number


def _available(case):
    if case in ("pygae", "model/pygae"):
        return cf is not None
    if case == "kingdon":
        return kingdon is not None
    return True


# ------------------------------------------------------------------- main ---

def time_inverse(dims=range(4, 12), cases=CASES, skip=(), seed=0):
    """Time every case at every dimension; print two Markdown tables.

    Returns a list of dicts: {"d", "case", "seconds", "residual"}.
    """
    skip = set(skip)
    rows = []
    for d in dims:
        Clif.Cl(d)
        A = Accum(np.random.default_rng(seed + d).standard_normal(1 << d))
        for case in cases:
            if (case, d) in skip or not _available(case):
                continue
            try:
                run, back = BUILDERS[case](d, A)
                res = _residual(A, back(run()))     # warm-up call, untimed
                sec = _best_time(run)
            except Exception as e:                  # a case that cannot run
                print(f"d={d} {case}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            rows.append(dict(d=d, case=case, seconds=sec, residual=res))
            print(f"d={d:2d}  {case:<11} {sec * 1e6:12.4g} us   residual {res:.1e}",
                  file=sys.stderr, flush=True)
    _print_tables(rows, list(dims), cases)
    return rows


MS_FROM = 8          # d >= MS_FROM is reported in milliseconds, below in microseconds


def _sig(v, n=4):
    """n significant figures, fixed point (no exponent)."""
    if v == 0:
        return "0"
    dec = max(0, n - 1 - math.floor(math.log10(abs(v))))
    return f"{v:.{dec}f}"


def _emit(header, body):
    """Print a Markdown pipe table with every column padded to a common width.

    body rows are lists of cells; a row given as a plain string is a label
    row, printed in the first column with the others left blank.
    """
    ncol = len(header)
    grid = [r if isinstance(r, list) else [r] + [""] * (ncol - 1) for r in body]
    w = [max(len(row[j]) for row in [header] + grid) for j in range(ncol)]

    def line(cells, label=False):
        out = [cells[0].ljust(w[0])]
        out += [c.rjust(w[j]) for j, c in enumerate(cells[1:], 1)]
        return "| " + " | ".join(out) + " |"

    print(line(header))
    print("|" + "|".join("-" * (x + 1) + (":" if j else "-") for j, x in enumerate(w)) + "|")
    for r in grid:
        print(line(r))


def _print_tables(rows, dims, cases):
    look = {(r["d"], r["case"]): r for r in rows}
    shown = [c for c in cases if any((d, c) in look for d in dims)]
    header = ["$d$"] + shown

    def cells(d, fmt):
        return [str(d)] + [fmt(look[(d, c)]) if (d, c) in look else "---" for c in shown]

    us = [d for d in dims if d < MS_FROM]
    ms = [d for d in dims if d >= MS_FROM]
    body = []
    if us:
        body.append(r"*(values in $\mu$s)*")
        body += [cells(d, lambda r: _sig(r["seconds"] * 1e6)) for d in us]
    if ms:
        body.append("*(values in ms)*")
        body += [cells(d, lambda r: _sig(r["seconds"] * 1e3)) for d in ms]
    print("\nTime per inverse\n")
    _emit(header, body)

    print("\nResidual $\\max\\lvert A A^{-1} - 1 \\rvert$\n")
    _emit(header, [cells(d, lambda r: f"{r['residual']:.1e}") for d in dims])


if __name__ == "__main__":
    time_inverse()
