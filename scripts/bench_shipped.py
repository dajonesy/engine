# file: scripts/bench_shipped.py
r"""As-shipped timing: one geometric product and one inverse per dimension.

For each $d$ the script registers $\mathrm{Cl}(d, 0)$, draws two random
multivectors with standard-normal coefficients, and times exactly what a
user calls:

    A * B          the geometric product
    A.inverse()    the inverse, dispatched by tau_api:
                   d = 0..3 closed form, 4..11 float64 tau-FLS,
                   12..15 double-double tau-FLS

Each time is the best of 5 batches of about 0.2 s, after an untimed
warm-up call (Numba compilation, table builds); a call slower than 1 s is
timed once.  The residual max|A A^{-1} - 1| uses clif's own product.

Output is a LaTeX table (booktabs, siunitx) on stdout, or into a file:

    python scripts/bench_shipped.py                 # d = 0..15
    python scripts/bench_shipped.py 0 11            # d = 0..11
    python scripts/bench_shipped.py 0 15 shipped.tex

Progress lines go to stderr.  The PARALLEL settings in force are written
into the table as a comment, so the table records the configuration it
measured.
"""
import sys
import timeit
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # engine root

import clif.context as Clif
from clif.multivector import Accum
import clif.inverse.sparse_fls as SF
import clif.inverse.sparse_fls_dd as SFDD

# unit blocks: (first d, last d, label, scale)
BLOCKS = [(0, 3, r"\si{\nano\second}", 1e9),
          (4, 9, r"\si{\micro\second}", 1e6),
          (10, 15, r"\si{\milli\second}", 1e3)]


def _best_time(f):
    t0 = perf_counter()
    f()
    one = perf_counter() - t0
    if one > 1.0:
        return one
    number = max(1, int(0.2 / max(one, 1e-9)))
    return min(timeit.repeat(f, number=number, repeat=5)) / number


def measure(d, seed=0):
    Clif.Cl(d)
    rng = np.random.default_rng(seed + d)
    A = Accum(rng.standard_normal(1 << d))
    B = Accum(rng.standard_normal(1 << d))
    X = A.inverse()                                    # warm-up, kept for residual
    P = (A * X).Reg
    res = max(abs(P[0] - 1.0), float(np.max(np.abs(P[1:]))) if d else 0.0)
    t_mul = _best_time(lambda: A * B)
    t_inv = _best_time(lambda: A.inverse())
    return t_mul, t_inv, res


def _fmt(v):
    """Four significant figures, fixed point."""
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 100:
        return f"{v:.1f}"
    if v >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _exp(r):
    if r == 0.0:
        return "0"
    m, e = f"{r:.1e}".split("e")
    return rf"${m} \times 10^{{{int(e)}}}$"


def tex(rows):
    out = [
        f"% PARALLEL_FROM: sparse_fls={SF.PARALLEL_FROM}, PARALLEL: sparse_fls_dd={SFDD.PARALLEL}",        r"\begin{table}[tb]",
        r"\centering",
        r"\caption{As-shipped timing: one geometric product and one inverse, "
        r"$\mathrm{Cl}(d,0)$, random multivectors.}",
        r"\label{tbl:shipped}",
        r"\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$d$ & product & inverse & \makecell{inverse /\\product} & residual \\",
        r"\midrule",
    ]
    first = True
    for lo, hi, unit, scale in BLOCKS:
        block = [r for r in rows if lo <= r[0] <= hi]
        if not block:
            continue
        if not first:
            out.append(r"\midrule")
        first = False
        out.append(rf"\multicolumn{{5}}{{l}}{{\textit{{(values in {unit})}}}} \\")
        for d, tm, ti, res in block:
            out.append(f"{d:2d} & {_fmt(tm * scale)} & {_fmt(ti * scale)} & "
                       f"{ti / tm:.2f} & {_exp(res)} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main(argv):
    lo = int(argv[0]) if len(argv) > 0 else 0
    hi = int(argv[1]) if len(argv) > 1 else 15
    path = argv[2] if len(argv) > 2 else None
    rows = []
    for d in range(lo, hi + 1):
        try:
            tm, ti, res = measure(d)
        except Exception as e:                         # report and carry on
            print(f"d={d:2d}  {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        rows.append((d, tm, ti, res))
        print(f"d={d:2d}  product {tm * 1e6:12.4f} us  inverse {ti * 1e6:12.4f} us"
              f"  residual {res:.1e}", file=sys.stderr, flush=True)
    text = tex(rows)
    if path:
        Path(path).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
