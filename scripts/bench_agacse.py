"""bench_agacse.py — timings for the AGACSE'26 talk.

Your package and pygae are both named `clifford`, so they cannot be imported
in the same process. Run twice, from different directories:

    cd ~/project/engine       && python3 ~/bench_agacse.py engine
    cd /tmp                   && python3 ~/bench_agacse.py pygae

Each prints a markdown table row set. Paste both into the slide.
"""
import sys, time, statistics
import numpy as np

DIMS = [4, 5, 6, 7, 8, 9, 10, 11]


def timeit(fn, reps):
    fn()                                   # warm (JIT / LAPACK)
    best = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best.append((time.perf_counter() - t0) / reps * 1e6)
    return min(best)


def bench_engine():
    import clifford.context as Clif
    from clifford.multivector import Accum
    print("| d | engine inverse (us) | residual |")
    print("|---|---|---|")
    for d in DIMS:
        try:
            Clif.Cl(d)
            A = Accum(); A.random()
            rA = A.inverse()
            r = (A * rA).Reg
            res = max(abs(r[0] - 1.0), np.abs(r[1:]).max())
            reps = 2000 if d <= 8 else (200 if d <= 10 else 50)
            us = timeit(A.inverse, reps)
            print(f"| {d} | {us:.1f} | {res:.1e} |", flush=True)
        except Exception as e:
            print(f"| {d} | — | {type(e).__name__}: {str(e)[:40]} |", flush=True)


def bench_pygae():
    import clifford as cf
    import warnings; warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    print("| d | pygae .inv() (us) | shirokov (us) | best |")
    print("|---|---|---|---|")
    for d in DIMS:
        try:
            layout, _ = cf.Cl(d)
            A = layout.MultiVector(value=rng.standard_normal(layout.gaDims))
            reps = 200 if d <= 8 else 20
            t_inv = timeit(A.inv, reps)
            try:
                t_shi = timeit(A.shirokov_inverse, max(2, reps // 10))
                s = f"{t_shi:.1f}"
                best = "laInv" if t_inv <= t_shi else "shirokov"
            except Exception:
                s, best = "—", "laInv"
            print(f"| {d} | {t_inv:.1f} | {s} | {best} |", flush=True)
        except Exception as e:
            print(f"| {d} | — | — | {type(e).__name__} |", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "engine"
    print(f"\n### {which}\n")
    (bench_engine if which == "engine" else bench_pygae)()
