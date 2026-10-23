#!/usr/bin/env python3
"""Signature sweep for the sparse FLS inverse.

For every Clifford algebra Cl(p,q,r) with 4 <= p+q+r <= 11 (344 signatures),
invert one random multivector and check the residual of A * inverse(A) - 1.
If that first draw misses TOL, RETRIES more draws are taken:
    ok          first draw passed
    ok (retry)  first draw missed, every retry passed (near-singular draw)
    flaky       some retries passed
    choke       nothing passed; `resid` is the median over all draws

The geometric product used for the residual is computed here from scratch,
independent of the engine under test.  Degenerate generators (r > 0) are
handled: any blade product containing a repeated null generator is zero.

Usage:
    python sig_sweep.py                 # sweep with the engine's inverse
    python sig_sweep.py --oracle        # sweep with a matrix-inverse oracle
    python sig_sweep.py --min-d 10      # start at d = 10
    python sig_sweep.py --max-d 8       # stop at d = 8
"""

import argparse
import csv
import sys
import time

import numpy as np

TOL, RETRIES = 1e-8, 3

# --------------------------------------------------------------------------
# ADAPTER -- the one place to edit if the engine's entry point differs.
# --------------------------------------------------------------------------

def signature(p, q, r):
    """Generator squares, in order: p of +1, q of -1, r of 0."""
    return [1] * p + [-1] * q + [0] * r


def engine_inverse(p, q, r):
    """Return a callable A -> iA on full 2^d ordinal arrays."""
    from clif.inverse.sparse_fls import TauInverse
    return TauInverse(p + q + r, signature(p, q, r)).inverse


# --------------------------------------------------------------------------
# Independent geometric product
# --------------------------------------------------------------------------

class Product:
    """Blade multiplication tables for one signature."""

    def __init__(self, p, q, r):
        d = p + q + r
        n = 1 << d
        self.d, self.n = d, n
        self.negmask = ((1 << q) - 1) << p
        self.nullmask = ((1 << r) - 1) << (p + q)

        a = np.arange(n, dtype=np.int32)[:, None]
        b = np.arange(n, dtype=np.int32)[None, :]
        pop = np.array([bin(i).count("1") for i in range(n)], dtype=np.int32)

        swaps = np.zeros((n, n), dtype=np.int32)       # generator-swap count
        x = a >> 1
        for _ in range(d):
            swaps += pop[x & b]
            x = x >> 1
        par = (swaps & 1) ^ (pop[a & b & self.negmask] & 1)

        sign = np.where(par == 0, 1, -1).astype(np.int8)
        sign[(a & b & self.nullmask) != 0] = 0         # repeated null generator

        self.sign = sign
        self.idx = (a ^ b).astype(np.int32)
        self.one = np.zeros(n)
        self.one[0] = 1.0

    def mul(self, A, B):
        M = (A[:, None] * B[None, :]) * self.sign
        return np.bincount(self.idx.ravel(), weights=M.ravel(), minlength=self.n)

    def matrix(self, A):
        """Left-multiplication matrix L with L @ B == A * B."""
        L = np.zeros((self.n, self.n))
        np.add.at(L, (self.idx, np.arange(self.n)[None, :]), A[:, None] * self.sign)
        return L


def oracle_inverse(p, q, r):
    """Inverse by solving the left-regular representation (validates the
    harness itself)."""
    prod = Product(p, q, r)
    return lambda A: np.linalg.solve(prod.matrix(A), prod.one)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def signatures(min_d, max_d):
    for d in range(min_d, max_d + 1):
        for p in range(d + 1):
            for q in range(d - p + 1):
                yield p, q, d - p - q


def draw(inv, prod, rng):
    """Relative residual for one random A; inf on any failure."""
    A = rng.standard_normal(prod.n)
    try:
        iA = np.array(inv(A), dtype=float).ravel()
    except Exception:
        return np.inf
    if not np.all(np.isfinite(iA)):
        return np.inf
    r = np.abs(prod.mul(A, iA) - prod.one).max()
    s = np.linalg.norm(A) * np.linalg.norm(iA)
    return r / s if s else r


def sweep(min_d, max_d, use_oracle, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for p, q, r in signatures(min_d, max_d):
        d = p + q + r
        tag = f"Cl({p},{q},{r})"
        t0 = time.perf_counter()
        try:
            inv = oracle_inverse(p, q, r) if use_oracle else engine_inverse(p, q, r)
        except Exception as exc:                        # build-time refusal
            rows.append(dict(sig=tag, d=d, draws=0, resid="",
                             status=f"build: {type(exc).__name__}: {exc}"[:90],
                             secs=f"{time.perf_counter() - t0:.1f}"))
            print(f"{tag:<12} {rows[-1]['status']}", flush=True)
            continue

        prod = Product(p, q, r)
        res = [draw(inv, prod, rng)]
        status = "ok"
        if res[0] > TOL:                                # near-singular or choking
            res += [draw(inv, prod, rng) for _ in range(RETRIES)]
            passed = sum(x <= TOL for x in res[1:])
            status = ("ok (retry)" if passed == RETRIES else
                      "flaky" if passed else "choke")
        resid = float(np.median(res)) if status in ("flaky", "choke") else \
            (res[0] if status == "ok" else max(res[1:]))

        rows.append(dict(sig=tag, d=d, draws=len(res), resid=f"{resid:.1e}",
                         status=status, secs=f"{time.perf_counter() - t0:.1f}"))
        print(f"{tag:<12} d={d:<3} resid={rows[-1]['resid']:<9} "
              f"{status:<11} {rows[-1]['secs']}s", flush=True)
    return rows


def report(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sig", "d", "draws", "resid",
                                           "status", "secs"])
        w.writeheader()
        w.writerows(rows)

    clean = [r for r in rows if r["status"].startswith("ok")]
    bad = [r for r in rows if not r["status"].startswith("ok")]
    print(f"\n{len(clean)}/{len(rows)} signatures clean; results in {path}")
    if clean:
        print(f"worst clean residual: {max(float(r['resid']) for r in clean):.1e}")
    for r in bad:
        print(f"  {r['status'].upper():<6} {r['sig']}: resid {r['resid'] or '-'}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-d", type=int, default=4)
    ap.add_argument("--max-d", type=int, default=11)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", default="sig_sweep.csv")
    args = ap.parse_args()
    sys.exit(report(sweep(args.min_d, args.max_d, args.oracle, args.seed),
                    args.out))


if __name__ == "__main__":
    main()
