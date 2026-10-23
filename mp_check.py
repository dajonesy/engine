# file: mp_check.py
"""
Extended-precision diagnostic for the model inverse.

Runs the model inverse at float64 and again at high precision, and reports
the max residual of A * inverse(A) - 1 for each.  The question it answers:
is the error at a given dimension conditioning (it collapses when precision
is raised) or structural (it does not)?

Usage
-----
    python mp_check.py            # d = 10 control, then d = 12
    python mp_check.py 12 13      # named dimensions only
"""

import sys
import time

import numpy as np

import clif.context as Clif
from clif.multivector import Accum
from clif.inverse import model_inverse

DPS = 50


def residual(P):
    """Max |coefficient| of P - 1, as a float."""
    return max(abs(float(P.Reg[i] - (1 if i == 0 else 0)))
               for i in range(P.bases))


def run(d, dps, seed=1):
    Clif.set_precision(0)          # build the context in float mode
    Clif.Cl(d)
    Clif.set_precision(dps)        # then switch before any Accum is made
    np.random.seed(seed)
    A = Accum.random()
    t = time.time()
    P = A * model_inverse.inverse(A)
    return residual(P), time.time() - t


def main(dims):
    print(f"{'d':>3}  {'float64':>10}  {'mp' + str(DPS):>10}   {'mp secs':>8}")
    for d in dims:
        e0, _ = run(d, 0)
        e1, t1 = run(d, DPS)
        print(f"{d:>3}  {e0:>10.2e}  {e1:>10.2e}   {t1:>8.1f}")


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [10, 12])
