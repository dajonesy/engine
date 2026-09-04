"""tau_api — the whole public surface of sparse_fls.

You never need to read sparse_fls.py to use it. It exposes one object,
TauInverse(sig), with one useful method:

    TauInverse(sig).inverse(A)   ->  A^-1

where A is a plain numpy array of length 2**d in ORDINAL (bitmask/XOR)
indexing -- i.e. exactly Accum's coefficient array. Output is the same shape.

Building a TauInverse costs real time (it constructs the shared table), so
build once per signature and keep it. That is all this module does.
"""
import numpy as np
from .sparse_fls import TauInverse
# sparse_fls should be v4

_CACHE = {}


def get_inverter(sig):
    """Return the (cached) inverter for this dimension and signature.

    sig is a sequence of +1/-1, ordered like the generators:
    sig[i] is e_i squared. Cl(3,1) -> [1, 1, 1, -1].
    the length of sig is of course the dimension
    """
    d = len(sig)
    key = (int(d), tuple(int(s) for s in sig))
    if key not in _CACHE:
        _CACHE[key] = TauInverse(key[0], list(key[1]))
    return _CACHE[key]

def inverse(A, sig):
    """One-shot inverse of an ordinal coefficient array."""
    return get_inverter(sig).inverse(np.asarray(A, dtype=np.float64))

def warm(sigs):
    """Pre-build tables so the first real call is not the slow one.

        warm([[-1,1,1,1],[1,-1,-1,-1]])
    """
    for sig in sigs:
        get_inverter(sig)


# --------------------------------------------------------------------------
# 
# Accum uses it: add this method to the class (or monkey-patch it on).
#
#     from tau_api import get_inverter
#
#     class Accum:
#         ...
#         def inverse(self):
#             inv = get_inverter(self.ctx.d, self.ctx.sig)
#             return Accum(self.ctx, inv.inverse(self.coeffs))
#
# The context already knows d and sig, so nothing needs registering by hand:
# the first inverse in a given algebra builds the table, every later one
# reuses it. If Accum stores coefficients under a different attribute name,
# swap `self.coeffs` for that.
#
# Whenever a new context is specified the routine context.initialize() gets
# the inverse entry point and saves it for class Accum (multivector.py) to 
# call.  
# --------------------------------------------------------------------------


if __name__ == "__main__":
    # Demo + self-check: uses the API exactly as a caller would, and verifies
    # the result against an independent full-algebra product defined here, so
    # this block depends on nothing outside the package.

    def _full_mul(A, B, sig):
        d = len(sig)
        n = 1 << d
        C = np.zeros(n)
        idx = np.arange(n)
        for a in range(n):
            if A[a] == 0.0:
                continue
            s = np.ones(n)
            sw = np.zeros(n, dtype=np.int64)
            for i in range(d):
                sw += ((idx >> i) & 1) * bin(a >> (i + 1)).count("1")
                if sig[i] != 1 and (a >> i) & 1:
                    s *= np.where((idx >> i) & 1, sig[i], 1.0)
            s *= np.where(sw & 1, -1.0, 1.0)
            C[a ^ idx] += s * A[a] * B
        return C

    for sig in [([1]*4), ([1]*6), ([1]*6 + [-1]), ([1]*9)]:
        d = len(sig)
        A = np.random.default_rng(0).standard_normal(1 << d)
        rA = inverse(A, sig)
        ident = _full_mul(A, rA, sig)
        err = max(abs(ident[0] - 1), np.abs(ident[1:]).max())
        p, q = sig.count(1), sig.count(-1)
        print(f"Cl({p},{q}) d={d}: ||A A^-1 - 1|| = {err:.1e}")
