# file: bench/kingdon_inverse.py
r"""Multivector inverses written against kingdon's public API, for benchmarking.

Two inverses, written the way a typical kingdon user would write them:
plain ``*`` for the geometric product, ``alg.multivector(values=, keys=)``
to build elements, and no access to kingdon internals.

``shirokov_inverse``
    Classical FLS with Shirokov's :math:`N = 2^{\lfloor (d+1)/2 \rfloor}`:
    :math:`N - 1` full products.  Same recursion as ``clif/inverse/fls.py``.

``tau_fls_inverse``
    Step down to the tau-domain with :math:`H = A\,\tau(A)`, run FLS on
    :math:`H` at half order :math:`N/2`, step up with
    :math:`A^{-1} = \tau(A)\,H^{-1}`.  Costs :math:`N/2 + 1` products.

Why kingdon's own ``inv()`` is not used: in kingdon 3.0.0 it is only
usable for a full multivector at :math:`d = 4`.  At :math:`d = 5` the
symbolic Hitzer codegen did not finish in 300 s; at :math:`d = 6` the
symbolic Shirokov path raises ``AttributeError`` (``RationalPolynomial``
has no ``items``); at :math:`d \ge 7` (kingdon's default "large" mode,
direct computation) it raises ``AttributeError`` (``numpy.float64`` has no
``items``).

How kingdon is told about the zeros
-----------------------------------
kingdon's sparsity is structural: an element carries only the keys it is
built with, and a product is generated for exactly those input keys.  So
the tau-domain is communicated by building :math:`H` and every FLS iterate
on the tau-domain keys only.  What kingdon cannot be told is that the
product of two tau-elements lands back in the tau-domain: its output keys
include every blade the inputs can reach, and the out-of-domain part is
computed and only then found to cancel numerically.  Each iterate is
therefore projected back onto the tau-domain keys, and the cancelled work
is paid for --- the cancellation savings the engine gets are not available
here.

Tau-involution template
-----------------------
:math:`\tau` is reversion composed with :math:`e_i \mapsto -e_i` for the
generators in the twist bitmask.  The twist is the top
:math:`k = \max(0, \lceil (d-6)/2 \rceil)` generators.  The tau-domain is
the :math:`+1` fixed set of :math:`\tau`.
"""

import numpy as np


def _grade(b: int) -> int:
    return bin(b).count("1")


def twist_bitmask(d: int) -> int:
    r"""Bitmask of the top :math:`\max(0, \lceil (d-6)/2 \rceil)` generators."""
    k = max(0, -(-(d - 6) // 2))
    return ((1 << k) - 1) << (d - k)


def tau_sign(b: int, twist: int) -> float:
    r"""Sign of :math:`\tau(e_b)` relative to :math:`e_b`."""
    g = _grade(b)
    s = (g * (g - 1) // 2) + _grade(b & twist)
    return -1.0 if s & 1 else 1.0


class TauFLS:
    """Per-algebra setup for ``tau_fls_inverse`` (tau-domain keys and signs)."""

    def __init__(self, alg):
        self.alg = alg
        d = alg.d
        self.N = 1 << ((d + 1) // 2)
        self.twist = twist_bitmask(d)
        keys = range(1 << d)
        self.sign = {b: tau_sign(b, self.twist) for b in keys}
        self.dom_keys = tuple(b for b in keys if self.sign[b] > 0)

    def tau(self, x):
        k = x.keys()
        s = np.array([self.sign[b] for b in k])
        return self.alg.multivector(values=s * np.asarray(x.values()), keys=k)

    def to_domain(self, x):
        """Keep only the tau-domain coefficients of ``x`` (step down / projection)."""
        v = dict(zip(x.keys(), x.values()))
        vals = np.array([v.get(b, 0.0) for b in self.dom_keys])
        return self.alg.multivector(values=vals, keys=self.dom_keys)

    def inverse(self, A):
        tA = self.tau(A)
        H = self.to_domain(A * tA)                 # step down
        n = self.N // 2
        M = H
        a = n * M.e
        adj = M
        for k in range(2, n + 1):
            adj = M - a                            # M_{k-1} - a_{k-1}, tau-domain keys
            M = self.to_domain(H * adj)            # product leaves the domain; project back
            a = (n / k) * M.e
        if a == 0.0:
            raise ValueError("multivector is not invertible: determinant is zero")
        return tA * (adj / a)                      # step up


def shirokov_inverse(A):
    r"""Classical FLS with Shirokov's :math:`N`; :math:`N - 1` products."""
    N = 1 << ((A.algebra.d + 1) // 2)
    M = A
    a = N * M.e
    adj = M
    for k in range(2, N + 1):
        adj = M - a
        M = A * adj
        a = (N / k) * M.e
    if a == 0.0:
        raise ValueError("multivector is not invertible: determinant is zero")
    return adj / a
