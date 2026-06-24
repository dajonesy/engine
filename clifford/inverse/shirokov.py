# file: clifford/inverse/shirokov.py
"""
The Faddeev–LeVerrier–Souriau multivector inverse.

This module implements the Faddeev–LeVerrier–Souriau (FLS) algorithm as
applied to Clifford algebras by Shirokov (2011) and noted in Lounesto (2001).

This module implements the algorithm described by Shirokov (2011) and noted in
Lounesto (2001).  It is valid for Clifford algebras of any dimension, though
its cost grows as ``2^{⌈(d+1)/2⌉} - 1`` multivector multiplications.

For dimensions ≤ 6 the Jones inverse (:mod:`clifford.inverse.jones`) is faster.

Algorithm
---------
Given a multivector *U* in ``Cl(p, q, r)`` of dimension *n*, let
``N = 2^{⌈(n+1)/2⌉}``.  Define the sequence::

    U_1 = U
    C_k = (N/k) * scalar_part(U_k)
    U_{k+1} = U * (U_k - C_k)

After ``N - 1`` steps, ``U_N`` is a scalar (if *U* is invertible).  The
inverse is ``(U_{N-1} - C_{N-1}) / scalar_part(U_N)``.

Numba strategy
--------------
Numba's ``@njit`` decorator cannot compile functions that manipulate Python
objects such as :class:`~clifford.multivector.Accum`.  The algorithm is
therefore implemented as two cooperating functions:

* :func:`_shirokov_reg` — a ``@njit``-decorated function that operates
  entirely on raw NumPy arrays, using the captured sign table.
* :func:`shirokov_inverse` — a plain Python wrapper that extracts the
  ``.Reg`` array, calls the jitted core, and wraps the result in a new
  :class:`~clifford.multivector.Accum`.

References
----------
Shirokov, D. S. (2011). Inverse elements in finite-dimensional Clifford algebras.
*Mathematical Notes*, 89(5–6), 872–878.
"""

import numpy as np
from numba import njit

import clifford.context as Clif
from clifford.multivector import Accum


def _make_shirokov_kernel(fast_mul):
    """Factory that closes a Numba kernel over the active sign-table multiplier.

    Parameters
    ----------
    fast_mul : callable
        The Numba-jitted ``(u_reg, v_reg) -> result_reg`` function from the
        active :class:`~clifford.sign_table.SignTable`.

    Returns
    -------
    callable
        A ``@njit``-compiled function
        ``_kernel(u_reg, n) -> adj_reg`` implementing the Shirokov iteration
        on raw NumPy arrays.

    Notes
    -----
    The factory pattern is required because Numba closures can only capture
    other Numba-compiled callables (not Python objects or NumPy arrays
    indirectly referenced through them).  Each call to
    :func:`clifford.context.Initialize` should be followed by a fresh call to
    this factory if the kernel is to be used subsequently.
    """
    @njit
    def _kernel(u_reg: np.ndarray, n: int) -> np.ndarray:
        """Shirokov iteration core (operates on raw coefficient arrays).

        Parameters
        ----------
        u_reg : numpy.ndarray
            Coefficient array of the multivector *U* to invert.
        n : int
            Algebra dimension.

        Returns
        -------
        numpy.ndarray
            Coefficient array of ``adj(U)`` — the adjugate — satisfying
            ``U * adj(U) = scalar * 1``.  Divide by the scalar part to get
            the inverse.

        Raises
        ------
        Exception
            If the scalar denominator is zero (singular multivector).
        """
        N = 1 << ((n + 1) // 2)
        uk = u_reg.copy()
        adj = uk.copy()            # will hold U_k - C_k on the last step
        for k in range(1, N):
            ck = (N / k) * uk[0]
            adj = uk.copy()
            adj[0] -= ck           # adj = U_k - C_k
            uk = fast_mul(u_reg, adj)
        return adj

    return _kernel


# Module-level cached kernel — rebuilt whenever the algebra changes.
_cached_kernel = None
_cached_table_id = None


def _get_kernel():
    """Return (or build) the Shirokov kernel for the current algebra.

    The kernel is cached and reused as long as the active sign table has not
    changed.
    """
    global _cached_kernel, _cached_table_id
    table = Clif._ActiveTable
    if table is None or table.fast_mul is None:
        raise NotImplementedError(
            "shirokov_inverse requires a Numba-compatible algebra "
            "(dimensions < 8 with an active sign table)."
        )
    table_id = id(table)
    if table_id != _cached_table_id:
        _cached_kernel   = _make_shirokov_kernel(table.fast_mul)
        _cached_table_id = table_id
    return _cached_kernel


def shirokov_inverse(U: Accum) -> Accum:
    """Compute the multiplicative inverse of *U* using the Shirokov algorithm.

    Parameters
    ----------
    U : Accum
        A non-singular multivector in the current algebra.

    Returns
    -------
    Accum
        The multiplicative inverse of *U*, satisfying
        ``shirokov_inverse(U) * U ≈ 1``.

    Raises
    ------
    ValueError
        If *U* is singular (its scalar adjugate denominator is zero).
    NotImplementedError
        If the current algebra has dimension ≥ 8 (no fast multiplier).

    Examples
    --------
    ::

        import clifford.context as Clif
        import clifford.util as util
        from clifford.inverse.shirokov import shirokov_inverse

        Clif.Cl(6)
        A = util.random(signature=0, dimensions=6)
        I = shirokov_inverse(A)
        print(I * A)   # should show only the scalar 1.0

    Notes
    -----
    Cost: ``2^{⌈(d+1)/2⌉} - 1`` multivector multiplications.
    For d=6 this is 7; for d=7 it is 15.  For d ≤ 6 prefer
    :func:`clifford.inverse.jones.jones_inverse`.
    """
    kernel = _get_kernel()
    n      = U.dimensions
    N      = 1 << ((n + 1) // 2)

    adj_reg = kernel(U.Reg, n)

    # Recover the scalar denominator: scalar_part(U * adj)
    # We recompute it here in Python because the kernel only returns adj.
    uk_reg  = Clif._ActiveTable.fast_mul(U.Reg, adj_reg)
    denom   = uk_reg[0]

    if abs(denom) < 1e-12:
        raise ValueError(
            "shirokov_inverse: multivector is singular "
            "(scalar adjugate denominator is zero)."
        )

    result = Accum()
    result.Reg = adj_reg * (1.0 / denom)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test / timing harness
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    TIMING_CALLS = 10000
    N_VERIFY     = 200
    TOL          = 1e-6

    # Test one representative signature per dimension.
    # Euclidean (all +1) is the cleanest; Minkowski added for d=4 as a bonus.
    test_cases = [
        (1, [1],                   "Euclidean 1D              [1]"),
        (2, [1, 1],                "Euclidean 2D            [1,1]"),
        (3, [1, 1, 1],             "Euclidean 3D          [1,1,1]"),
        (4, [1, 1, 1, 1],          "Euclidean 4D        [1,1,1,1]"),
        (4, [1,-1,-1,-1],          "Minkowski STA    [1,-1,-1,-1]"),
        (5, [1, 1, 1, 1, 1],       "Euclidean 5D      [1,1,1,1,1]"),
        (6, [1, 1, 1, 1, 1, 1],    "Euclidean 6D    [1,1,1,1,1,1]"),
        (7, [1, 1, 1, 1, 1, 1, 1], "Euclidean 7D  [1,1,1,1,1,1,1]"),
    ]

    print(f"\n{'═'*60}")
    print("  Shirokov–Lounesto inverse  —  timing & correctness")
    print(f"{'═'*60}")
    print(f"  {'Signature':<28} {'ops':>4}  {'JIT ms':>7}  "
          f"{'ns/call':>8}  {'Mcall/s':>7}  {'result':>6}")
    print(f"  {'─'*28} {'─'*4}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}")

    for dim, sig, label in test_cases:

        # ── initialise algebra ────────────────────────────────────────────
        Clif.Layout(sig)              # sets _ActiveTable, fast_mul, etc.
        N_steps = (1 << ((dim + 1) // 2)) - 1   # op count

        # ── JIT compile ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        _cached_kernel   = None      # force rebuild for fresh algebra
        _cached_table_id = None
        warmup = Accum()
        warmup.Reg = np.random.normal(0.0, 1.0, Clif.bases())
        try:
            _ = shirokov_inverse(warmup)
        except ValueError:
            pass
        compile_ms = (time.perf_counter() - t0) * 1e3

        # ── correctness ───────────────────────────────────────────────────
        passed = failed = skipped = 0
        for _ in range(N_VERIFY):
            A = Accum()
            A.random()
            try:
                iA   = shirokov_inverse(A)
                prod = (iA * A).Reg
                err  = abs(prod[0] - 1.0) + np.sum(np.abs(prod[1:]))
                if err < TOL:
                    passed += 1
                else:
                    failed += 1
            except ValueError:
                skipped += 1

        status = "✓ PASS" if failed == 0 else f"✗ {failed} FAIL"

        # ── throughput ────────────────────────────────────────────────────
        A_fixed = Accum()
        A_fixed.Reg = np.random.normal(0.0, 1.0, Clif.bases())
        t0 = time.perf_counter()
        for _ in range(TIMING_CALLS):
            shirokov_inverse(A_fixed)
        elapsed = time.perf_counter() - t0
        ns_per  = elapsed / TIMING_CALLS * 1e9
        mps     = TIMING_CALLS / elapsed / 1e6
        print(f"  {label:<28} {N_steps:>4}  {compile_ms:>7.0f}  "
              f"{ns_per:>7.0f}  {mps:>6.4f}  {status}")

    print(f"\n  {TIMING_CALLS:,} calls per timing run, "
          f"{N_VERIFY} random multivectors verified per signature")
    print(f"  ops = multivector multiplications per inverse call")
    print(f"{'═'*60}")