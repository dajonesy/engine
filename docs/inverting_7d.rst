Inverting a 7D multivector
==========================

The :mod:`clifford.inverse.self_reverse_7d` module gives you two entry
points:

* :func:`~clifford.inverse.self_reverse_7d.i7_inverse` — inverts a
  **general** 7D multivector ``U`` via the Abdulkhaev–Shirokov pipeline.
  This is the function you want 99% of the time.

* :func:`~clifford.inverse.self_reverse_7d.sri7_inverse` — inverts only
  a multivector that is already in the form ``W = U · b_{2,3,6}(U)``,
  i.e., already supported on the 56-slot M7 subspace.  This is the
  *inner kernel* that ``i7_inverse`` calls under the hood.  Use it
  directly only if you've built ``W`` yourself.


Quick start
-----------

Inverting a random multivector and verifying the result::

   import clifford.util as util
   import clifford.multivector as mv
   from clifford.inverse.self_reverse_7d import i7_inverse

   # 1. Build a random multivector in Cl(7, 0, 0).
   A = util.random(signature=0, dimensions=7)

   # 2. Invert.  i7_inverse takes the 128-coefficient array (A.Reg)
   #    and returns the 128-coefficient array of A^{-1}.
   iA = mv.Accum()
   iA.Reg = i7_inverse(A.Reg)

   # 3. Verify: iA * A should print a clean unit scalar
   #    (1.0 in slot 0, ~1e-15 elsewhere — that's machine precision).
   print(iA * A)


Timing
------

The kernel JITs on first call.  Discard that first call when timing::

   import time
   from clifford.inverse.self_reverse_7d import i7_inverse

   A = util.random(signature=0, dimensions=7)
   _ = i7_inverse(A.Reg)            # warm up the Numba JIT

   N = 1000
   t0 = time.perf_counter()
   for _ in range(N):
       _ = i7_inverse(A.Reg)
   elapsed = (time.perf_counter() - t0) / N * 1e6
   print(f"{elapsed:.2f} µs/call")


Changing the signature
----------------------

The module bootstraps for ``Cl(7, 0, 0)``.  Other 7D signatures are
loaded via :func:`~clifford.inverse.self_reverse_7d.reinit`, which takes
a 128×128 ``{-1, 0, +1}`` sign table for the chosen ``Cl(p, q, r)`` and
rebuilds the cached kernels::

   import numpy as np
   from clifford.inverse.self_reverse_7d import reinit

   # Construct (or load) the sign table for your target signature.
   my_table = build_my_sign_table()       # shape (128, 128), dtype int8

   reinit(my_table)
   # All subsequent calls to i7_inverse use the new signature.


Handling singular inputs
------------------------

If ``U`` is singular under the A&S pipeline (its scalar denominator
vanishes), :func:`i7_inverse` raises :class:`ValueError`::

   from clifford.inverse.self_reverse_7d import i7_inverse

   try:
       iA = i7_inverse(A.Reg)
   except ValueError as exc:
       print(f"Singular: {exc}")
