# file: clif/inverse/fls.py
r"""Classical Faddeev-LeVerrier-Souriau (FLS) multivector inverse.

This is the textbook method, stated directly in the algebra: the one the
three names belong to, with no involution, no grade pattern, and no
precomputed tables.  It is the reference against which the tau-domain
engine is measured, and it is deliberately unoptimized --- its value is
that it is short enough to read and general enough to run anywhere.

Method
------
Faddeev-LeVerrier builds a characteristic polynomial by alternating a
product with a trace, and reads the inverse off the last adjugate.  Here
the matrix is the complex matrix representation of the algebra, of size
:math:`N = 2^{\lfloor (d+1)/2 \rfloor}`.  The observation that makes it
work inside a Clifford algebra without ever forming a matrix is that the
trace of that representation is :math:`N \langle \cdot \rangle_0`, so the
whole recursion closes in the algebra.

The recursion is

.. math::

    M_1 = A, \qquad a_1 = N \langle M_1 \rangle_0

    M_k = A \, (M_{k-1} - a_{k-1}), \qquad
    a_k = \frac{N}{k} \langle M_k \rangle_0, \qquad k = 2 \dots N

At the end :math:`M_N - a_N` vanishes, which gives

.. math::

    A^{-1} = \frac{M_{N-1} - a_{N-1}}{a_N}, \qquad
    \operatorname{Det}(A) = -a_N \quad (N \ge 2)

:math:`\operatorname{Det}` is the determinant of the :math:`N \times N`
representation, not of the :math:`2^d \times 2^d` map :math:`X \mapsto AX`;
the two are related by :math:`\det(X \mapsto AX) =
\operatorname{Det}(A)^{2^d / N}`.

Cost and scope
--------------
The loop runs :math:`N - 1` geometric products, and :math:`N` doubles with
every two dimensions: 7 products at :math:`d = 6`, 63 at :math:`d = 11`,
255 at :math:`d = 15`.  Nothing in the derivation caps the dimension --- the
method is unbounded, and what runs out first is double precision, since
each product accumulates rounding across a growing coefficient array.

There is deliberately no sparse variant.  The recursion visits the whole
multivector at every step by construction, so there is no sparsity in it
to exploit.

Changes
-------
2026-10-24 --- baseline correction; timings at even :math:`d` change.

* :math:`N` was :math:`2^{\lfloor d/2 \rfloor + 1}`.  That equals
  Shirokov's :math:`2^{\lfloor (d+1)/2 \rfloor}` at odd :math:`d` and is
  twice it at even :math:`d`.  The doubled value amounts to running FLS on
  two stacked copies of the representation: the inverse is still correct,
  but it costs :math:`2N - 1` products instead of :math:`N - 1` (15 vs 7
  at :math:`d = 6`, 31 vs 15 at :math:`d = 8`, 63 vs 31 at :math:`d = 10`),
  carries more rounding, and ends with :math:`a_{2N} =
  -\operatorname{Det}(A)^2`.  Native-FLS timings recorded before this date
  at even :math:`d` are not comparable with later ones; odd :math:`d` is
  unaffected.
* ``fls_determinant`` returned :math:`a_N`; it now returns
  :math:`\operatorname{Det}(A) = -a_N`.  Checked at :math:`d = 2`, where
  :math:`\operatorname{Det}` of a quaternion is its squared norm.

Checked against a dense reference for :math:`d = 2 \dots 8`:
:math:`|\det(X \mapsto AX)| = |\operatorname{Det}(A)|^{2^d/N}` to six
places, and residuals equal or smaller than with the old :math:`N`.

References
----------
Shirokov, D. S. --- On computing the determinant, other characteristic
polynomial coefficients, and inverse in Clifford algebras of arbitrary
dimension.  Computational and Applied Mathematics 40 (2021) 173,
doi:10.1007/s40314-021-01536-0, arXiv:2005.04015.  The normalization
:math:`N = 2^{\lfloor (n+1)/2 \rfloor}`, the recursion, and
:math:`\operatorname{Det} = -a_N` follow the paper.
"""

from clif.multivector import Accum


def _trace_factor(dim: int) -> int:
    r"""Shirokov's :math:`N`: size of the complex matrix representation.

    Dimensions :math:`2m - 1` and :math:`2m` share a value.

    Parameters
    ----------
    dim : int
        Algebra dimension.

    Returns
    -------
    int
        :math:`2^{\lfloor (dim+1)/2 \rfloor}`, the number of iteration
        steps and the factor by which the scalar part is scaled to give
        the trace.
    """
    return 1 << ((dim + 1) // 2)


def fls_inverse(A: Accum) -> Accum:
    r"""Multiplicative inverse by classical Faddeev-LeVerrier-Souriau.

    Runs the full recursion, which costs :math:`N - 1` geometric products
    for :math:`N = 2^{\lfloor (d+1)/2 \rfloor}`.  No dimension limit is
    imposed; accuracy degrades with dimension as rounding accumulates
    across the iteration.

    Parameters
    ----------
    A : Accum
        Multivector to invert, in the algebra currently configured by
        :mod:`clif.context`.

    Returns
    -------
    Accum
        Inverse of ``A``.  ``A`` is not modified.

    Raises
    ------
    ValueError
        The determinant came out zero, so ``A`` has no inverse.  A very
        small determinant is not distinguished from an exact zero here;
        check the residual of the result if that matters.

    Examples
    --------
    ::

        import clif.context as Clif
        from clif.multivector import Accum
        from clif.inverse.fls import fls_inverse

        Clif.Cl(6)
        A = Accum.random()
        iA = fls_inverse(A)
        print(max(abs((A * iA).Reg[1:])))   # residual, near zero

    Notes
    -----
    The scalar ``a`` carried through the loop is :math:`a_k`, and ``adj``
    holds :math:`M_{k-1} - a_{k-1}` from the previous step.  On the final
    pass ``adj`` is therefore the numerator of the inverse and ``a`` is
    :math:`a_N = -\operatorname{Det}(A)`, which is why both are read after
    the loop rather than inside it.
    """
    n_steps = _trace_factor(A.dimensions)

    M = A.copy()                       # M_1
    a = n_steps * M.Reg[0]             # a_1
    adj = M                            # placeholder for N == 1

    for k in range(2, n_steps + 1):
        adj = M.copy()
        adj.Reg[0] -= a                # M_{k-1} - a_{k-1}
        M = A * adj                    # M_k
        a = (n_steps / k) * M.Reg[0]   # a_k

    if a == 0.0:
        raise ValueError("multivector is not invertible: determinant is zero")

    return (1.0 / a) * adj


def fls_determinant(A: Accum) -> float:
    r"""Shirokov determinant :math:`\operatorname{Det}(A) = -a_N`.

    Runs the same recursion as :func:`fls_inverse` and stops at
    :math:`a_N`.  Costs the same :math:`N - 1` products, so there is no
    saving in asking for the determinant alone.  This is the determinant
    of the :math:`N \times N` representation; the determinant of
    :math:`X \mapsto AX` is its :math:`2^d / N` power.

    Parameters
    ----------
    A : Accum
        Multivector, in the algebra currently configured by
        :mod:`clif.context`.

    Returns
    -------
    float
        The determinant.  Zero means ``A`` is not invertible.
    """
    n_steps = _trace_factor(A.dimensions)

    M = A.copy()
    a = n_steps * M.Reg[0]

    for k in range(2, n_steps + 1):
        adj = M.copy()
        adj.Reg[0] -= a
        M = A * adj
        a = (n_steps / k) * M.Reg[0]

    return -a
