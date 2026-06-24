# file: clifford/context.py
"""
Algebra context — global state for the current Clifford algebra.

This module holds the parameters that define the algebra in use: the number of
dimensions, the metric signature (which basis vectors square to −1), and the
degeneracy mask (which basis vectors square to 0).  It also maintains the
pre-computed :data:`Grade` table and the active :class:`~clifford.sign_table.SignTable`.

All multivectors created after a call to :func:`Initialize` (or the convenience
wrappers :func:`Layout` and :func:`Cl`) share the same algebra context.

.. warning::

    The context is **module-level mutable state**.  Two multivectors created
    under different :func:`Initialize` calls are mathematically incompatible.
    Each :class:`~clifford.multivector.Accum` instance records the context at
    the time of its creation; use :func:`Check` to verify compatibility before
    operating on a pair.

Initialisation functions
------------------------
Three entry points are provided for different workflows:

* :func:`Initialize` — specify dimension, signature mask, and degeneracy mask
  directly (low-level, most explicit).
* :func:`Layout` — specify the metric signature as a list of ``{+1, 0, −1}``
  values, one per basis vector.
* :func:`Cl` — specify the algebra as ``Cl(p, q, r)`` — ``p`` positive,
  ``q`` negative, ``r`` degenerate basis vectors.

Examples
--------
Euclidean 3-space::

    import clifford.context as Clif
    Clif.Cl(3)           # Cl(3, 0, 0)

Minkowski spacetime (one time-like dimension)::

    Clif.Cl(1, 3)        # Cl(1, 3, 0)

Projective geometry (one degenerate dimension)::

    Clif.Cl(3, 0, 1)     # Cl(3, 0, 1)

Explicit per-axis metric::

    Clif.Layout([1, 1, -1, 0])
"""

import numpy as np
import clifford.sign_table as ST


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

#: Number of basis vectors in the current algebra.
dimensions: int = 0

#: Bitmask selecting the basis vectors that square to −1.
#: Bit *k* is set when basis vector *e*\ (k+1) has negative signature.
signature: int = 0

#: Bitmask selecting the basis vectors that square to 0 (degenerate/null).
#: Bit *k* is set when basis vector *e*\ (k+1) is degenerate.
degenerate: int = 0

#: Grade table: ``Grade[i]`` is the number of set bits in ``i``, i.e. the
#: grade of the blade whose ordinal index is ``i``.
#: Recomputed by :func:`Initialize`.
Grade: np.ndarray = []

#: The active :class:`~clifford.sign_table.SignTable` for the current algebra.
#: Set by :func:`Initialize`; ``None`` until the first call.
_ActiveTable = None


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def bases() -> int:
    """Return the number of basis blades in the current algebra.

    Equal to ``2 ** dimensions``.

    Returns
    -------
    int
        The number of basis blades.

    Examples
    --------
    ::

        Clif.Cl(3)
        assert Clif.bases() == 8   # scalar + 3 vectors + 3 bivectors + 1 trivector
    """
    return 1 << dimensions


def mask() -> int:
    """Return a bitmask covering all valid ordinal indices.

    Equal to ``bases() - 1``.

    Returns
    -------
    int
        Bitmask with the lowest ``dimensions`` bits set.
    """
    return (1 << dimensions) - 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_grade_table( dimensions: int ) -> np.ndarray:
    """Recompute the :data:`Grade` table for the current :data:`dimensions`.

    ``Grade[i]`` equals the number of 1-bits (population count) in ``i``,
    which is the grade of the blade with ordinal index ``i``.

    This function modifies the module global :data:`Grade` in place.
    """

    table = np.empty( 1<<dimensions, dtype=np.uint8 )
    table[0] = 0
    for d in range( dimensions ):
        m = 1<<d
        for n in range( m ):
            table[m + n] = 1 + table[n]
    return table


# ---------------------------------------------------------------------------
# Initialisation entry points
# ---------------------------------------------------------------------------

def Initialize(d: int, s: int = 0, x: int = 0) -> None:
    """Configure the algebra by specifying dimension and metric masks.

    This is the lowest-level initialisation entry point.  :func:`Layout` and
    :func:`Cl` are more convenient for most uses.

    Parameters
    ----------
    d : int
        Number of basis vectors (dimensions of the vector space).
    s : int, optional
        Signature mask.  Bit *k* is set when basis vector *e*\\ (k+1) squares
        to −1.  Default ``0`` gives a fully Euclidean algebra.
    x : int, optional
        Degeneracy mask.  Bit *k* is set when basis vector *e*\\ (k+1) squares
        to 0.  Default ``0`` gives a non-degenerate algebra.

    Notes
    -----
    Calling this function rebuilds the grade table and instantiates a new
    :class:`~clifford.sign_table.SignTable`.  For dimensions ≥ 8 the sign
    table is not pre-built (the Numba multiplier is not generated); a
    different multiplication strategy is required for those cases.

    Examples
    --------
    Euclidean 6-space::

        Initialize(6)

    Minkowski space (negative first axis)::

        Initialize(4, s=0b0001)

    Degenerate axis (projective)::

        Initialize(4, x=0b1000)
    """
    global dimensions, signature, degenerate, Grade, _ActiveTable
    dimensions = d
    signature  = s
    degenerate = x
    Grade = _build_grade_table( d )
    _ActiveTable = ST.SignTable(d)
    return


def Layout(Signature: list) -> None:
    """Configure the algebra from a per-axis metric list.

    Parameters
    ----------
    Signature : list of int
        A list of ``+1``, ``0``, or ``-1`` values, one per basis vector.
        The first entry describes *e1*, the second *e2*, and so on.

    Examples
    --------
    Euclidean 3D::

        Layout([1, 1, 1])

    Minkowski (time-first)::

        Layout([1, -1, -1, -1])

    Projective 3D (degenerate first axis)::

        Layout([0, 1, 1, 1])
    """
    d = len(Signature)
    s = 0
    x = 0
    for w in Signature[::-1]:
        s <<= 1
        x <<= 1
        if w < 0:
            s |= 1
        elif w == 0:
            x |= 1
    Initialize(d, s, x)
    return

def get_signature():
    global dimensions
    global signature
    global degenerate
    layout = []
    for d in range(dimensions):
        mask = 1 << d
        if 0 != degenerate & mask:
            this1 = 0
        elif 0 != signature & mask:
            this1 = -1
        else:
            this1 = 1
        layout += [ this1 ]
    return layout        

def Cl(p: int, q: int = 0, r: int = 0) -> None:
    """Configure the algebra as Cl(p, q, r).

    Constructs the Clifford algebra with ``p`` basis vectors squaring to +1,
    ``q`` squaring to −1, and ``r`` squaring to 0, in that order.

    Parameters
    ----------
    p : int
        Number of positive-signature (Euclidean) basis vectors.
    q : int, optional
        Number of negative-signature basis vectors.  Default 0.
    r : int, optional
        Number of degenerate (null) basis vectors.  Default 0.

    Examples
    --------
    Euclidean 3-space ``Cl(3, 0)``::

        Cl(3)

    Spacetime algebra ``Cl(1, 3)``::

        Cl(1, 3)

    Conformal model of 3D space ``Cl(4, 1)``::

        Cl(4, 1)
    """
    Layout( [1]*p + [-1]*q + [0]*r )
    return


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def Check(A, B) -> None:
    """Assert that two multivectors share the same algebra context.

    Parameters
    ----------
    A, B : :class:`~clifford.multivector.Accum`
        The two multivectors to compare.

    Raises
    ------
    AssertionError
        If the multivectors differ in dimension, signature, or degeneracy.

    Notes
    -----
    This check is *not* called automatically during arithmetic for performance
    reasons.  Call it explicitly during debugging or testing.
    """
    assert A.dimensions == B.dimensions, \
        f"Dimension mismatch: {A.dimensions} vs {B.dimensions}"
    assert A.signature  == B.signature,  \
        f"Signature mismatch: {A.signature:#010b} vs {B.signature:#010b}"
    assert A.degenerate == B.degenerate, \
        f"Degeneracy mismatch: {A.degenerate:#010b} vs {B.degenerate:#010b}"
