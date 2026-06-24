# file: clifford/inverse/euclidean.py
"""
Top-level Euclidean multivector inverse for Cl(d, 0), d in 6-13.
"""

from clifford.multivector import Accum
from clifford.inverse.sparse_fls import sparse_fls_inverse
from clifford.inverse.newton    import newton_schulz


def euclidean_inverse(A: Accum, refine: bool = True) -> Accum:
    """Inverse of a non-singular multivector in Euclidean Cl(d, 0), d in 6-13.

    Uses the sparse FLS kernel for speed.  At dimensions 12-13 the
    polynomial iteration accumulates enough floating-point error that
    the raw result has residuals of O(10⁻⁵) to O(10⁻³); when
    ``refine=True`` (the default) Newton-Schulz iteration is applied
    automatically to recover machine precision.

    Parameters
    ----------
    A : Accum
        Non-singular multivector.
    refine : bool, optional
        If True (default), apply Newton-Schulz refinement for dim >= 12.
        Set False to return the raw FLS result at any dimension.

    Returns
    -------
    Accum
        Multiplicative inverse of A.

    Raises
    ------
    ValueError
        If dimension is outside 6-13.

    Notes
    -----
    Typical residuals (max |A · A⁻¹ − e₀| over all blades):

    ======  ============  ===============
    dim     FLS only      FLS + NS
    ======  ============  ===============
    6       1.4 × 10⁻¹⁴  —
    7       1.7 × 10⁻¹⁵  —
    8       2.4 × 10⁻¹⁴  —
    9       1.8 × 10⁻¹²  —
    10      3.6 × 10⁻¹⁰  —
    11      2.9 × 10⁻¹⁰  —
    12      9.4 × 10⁻⁶   9.0 × 10⁻¹⁶
    13      4.2 × 10⁻³   1.8 × 10⁻¹⁵
    ======  ============  ===============
    """
    X = sparse_fls_inverse(A)
    if refine and A.dimensions >= 12:
        X = newton_schulz(A, X)
    return X
