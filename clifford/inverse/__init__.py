# file: clifford.inverse.__init__.py
# 2026.05.14 DAJones: created

"""
clifford.inverse
~~~~~~~~~~~~~~~~
Fast, JIT-compiled multivector inverse routines for arbitrary signature.
"""

from .fls        import fls_inverse, even_inverse, odd_inverse, inverse_involution
from .newton     import newton_schulz
from .sparse_fls import sparse_fls_inverse, sparse_even_inverse, sparse_odd_inverse
from .euclidean  import euclidean_inverse

__all__ = [
    "euclidean_inverse",
    "fls_inverse", "even_inverse", "odd_inverse", "inverse_involution",
    "newton_schulz",
    "sparse_fls_inverse", "sparse_even_inverse", "sparse_odd_inverse",
]