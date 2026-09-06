# file: clif/inverse/__init__.py
# 2026.05.14 DAJones: created
# 2026.08.17 DAJones: updated sparse_fls and added tau_api

"""
clif.inverse
~~~~~~~~~~~~
Fast, JIT-compiled multivector inverse routines for arbitrary signature.
"""

from .fls        import fls_inverse, even_inverse, odd_inverse, inverse_involution
from .newton     import newton_schulz
from .euclidean  import euclidean_inverse
from .sparse_fls import TauInverse
from .tau_api    import get_inverter, inverse, warm

__all__ = [
    "euclidean_inverse",
    "fls_inverse", "even_inverse", "odd_inverse", "inverse_involution",
    "newton_schulz",
    "TauInverse",
    "get_inverter", "inverse", "warm"
]
