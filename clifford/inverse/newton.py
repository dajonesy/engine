# file: clifford/inverse/newton.py
"""
Newton-Schulz iterative inverse refinement for Clifford multivectors.

Given an approximate inverse X0 ≈ A^{-1} (e.g. from an FLS pass), the
Newton-Schulz iteration

    X_{n+1} = X_n * (2·e0 - A * X_n)

converges quadratically to A^{-1} whenever the initial residual
||e0 - A·X_0||_max < 1.  Two or three iterations typically recover
full float64 precision from an FLS result that has lost several digits
at high dimensions.
"""

import numpy as np
import clifford.context as Clif
from clifford.multivector import Accum


def newton_schulz(A: Accum, X0: Accum, tol: float = 1e-13, max_iter: int = 20) -> Accum:
    """Refine an approximate multivector inverse via Newton-Schulz iteration.

    Parameters
    ----------
    A : Accum
        The multivector to invert.
    X0 : Accum
        Initial approximation to A^{-1}.
    tol : float
        Convergence threshold on ||e0 - A·X||_max.
    max_iter : int
        Maximum iterations before returning the best result so far.

    Returns
    -------
    Accum
        Refined inverse, satisfying A * result ≈ e0 within tol.
    """
    mul = Clif._ActiveTable.fast_mul
    a = A.Reg
    x = X0.Reg.copy()

    two = np.zeros_like(a)
    two[0] = 2.0

    ax = mul(a, x)
    for _ in range(max_iter):
        err = abs(ax[0] - 1.0)
        if x.shape[0] > 1:
            err = max(err, float(np.max(np.abs(ax[1:]))))
        if err < tol:
            break
        x  = mul(x, two - ax)
        ax = mul(a, x)

    result = Accum()
    result.Reg = x
    return result
