"""
fls_eigen.py
============

.. note::

    **Unverified artifact.** This module uses the *pygae* ``clifford`` package
    (``from clifford import Cl``) rather than ``clif``.  The eigendecomposition
    work is unfinished and has not been adapted to ``clif``'s Accum API.
    It is kept as a reference for the spectral-projector algebra.  Do not import
    it as part of the ``clif`` package.



Faddeev-LeVerrier-Shirokov (FLS) for multivector inverses in Cl(p,q),
extended to expose the *eigenstructure* that the recursion already computes
for free.

The core observation
--------------------
FLS is the Faddeev-LeVerrier algorithm with the substitution

        trace( gamma(M) )  ->  N * <M>_0          (N = 2**ceil(n/2))

where gamma is a faithful matrix representation, <.>_0 is the scalar
(grade-0) part, and N is the size of the relevant (reduced) representation.
Because it *is* Faddeev-LeVerrier, the scalars c_1..c_N it produces along the
way are exactly the coefficients of the characteristic polynomial of gamma(U):

        chi(L) = L**N - c_1 L**(N-1) - c_2 L**(N-2) - ... - c_N

So the spectrum is one np.roots() call away, the c_k are the elementary
symmetric functions of the eigenvalues (signed), and U satisfies chi(U)=0 as
a genuine multivector identity (verified to machine precision).

The eigenvalues live in C: real ones, plus complex-conjugate pairs whose count
reflects the C-type / H-type structure of the algebra. They are the *reduced*
spectrum (degree N), not the inflated 2**n regular-representation spectrum;
each appears in the regular rep with multiplicity 2**n / N.

Bonus: the spectral projectors (Frobenius covariants) can be built by Lagrange
interpolation *in the geometric product itself*, giving the spectral
decomposition U = sum_i lambda_i P_i with the P_i as (generally complex)
multivectors -- no matrix representation ever formed.

Requires: numpy, clifford (pygae).
Tested across Cl(2,0), Cl(3,0), Cl(0,3), Cl(1,3), Cl(3,1), Cl(0,6), Cl(4,2):
inverse and Cayley-Hamilton residuals at machine precision.
"""

from math import ceil
from dataclasses import dataclass
import numpy as np


def _scalar(U):
    """Grade-0 (scalar) coefficient of a clifford multivector."""
    return float(U.value[0])


@dataclass
class FLSResult:
    inverse: object          # U^{-1} as a multivector
    adjugate: object         # U_(N-1) - c_{N-1};  U * adjugate = det
    det: float               # c_N  (= (-1)**(N+1) * prod(eigenvalues))
    charpoly: np.ndarray     # monic coeffs [1, -c1, -c2, ..., -cN], high->low
    coeffs: np.ndarray       # the raw FLS scalars [c_1, ..., c_N]
    N: int                   # representation degree = 2**ceil(n/2)

    @property
    def eigenvalues(self):
        return np.roots(self.charpoly)


def fls(U):
    """Run FLS on a clifford multivector U.

    Returns an FLSResult. The recursion is

        U_(1) = U
        c_k   = (N/k) * <U_(k)>_0
        U_(k+1) = U * (U_(k) - c_k)

    with N = 2**ceil(n/2). Then  U * (U_(N-1) - c_{N-1}) = c_N  (a scalar),
    so  U^{-1} = (U_(N-1) - c_{N-1}) / c_N.
    """
    layout = U.layout
    n = layout.dims
    N = 2 ** ceil(n / 2)

    c = np.empty(N + 1)            # c[1..N]; c[0] unused
    Uk = 1.0 * U                   # U_(1)
    adjugate = None
    one = 1.0 + 0 * U

    for k in range(1, N + 1):
        c[k] = (N / k) * _scalar(Uk)
        if k < N:
            t = Uk - c[k]          # U_(k) - c_k
            if k == N - 1:
                adjugate = t       # the adjugate multivector
            Uk = U * t             # U_(k+1)

    det = c[N]
    if N == 1:                     # n = 0, U is a pure scalar
        inverse = one / c[1]
        adjugate = one
    else:
        inverse = adjugate / det

    charpoly = np.concatenate(([1.0], -c[1:]))   # [1, -c1, ..., -cN]
    return FLSResult(inverse=inverse, adjugate=adjugate, det=det,
                     charpoly=charpoly, coeffs=c[1:].copy(), N=N)


def power_sums_from_moments(U, N):
    """Alternative route to the same spectrum: the moments

        m_k = sum_i lambda_i**k = N * <U**k>_0

    feed Newton's identities. Handy as an independent cross-check of fls().
    Returns the array [m_1, ..., m_N].
    """
    m = np.empty(N)
    Uk = 1.0 * U
    for k in range(1, N + 1):
        Uk = Uk if k == 1 else Uk * U
        m[k - 1] = N * _scalar(Uk)
    return m


def spectral_projectors(U, eigenvalues=None, tol=1e-7, verify=True):
    """Spectral projectors (Frobenius covariants) as multivectors.

    For a diagonalizable U with distinct eigenvalues {l_i},

        P_i = prod_{j != i} (U - l_j) / (l_i - l_j)

    satisfies  P_i P_j = delta_ij P_i,  sum_i P_i = 1,  U = sum_i l_i P_i,
    with each P_i a (generally complex) multivector. Nearly-equal eigenvalues
    are clustered to one distinct value first, so repeated-but-diagonalizable
    spectra are handled correctly.

    Returns (distinct_eigenvalues, projectors). If verify=True and U is not
    diagonalizable (nontrivial Jordan structure), raises ValueError -- the
    interpolation identity genuinely fails there, and that failure is itself
    diagnostic.
    """
    if eigenvalues is None:
        eigenvalues = fls(U).eigenvalues

    # cluster into distinct eigenvalues
    distinct = []
    for lam in eigenvalues:
        for d in distinct:
            if abs(lam - d) < tol:
                break
        else:
            distinct.append(complex(lam))
    distinct = np.array(distinct)

    one = (1.0 + 0j) * (1.0 + 0 * U)
    projectors = []
    for i, li in enumerate(distinct):
        Pi = one
        for j, lj in enumerate(distinct):
            if j == i:
                continue
            Pi = Pi * (U - lj) / (li - lj)
        projectors.append(Pi)

    if verify:
        S = sum(projectors[1:], projectors[0])
        recon = sum((distinct[i] * projectors[i] for i in range(1, len(distinct))),
                    distinct[0] * projectors[0])
        e_sum = np.max(np.abs((S - 1).value))
        e_rec = np.max(np.abs((recon - U).value))
        e_idem = max(np.max(np.abs((P * P - P).value)) for P in projectors)
        if max(e_sum, e_rec, e_idem) > 1e-6:
            raise ValueError(
                "U does not appear diagonalizable (sum-of-P, reconstruction, "
                f"or idempotency failed: {e_sum:.1e}, {e_rec:.1e}, {e_idem:.1e}). "
                "Nontrivial Jordan structure needs the generalized eigenprojector "
                "route instead.")
    return distinct, projectors


# --------------------------------------------------------------------------
if __name__ == "__main__":
    from clifford import Cl

    np.random.seed(7)
    layout, blades = Cl(0, 6)                       # n=6, N=8 -- your home turf
    U = layout.MultiVector(value=np.random.randn(layout.gaDims))

    r = fls(U)
    print(f"Cl(0,6):  N = {r.N}")
    print(f"  inverse residual |U U^-1 - 1| = "
          f"{np.max(np.abs((U*r.inverse - (1+0*U)).value)):.2e}")
    # Cayley-Hamilton: chi(U) = 0 in the algebra (Horner in the geometric product)
    P = r.charpoly[0] * (1 + 0 * U)
    for a in r.charpoly[1:]:
        P = U * P + a
    print(f"  Cayley-Hamilton |chi(U)|        = {np.max(np.abs(P.value)):.2e}")
    print(f"  det (c_N)                       = {r.det:.4f}")
    print(f"  elementary symmetric c_1..c_N   = {np.round(r.coeffs, 3)}")
    print("  eigenvalues:")
    for lam in r.eigenvalues:
        print(f"    {lam.real:+.4f} {lam.imag:+.4f}i")

    vals, Ps = spectral_projectors(U, r.eigenvalues)
    print(f"  spectral decomposition into {len(vals)} projectors verified "
          f"(sum P = 1, U = sum lambda_i P_i, P_i^2 = P_i).")
