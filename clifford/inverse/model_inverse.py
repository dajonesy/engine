# file: model_inverse.py
"""
High level multivector inverse model.

This module is an unoptimized working example of a novel inverse method.
It is restricted to dimensions 4-11.
On import, two tables are created to serve dimensions 4-11:

* :data:`Tweak[d-4]` -- FLS coefficients used to tweak the scalar
  (and for odd dimensions, the pseudoscalar also) in between repeated
  products, forming a multivector polynomial.
* :data:`Involution[d-4]` -- an array of signed units describing the
  involution tau: reversion composed with a Thue-Morse sign layer on the
  top ceil((d-6)/2) basis vectors.

* :func:`mv_inv(A)` -- returns the inverse of multivector A.
"""
import clifford.context as Clif
from clifford.multivector import Accum

invo_4   = [ 1]*16
invo_5   = [ 1]*32
invo_6   = [ 1]*64
invo_7   = [ 1]*64  + [-1]*64
invo_8   = [ 1]*128 + [-1]*128
invo_9   = [ 1]*128 + [-1]*128 + [-1]*128 + [ 1]*128
invo_10  = [ 1]*256 + [-1]*256 + [-1]*256 + [ 1]*256
invo_11  = [ 1]*256 + [-1]*256 + [-1]*256 + [ 1]*256 + [-1]*256 + [ 1]*256 + [ 1]*256 + [-1]*256

involution_fix = (invo_4, invo_5, invo_6, invo_7, invo_8, invo_9, invo_10, invo_11)

Involution = []
for invo in involution_fix:
    Involution.append(
        [(1, 1, -1, -1)[3 & Clif.Grade[n]] * invo[n] for n in range(len(invo))]
    )

tweak_4_5   = [1]
tweak_6_7   = [3, 1, 1/3]
tweak_8_9   = [7, 3, 5/3, 1, 3/5, 1/3, 1/7]
tweak_10_11 = [15, 7, 13/3, 3, 11/5, 5/3, 9/7, 1, 7/9, 3/5, 5/11, 1/3, 3/13, 1/7, 1/15]
Tweak = (tweak_4_5, tweak_4_5, tweak_6_7, tweak_6_7, tweak_8_9, tweak_8_9, tweak_10_11, tweak_10_11)


# T = tweaks, I = involution template, A = multivector
def mv_inv(A: Accum) -> Accum:
    """
    Returns the inverse of the multivector A (when it exists).

    Example
    -------
    import clifford.context as Clif
    Clif.Cl(6)            # register a context
    A = Accum()           # create a multivector with the registered context
    A.random()            # fill the multivector with random values
    rA = mv_inv(A)        # calculate the inverse
    print(A * rA)         # show that rA is the inverse of A (the product is 1.0)

    """

    n = A.dimensions
    if n < 4 or n > 11:
        return None
    odd = 1 == (1 & n)
    if odd:
        PSEUDO = (1 << n) - 1

    I = Involution[n - 4]
    B = I * A
    C = A * B          # C = AB  ->  1/A = B/C

    # ---------
    # Stepping into a specialized optimizable polynomial domain where:
    # # domain closure for polynomials
    # # commutativity of components (because involution is anti-automorphism)

    T = Tweak[n - 4]
    D = C.copy()
    D.Reg[0] *= -T[0]
    if odd:
        D.Reg[PSEUDO] *= -T[0]
    for i in range(1, len(T)):
        D *= C
        D.Reg[0] *= -T[i]
        if odd:
            D.Reg[PSEUDO] *= -T[i]
    E = D * C

    # ---------
    # The polynomial's final product collapses to the center of the algebra:
    # a scalar for even d, a scalar + pseudoscalar for odd d.  This is where
    # the determinant lives.

    if odd:  # E[0, PSEUDO] are the only non-zero elements
        # E = DC  ->  1/C = D/E = E'D / E'E
        Ep = E.copy()
        Ep.Reg[PSEUDO] *= -1
        EpE = Ep * E                       # scalar only
        rC = (1 / EpE.Reg[0]) * Ep * D
    else:    # even E is scalar
        # E = DC  ->  1/C = D/E  ->  1/A = BD/E
        rC = (1 / E.Reg[0]) * D

    rA = B * rC        # 1/A = B/C
    return rA
