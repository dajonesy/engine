# file: clif/inverse/model_inverse.py
"""
High level multivector inverse model.

This module is an unoptimized working example of a novel inverse method.
It is primarily for dimensions 4-11, which perform well with double precision.
To see the effect where extended precision becomes important 12D and 13D are included.

On import, two tables are created to serve dimensions 4-13:

* :data:`Tweak[d-4]` -- FLS coefficients used to tweak the scalar
  (and for odd dimensions, the pseudoscalar also) in between repeated
  products, forming a multivector polynomial.
* :data:`Involution[d-4]` -- an array of signed units describing the
  involution tau: reversion composed with a Thue-Morse sign layer on the
  top ceil((d-6)/2) basis vectors.

* :func:`inverse(A)` -- returns the inverse of multivector A.
"""
import clif.context as Clif

invo_4   = [ 1]*16
invo_5   = [ 1]*32
invo_6   = [ 1]*64
invo_7   = [ 1]*64  + [-1]*64
invo_8   = [ 1]*128 + [-1]*128
invo_9   = [ 1]*128 + [-1]*128 + [-1]*128 + [ 1]*128
invo_10  = [ 1]*256 + [-1]*256 + [-1]*256 + [ 1]*256
invo_11  = [ 1]*256 + [-1]*256 + [-1]*256 + [ 1]*256 + [-1]*256 + [ 1]*256 + [ 1]*256 + [-1]*256
invo_12  = [ 1]*512 + [-1]*512 + [-1]*512 + [ 1]*512 + [-1]*512 + [ 1]*512 + [ 1]*512 + [-1]*512
invo_13 = ([ 1]*512 + [-1]*512 + [-1]*512 + [ 1]*512 + [-1]*512 + [ 1]*512 + [ 1]*512 + [-1]*512 +
           [-1]*512 + [ 1]*512 + [ 1]*512 + [-1]*512 + [ 1]*512 + [-1]*512 + [-1]*512 + [ 1]*512)
invo_14 =  ([ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 +
            [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 )
invo_15 =  ([ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 +
            [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 +
            [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 +
            [ 1]*1024 + [-1]*1024 + [-1]*1024 + [ 1]*1024 + [-1]*1024 + [ 1]*1024 + [ 1]*1024 + [-1]*1024 )

involution_fix = (invo_4, invo_5, invo_6, invo_7, 
                  invo_8, invo_9, invo_10, invo_11, 
                  invo_12, invo_13, invo_14, invo_15)

Involution = []
for invo in involution_fix:
    Involution.append(
        [(1, 1, -1, -1)[3 & bin(n).count('1')] * invo[n] for n in range(len(invo))]
    )

tweak_4_5   = [1]
tweak_6_7   = [3, 1, 1/3]
tweak_8_9   = [7, 3, 5/3, 1, 3/5, 1/3, 1/7]
tweak_10_11 = [15, 7, 13/3, 3, 11/5, 5/3, 9/7, 1, 7/9, 3/5, 5/11, 1/3, 3/13, 1/7, 1/15]
tweak_12_13 = [31, 15, 29/3, 7, 27/5, 13/3, 25/7, 3, 23/9, 11/5, 21/11, 5/3, 19/13, 9/7, 17/15, 1, 
               15/17, 7/9, 13/19, 3/5, 11/21, 5/11, 9/23, 1/3, 7/25, 3/13, 5/27, 1/7, 3/29, 1/15, 1/31]
tweak_14_15 = [m / (64-m) for m in range(63,1,-1)]

Tweak = (tweak_4_5, tweak_4_5, tweak_6_7, tweak_6_7, 
         tweak_8_9, tweak_8_9, tweak_10_11, tweak_10_11, 
         tweak_12_13, tweak_12_13, tweak_14_15, tweak_14_15)


# T = tweaks, I = involution template, A = multivector
def inverse( A ):
    """
    Returns the inverse of the multivector A (when it exists).

    Example
    -------
    import clif.context as Clif
    Clif.Cl(6)            # register a context
    A = Accum.random()    # create a multivector with the registered context
    rA = inverse(A)       # calculate the inverse
    print(A * rA)         # show that rA is the inverse of A (the product is 1.0)

    """

    n = A.dimensions
    if n < 4 or n > 15:
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
    # a scalar for even d, a scalar + pseudoscalar for odd d.  
    # This is where the determinant lives.

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
