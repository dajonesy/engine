# file: I7_instrumented.py
# started as PublicClifford/I6_instrumented : February 2026
# adapted to 7D inverse examples using Clifford.py

# A fully instrumented version of the 7D inverse algorithm
# for use with home-grown Clifford.py module.

# adaptable constants
import clif.util

VERBOSE = True # when True, full instrumentation is provided
SMALL = 1e-12

Signs236 = [
    1, -1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 
    1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 
    1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 
    1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1,
    1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1,
    1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1,
    1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1,
    1, -1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1]
GISigns = [
    1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1,
    1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
    1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1,
    1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1,
    1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1,
    1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1,
    1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1,
    1, 1, -1, -1, 1, -1, 1, 1, -1]

def el_label( n ):
    s = 'e'
    for i in range(1,8):
        if 0<((1<<(i-1))&n):
            s += str(i)
    return s

def mv_list( Reg ):
    s = f"{Reg[0]:g}" # print the scalar part unconditionally
    for i in range(1,len(Reg)):
        r = Reg[i]
        if - SMALL < r and r < SMALL:
            continue
        if r < 0.0:
            s += ' - '
            m = -r
        else:
            s += ' + '
            m = r
        s += f"({m:g}^{el_label(i)})"
    return s

# we turn the negation function into a tweak function
def K( W, s=1 ): 
    V = W.copy()
    V.Reg[0] *= -s
    V.Reg[127] *= -s
    return V

def I7_instrumented( U ):
#    util.printSparceAccum( U )
    if VERBOSE: print( "\nU = ", mv_list(U.Reg) )
    Up = Signs236 * U
    if VERBOSE: print( "\nUp = Signs236 * U = ", mv_list(Up.Reg) )
    W = U * Up
    if VERBOSE: print( "\nW = U * Up = ", mv_list(W.Reg) )
    I1 = K(  W,  3 ) * W
    if VERBOSE: print( "\nI1 = K(  W,  3 ) * W = ", mv_list(I1.Reg) )
    I0 = K( I1,  1 ) * W
    if VERBOSE: print( "\nI0 = K( I1,  1 ) * W = ", mv_list(I0.Reg) )
    I  = K( I0, 1/3)
    if VERBOSE: print( "\nI  = K( I0, 1/3) = ", mv_list(I.Reg) )
    V = I * W
    if VERBOSE: print( "\nV = I * W = ", mv_list(V.Reg) )
    # IW = V so 1/W = I/V : because this reciprocal is what we care about 
    # it does not matter if I and V contain the common factor -1/3 
    # needed to express the determinant
    # V is all zero except for scalar and pseudoscalar parts
    Vp = GISigns * V # use the grade involution to obtain something like a conjugate in the complex sense
    if VERBOSE: print( "\nVp = GISigns * V = ", mv_list(Vp.Reg) )
    D = Vp * V # is now a scalar
    if VERBOSE: print( "\nD = Vp * V = ", mv_list(D.Reg) )
    d = D.Reg[0]
    # D = V'V so 1/V = V'/D
    # remember 1/W = I/V so 1/W = IV'/D
    rW = (1/d) * I * Vp # this is 1/W
    if VERBOSE: print( "\nrW = (1/d) * I * Vp = 1/W = ", mv_list(rW.Reg) )
    # remember W = UUp so 1/U = Up/W
    if VERBOSE: print( "\nW * rW = ", mv_list((W*rW).Reg) )
    rU = Up * rW
    if VERBOSE: print( "\nrU = Up * rW = 1/U = ", mv_list(rU.Reg) )
    if VERBOSE: print( "\nU * rU = ", mv_list((U*rU).Reg) )
    return rU
