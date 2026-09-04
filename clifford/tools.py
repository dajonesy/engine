# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:41:04 2020

@author: David Jones

This file was created by extracting the functions needed in 
"Clifford Multiplication Demo.ipynb"

Investigating patterns in the Clifford multiplication, we need some utilities
which are not necessary to implement the processing of multivectors.

"""
VERBOSE = False

#==============================================================================

def GradeCount( n ): # assume nBits>=0
    nGrade = 0
    while 0 != n:
        nGrade += 1 & n
        n >>= 1
    return nGrade

# utility to count the bivector inversions required to sort the vectors
def SwapCount( bitsLeft, bitsRight ):
    nSwap=0
#    if 0 != signature:
#        bitsLeft=signatureAdjustment(bitsLeft)
#        bitsRight=signatureAdjustment(bitsRight)
    nJump = GradeCount( bitsRight ) # the number of high order vecs in the right pile for a lower order vec from the left to jump over
    mask = 1
    while 0 != nJump:
        if 0 != (mask & bitsRight): # found low order vec on the right
            nJump -= 1 # leave it on the right
        if 0 != (mask & bitsLeft): # found low order vec on the left
            nSwap = nSwap + nJump # it moves over this many high order vecs on the right
        mask <<= 1
    return nSwap

# subset sign table formation
def sign_table( elements, verify=False ):
    table = []
    for i in range(len(elements)):
        row = []
        el = elements[i]
        for j in range(len(elements)):
            er = elements[j]
            n = SwapCount( el, er )
            row += [[1,-1][1&n]]
            ep = el ^ er
            k = elements.index(ep)
            if verify and k != i ^ j:
                print(k,' != ',i,' ^ ',j)
        table += [row]
    return table
def SignTable(elements,verify=False): return sign_table(elements,verify)

# creating a whole sign table that we are then going to reduce the rows to reference indecies wastes memory
# especially for higher dimensions, it is better to have a function which returns just one sign table row
# elements is a defined subspace (which may be the whole space)
# row indexs into elements
def sign_row( row, elements ):
    Row = []
    r = elements[row]
    for c in elements:
        n = SwapCount( r, c )
        Row += [[1,-1][1&n]]
    return Row
def SignRow( row, elements ): return sign_row( row, elements )

# returns a vector of signs for the commuted product
def sign_col( col, elements ):
    Col = []
    c = elements[col]
    for r in elements:
        n = SwapCount( r, c )
        Col += [[1,-1][1&n]]
    return Col
def SignCol( col, elements ): return sign_col( col, elements )

# given a basis of n elements, return its closure (of length 1<<n)
def sub_algebra( basis ):
    n = len(basis)
    X = []
    for i in range(1<<n):
        x = 0
        for j in range(n):
            k = 1<<j
            if 0 != k&i:
                x ^= basis[j]
        #assert( not x in X )
        X += [x]
    if VERBOSE: print(X)
    return X
def SubAlgebra(basis): return sub_algebra(basis)
        
#==============================================================================

# given X[i][j] return X[j][i]
def transpose(X):
    Y = []
    for i in range(len(X[0])):
        V = []
        for j in range(len(X)):
            V += [X[j][i]]
        Y += [V]
    return Y
def Transpose(X): return transpose(X)

def MatrixDirectProduct( a, b ):
    c = []
    for i in range(len(a)):
        c += [[ a[i][j]*b[i][j] for j in range(len(a[i])) ]]
    return c

PATTERN_TYPE = 1
# we deal with two kinds of tables: signs and binary
# this function will print a nice table in either case
def print_table_pattern( table ):
    for i in range(len(table)):
        line="  "
        for j in range(len(table[i])):
            if PATTERN_TYPE == 1:
                if table[i][j] < 0 :
                    line += ' -'
                if table[i][j] > 0 :
                    line += ' +'
                if table[i][j] == 0 :
                    line += ' 0'
        print(line)
def PrintPattern( table ): return print_table_pattern( table )

# If a function is defined b = function(i,j) for 0 <= i,j < bases
# table the results of integer b: -1 <= b <= 1
def TableFunctional(bases,function):
    table = []
    for i in range(bases):
        row = []
        for j in range(bases):
            row += [function(i,j)]
        table += [row]
    if VERBOSE: PrintPattern(table)
    return table

def index_to_sign_row( n, i ):
    '''return the ith row from the n dimensional reference'''
    r = [1]
    for j in range(n):
        if 0 == ((1<<j) & i):
            a = r.copy() 
        else:
            a = [-k for k in r ]
        r += a
    #if VERBOSE: PrintPattern([r])
    return r

# given some row from the reference sign table
# return its index in the canonical Hadamard matrix
# since the reference is symmetric, it works for columns too
def sign_row_to_index(r):
    N = len(r)>>1
    n = N
    m = 0
    while 0 != n:
        m = m<<1
        if r[n]<0:
            m += 1
        n >>= 1
    return m
def sign_row_to_index_unverified(r): return sign_row_to_index(r)

# return its index in the canonical Hadamard matrix
def SignRowToIndex(r):
    N = len(r)>>1
    n = N
    m = 0
    while 0 != n:
        m = m<<1
        if r[n]<0:
            m += 1
        n >>= 1
    # if VERBOSE: PrintPattern([m])
    # we kind of assumed that r was some sign pattern from a reference Hadamard
    # we do ourselves a benefit to verify that before returning an answer
    s = IndexToSignRow( N, m )
    for i in range(len(r)):
        assert s[i] == r[i], "vector is not Hadamard"
    return m

def sign_row_to_string( row ) -> str:
    s = ''
    for i in row:
        if i == 0: c = ' .'
        if i  > 0: c = ' +'
        if i  < 0: c = ' -'
        s += c
    return  s

def bitstrip( n, v ):
    '''given v[] a vector of integers < 2**n, return a binary matrix whose rows are bit places of v[]'''
    m = []
    for i in range(n):
        b = []
        for j in range(len(v)):
            b += [1 & (v[j]>>i)]
        m += [b]
    if VERBOSE: PrintPattern(m)
    return m

def BinaryMatrixToSignPattern(m):
    v = []
    for i in range(len(m)):
        r = m[i]
        v += [[[1,-1][k] for k in r]]
    if VERBOSE: PrintPattern(v)
    return v      
    
# r = R[i] is a binary reference vector of length 2**n
# R is the symmetric binary reference matrix 
# R is related to the canonical Hadamard matrix reference by substitution [1,-1]->[0,1]
def BinaryReferenceFromIndex( n, i ):
    r = [0]
    for m in range(n):
        if 0 == i&(1<<m): # test bit[m]
            r += r
        else:
            r += [[1,0][r[j]] for j in range(len(r))] # the binary complement of r[]
    return r

# determine the index corresponding to a given binary reference vector
def BinaryReferenceToIndex( r ):
    # we only have to look at the 'vector' indecies to determine which reference 
    n = len(r)>>1
    i = 0
    while 0 != n:
        i <<= 1
        i |= r[n]
        n >>= 1
    # validate that r[] is actually R[i][] by comparing r[] to a reconstruction of it
    n = len(r)
    s = BinaryReferenceFromIndex( n, i )
    for m in range(n):
        if s[m] != r[m]:
            i = -1 # an invalid index
            break
    return i
        
