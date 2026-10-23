# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 18:41:04 2020

@author: David Jones

This file was created by extracting the functions needed in 
"Clifford Multiplication Demo.ipynb"

Investigating patterns in the Clifford multiplication, we need some utilities
which are not necessary to implement the processing of multivectors.

Basis blades are identified by an integer bitmask: bit ``j`` set means the
basis vector :math:`e_{j+1}` is present in the blade, so the blade index and
the blade's vector content are the same integer.  The product of two blades
has bitmask ``a ^ b``, and all of the sign bookkeeping reduces to counting
transpositions (:func:`swap_count`).

The sign table of a full :math:`2^n` algebra is, up to the substitution
:math:`+1 \\mapsto 0,\\ -1 \\mapsto 1`, a Sylvester-Hadamard matrix.  Several
functions here move between the three equivalent representations: sign rows
(:math:`\\pm 1`), binary reference rows (:math:`0/1`), and the integer index
of a row within the canonical matrix.
"""
VERBOSE = False

#==============================================================================

def grade_count( n ): # assume n_bits>=0
    """Grade of the blade with bitmask ``n``.

    Counts the set bits of ``n``, each of which is one basis vector present
    in the blade.

    Parameters
    ----------
    n : int
        Blade bitmask.  Assumed non-negative.

    Returns
    -------
    int
        Number of basis vectors in the blade, i.e. its grade.

    Examples
    --------
    ::

        grade_count(0b1011)   # 3, the blade e1 e2 e4
    """
    n_grade = 0
    while 0 != n:
        n_grade += 1 & n
        n >>= 1
    return n_grade

# utility to count the bivector inversions required to sort the vectors
def swap_count( bits_left, bits_right ):
    """Transpositions needed to merge two blades into sorted order.

    The geometric product of two blades is formed by concatenating their
    basis vectors and then sorting them back into ascending order.  Each
    adjacent swap of distinct basis vectors contributes a factor of
    :math:`-1`, so the parity of the count returned here is the sign of the
    product.

    Every basis vector in the left blade must jump over each higher-order
    basis vector in the right blade.  The loop walks the bit places from low
    to high, keeping ``n_jump`` equal to the number of right-hand vectors
    still above the current position.

    Parameters
    ----------
    bits_left : int
        Bitmask of the left-hand blade.
    bits_right : int
        Bitmask of the right-hand blade.

    Returns
    -------
    int
        Number of adjacent transpositions.  Only its parity is significant;
        the sign of the product is ``[1, -1][1 & swap_count(a, b)]``.

    Notes
    -----
    This is the metric-free part of the sign.  Squares of repeated basis
    vectors, which is where the signature and any degenerate basis vectors
    enter, are handled by the caller.
    """
    n_swap = 0
    n_jump = grade_count( bits_right ) # the number of high order vecs in the right pile for a lower order vec from the left to jump over
    mask = 1
    while 0 != n_jump:
        if 0 != (mask & bits_right): # found low order vec on the right
            n_jump -= 1 # leave it on the right
        if 0 != (mask & bits_left): # found low order vec on the left
            n_swap = n_swap + n_jump # it moves over this many high order vecs on the right
        mask <<= 1
    return n_swap

# subset sign table formation
def sign_table( elements, signature=0, verify=False ):
    """Full sign table over a set of blades.

    Parameters
    ----------
    elements : list of int
        Blade bitmasks.  For a closed subspace this is the output of
        :func:`sub_algebra`; for a whole algebra it is ``range(2**n)``.
    signature : int
        Bit mask marking negative signature by dimension.
        Use -1 for anti-euclidean.
    verify : bool, optional
        If true, check that the product of ``elements[i]`` and
        ``elements[j]`` lands at position ``i ^ j``, printing a line for
        every position where it does not.  This holds when ``elements`` is
        listed in the order :func:`sub_algebra` produces, and fails
        otherwise.

    Returns
    -------
    list of list of int
        Square table of :math:`\\pm 1`, where entry ``[i][j]`` is the sign
        of the product of ``elements[i]`` with ``elements[j]``.

    See Also
    --------
    sign_row : One row, without building the whole table.
    sign_col : One column, i.e. the commuted product.
    """
    table = []
    for i in range(len(elements)):
        row = []
        el = elements[i]
        for j in range(len(elements)):
            er = elements[j]
            n = swap_count( el, er )
            n += grade_count( signature & el & er )
            row += [[1,-1][1&n]]
            ep = el ^ er
            k = elements.index(ep)
            if verify and k != i ^ j:
                print(k,' != ',i,' ^ ',j)
        table += [row]
    return table

# creating a whole sign table that we are then going to reduce the rows to reference indecies wastes memory
# especially for higher dimensions, it is better to have a function which returns just one sign table row
# elements in a defined subspace (which may be the whole space under a different basis)
# row indexs into elements
def sign_row( row, elements ):
    """One row of the sign table.

    Building a whole table only to reduce its rows to reference indices
    wastes memory in higher dimensions; this returns a single row.

    Parameters
    ----------
    row : int
        Position within ``elements`` of the left-hand blade.
    elements : list of int
        Blade bitmasks defining the subspace, which may be the whole space.

    Returns
    -------
    list of int
        Signs of ``elements[row]`` multiplied by each blade in turn.
    """
    signs = []
    r = elements[row]
    for c in elements:
        n = swap_count( r, c )
        signs += [[1,-1][1&n]]
    return signs

# returns a vector of signs for the commuted product
def sign_col( col, elements ):
    """One column of the sign table, i.e. the commuted product.

    Where :func:`sign_row` holds the left operand fixed, this holds the
    right operand fixed.

    Parameters
    ----------
    col : int
        Position within ``elements`` of the right-hand blade.
    elements : list of int
        Blade bitmasks defining the subspace.

    Returns
    -------
    list of int
        Signs of each blade in turn multiplied by ``elements[col]``.
    """
    signs = []
    c = elements[col]
    for r in elements:
        n = swap_count( r, c )
        signs += [[1,-1][1&n]]
    return signs

# given a basis of n elements, return its closure (of length 1<<n)
def sub_algebra( basis ):
    """Closure of a generating set of blades under the geometric product.

    Blade bitmasks compose by exclusive-or, so the closure of ``n``
    generators is the set of exclusive-ors of every subset of them.  The
    result is listed so that position ``i`` holds the product of exactly
    those generators whose bit is set in ``i``, which is the ordering
    :func:`sign_table` assumes.

    Parameters
    ----------
    basis : list of int
        Generating blade bitmasks.

    Returns
    -------
    list of int
        The closure, of length ``1 << len(basis)``.

    Notes
    -----
    Nothing checks that the generators are independent.  If two subsets
    produce the same blade the result contains duplicates and is not a
    basis; the commented-out assertion marks where that test would go.
    """
    n = len(basis)
    closure = []
    for i in range(1<<n):
        x = 0
        for j in range(n):
            k = 1<<j
            if 0 != k&i:
                x ^= basis[j]
        #assert( not x in closure )
        closure += [x]
    if VERBOSE: print(closure)
    return closure
        
#==============================================================================

# given x[i][j] return y[j][i]
def transpose(x):
    """Transpose a rectangular list of lists.

    Parameters
    ----------
    x : list of list
        Rows of equal length.

    Returns
    -------
    list of list
        ``y`` such that ``y[i][j] == x[j][i]``.
    """
    y = []
    for i in range(len(x[0])):
        v = []
        for j in range(len(x)):
            v += [x[j][i]]
        y += [v]
    return y

def matrix_direct_product( a, b ):
    """Entrywise product of two equally shaped tables.

    Parameters
    ----------
    a : list of list of int
        Left table.
    b : list of list of int
        Right table, of the same shape as ``a``.

    Returns
    -------
    list of list of int
        Table ``c`` with ``c[i][j] == a[i][j] * b[i][j]``.

    Notes
    -----
    Despite the name this is the entrywise (Hadamard) product, not the
    Kronecker product.  It is the operation that composes two sign patterns,
    for instance applying a signature pattern to a metric-free sign table.
    """
    c = []
    for i in range(len(a)):
        c += [[ a[i][j]*b[i][j] for j in range(len(a[i])) ]]
    return c

#: Selects the rendering used by :func:`print_table_pattern`.
PATTERN_TYPE = 1
# we deal with two kinds of tables: signs and binary
# this function will print a nice table in either case
def print_table_pattern( table ):
    """Print a table of signs as a grid of ``+``, ``-`` and ``0``.

    Reading a sign table as a picture makes its block structure visible in a
    way a table of numbers does not.

    Parameters
    ----------
    table : list of list of int
        Entries are compared against zero, so any signed numeric table will
        do.

    Returns
    -------
    None
        Output goes to standard output.
    """
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

# If a function is defined b = function(i,j) for 0 <= i,j < bases
# table the results of integer b: -1 <= b <= 1
def table_functional(bases, function):
    """Tabulate a two-argument function over a square index range.

    Parameters
    ----------
    bases : int
        Side length of the table; indices run over ``range(bases)``.
    function : callable
        Called as ``function(i, j)``.  For the result to print sensibly
        under :func:`print_table_pattern` it should return ``-1``, ``0`` or
        ``1``.

    Returns
    -------
    list of list
        Table with entry ``[i][j]`` equal to ``function(i, j)``.
    """
    table = []
    for i in range(bases):
        row = []
        for j in range(bases):
            row += [function(i,j)]
        table += [row]
    if VERBOSE: print_table_pattern(table)
    return table

def index_to_sign_row( n, i ):
    '''return the ith row from the n dimensional reference

    Builds row ``i`` of the :math:`2^n` Sylvester-Hadamard matrix directly,
    by doubling: at each step the row so far is followed either by a copy of
    itself or by its negation, according to bit ``j`` of ``i``.

    Parameters
    ----------
    n : int
        Dimension; the row returned has length ``2**n``.
    i : int
        Which row, in the range ``0`` to ``2**n - 1``.

    Returns
    -------
    list of int
        Row of :math:`\\pm 1` values.

    See Also
    --------
    sign_row_to_index : The inverse map.
    binary_reference_from_index : The same row in 0/1 form.
    '''
    r = [1]
    for j in range(n):
        if 0 == ((1<<j) & i):
            a = r.copy() 
        else:
            a = [-k for k in r ]
        r += a
    #if VERBOSE: print_table_pattern([r])
    return r

# given some row from the reference sign table
# return its index in the canonical Hadamard matrix
# since the reference is symmetric, it works for columns too
def sign_row_to_index(r):
    """Index of a sign row within the canonical Hadamard matrix.

    Only the entries at power-of-two positions are consulted: those are the
    vector positions, and their signs spell out the index in binary.  The
    result is not checked against a reconstruction of the row, so a vector
    that is not a Hadamard row yields a meaningless index rather than an
    error.

    Because the reference matrix is symmetric this works for columns too.

    Parameters
    ----------
    r : list of int
        Sign row of length ``2**n``.

    Returns
    -------
    int
        Row index.

    See Also
    --------
    sign_row_to_index_verified : Same map, with the row checked.
    """
    half = len(r)>>1
    n = half
    m = 0
    while 0 != n:
        m = m<<1
        if r[n]<0:
            m += 1
        n >>= 1
    return m

# return its index in the canonical Hadamard matrix
def sign_row_to_index_verified(r):
    """Index of a sign row, with the row verified.

    As :func:`sign_row_to_index`, but the candidate index is used to rebuild
    the row, which is then compared entry by entry against the input.

    Parameters
    ----------
    r : list of int
        Sign row of length ``2**n``.

    Returns
    -------
    int
        Row index.

    Raises
    ------
    AssertionError
        The input is not a row of the canonical Hadamard matrix.
    """
    half = len(r)>>1
    n = half
    m = 0
    while 0 != n:
        m = m<<1
        if r[n]<0:
            m += 1
        n >>= 1
    # if VERBOSE: print_table_pattern([m])
    # we kind of assumed that r was some sign pattern from a reference Hadamard
    # we do ourselves a benefit to verify that before returning an answer
    s = index_to_sign_row( half, m )
    for i in range(len(r)):
        assert s[i] == r[i], "vector is not Hadamard"
    return m

def sign_row_to_string( row ) -> str:
    """Render one row of signs as a string.

    Parameters
    ----------
    row : list of int
        Entries are compared against zero.

    Returns
    -------
    str
        Two characters per entry: ``' +'``, ``' -'`` or ``' .'`` for zero.
    """
    s = ''
    for i in row:
        if i == 0: c = ' .'
        if i  > 0: c = ' +'
        if i  < 0: c = ' -'
        s += c
    return  s

def bitstrip( n, v ):
    '''given v[] a vector of integers < 2**n, return a binary matrix whose rows are bit places of v[]

    Row ``i`` of the result collects bit ``i`` of every entry of ``v``, so
    the matrix is the transpose of the obvious bit layout.  This is how a
    list of blade bitmasks is turned into the binary rows that
    :func:`binary_matrix_to_sign_pattern` converts to sign patterns.

    Parameters
    ----------
    n : int
        Number of bit places to extract.
    v : list of int
        Values, each less than ``2**n``.

    Returns
    -------
    list of list of int
        Matrix of ``n`` rows by ``len(v)`` columns of 0 and 1.
    '''
    m = []
    for i in range(n):
        b = []
        for j in range(len(v)):
            b += [1 & (v[j]>>i)]
        m += [b]
    if VERBOSE: print_table_pattern(m)
    return m

def binary_matrix_to_sign_pattern(m):
    """Substitute :math:`0 \\mapsto +1` and :math:`1 \\mapsto -1` throughout.

    Parameters
    ----------
    m : list of list of int
        Matrix of 0 and 1.

    Returns
    -------
    list of list of int
        The same shape, with entries :math:`\\pm 1`.
    """
    v = []
    for i in range(len(m)):
        r = m[i]
        v += [[[1,-1][k] for k in r]]
    if VERBOSE: print_table_pattern(v)
    return v      
    
# r = reference[i] is a binary reference vector of length 2**n
# reference is the symmetric binary reference matrix 
# reference is related to the canonical Hadamard matrix by substitution [1,-1]->[0,1]
def binary_reference_from_index( n, i ):
    """Row ``i`` of the binary reference matrix, in 0/1 form.

    The binary reference matrix is the canonical Hadamard matrix under the
    substitution :math:`+1 \\mapsto 0,\\ -1 \\mapsto 1`, and like it is
    symmetric.  The row is built by doubling: at each step the row so far is
    followed either by a copy of itself or by its complement.

    Parameters
    ----------
    n : int
        Dimension; the row returned has length ``2**n``.
    i : int
        Which row, in the range ``0`` to ``2**n - 1``.

    Returns
    -------
    list of int
        Row of 0 and 1.

    See Also
    --------
    index_to_sign_row : The same row in :math:`\\pm 1` form.
    """
    r = [0]
    for m in range(n):
        if 0 == i&(1<<m): # test bit[m]
            r += r
        else:
            r += [[1,0][r[j]] for j in range(len(r))] # the binary complement of r[]
    return r

# determine the index corresponding to a given binary reference vector
def binary_reference_to_index( r ):
    """Index of a binary reference row, with the row verified.

    Only the vector positions, those at powers of two, are needed to read
    off the index.  The candidate is then used to rebuild the row for
    comparison against the input.

    Parameters
    ----------
    r : list of int
        Binary reference row of length ``2**n``.

    Returns
    -------
    int
        Row index, or ``-1`` if the input is not a row of the reference
        matrix.
    """
    # we only have to look at the 'vector' indecies to determine which reference 
    n = len(r)>>1
    i = 0
    while 0 != n:
        i <<= 1
        i |= r[n]
        n >>= 1
    # validate that r[] is actually reference[i][] by comparing r[] to a reconstruction of it
    n = len(r).bit_length() - 1 # dimension: len(r) is 2**n
    s = binary_reference_from_index( n, i )
    for m in range(len(r)):
        if s[m] != r[m]:
            i = -1 # an invalid index
            break
    return i
        
