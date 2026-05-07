# file: clifford/util.py
"""
Convenience functions for constructing and operating on multivectors.

This module provides:

* **Printing** — :func:`print_accum`, :func:`print_sparse`, :func:`print2_accum`,
  :func:`print2_sparse`
* **Construction** — :func:`random`, :func:`grade_restricted_random`
* **Grade operations** — :func:`grade`, :func:`involve`
* **Bit / grade utilities** — :func:`invert_by_bit`, :func:`invert_by_bits`,
  :func:`invert_by_grade`, :func:`test_clear_grade`, :func:`test_equality`
* **Subspace tests** — :func:`closed_test`, :func:`xor_test`,
  :func:`complement_test`, :func:`all_test`
* **Inverse algorithms** (dimension-specific closed forms):
  :func:`I4`, :func:`I5`, :func:`I6`

All functions operate on the algebra currently configured in
:mod:`clifford.context`.

Notes
-----
The original ``util.py`` also contained ``promote()``, ``I4A()``, ``I5O()``,
``I6O()`` (optimised variants using ``sri`` look-up tables), and the ``X``
diagnostic class.  These are preserved in ``archive/util_original.py`` and
will be migrated once the corresponding supporting modules are in place.
"""

from math import sqrt
import numpy as np

import clifford.context as Clif
from clifford.multivector import Accum


# ---------------------------------------------------------------------------
# Format strings
# ---------------------------------------------------------------------------

_FMT1 = '{0:3d}. {0:08b} {1:15.8f}'
_FMT2 = '{0:3d}. {0:08b} {1:15.8f} {2:15.8f}'

#: Threshold for "effectively zero" in print and equality tests.
SMALL: float = 1e-5


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_accum(A: Accum) -> None:
    """Print all coefficients of *A*, one per line.

    Each line shows the ordinal index (decimal and binary) followed by the
    coefficient value.  All ``2^d`` coefficients are printed, including zeros.

    Parameters
    ----------
    A : Accum
        The multivector to print.
    """
    for n in range(Clif.bases()):
        print(_FMT1.format(n, A.Reg[n]))


def print_sparse(A: Accum) -> None:
    """Print only the non-zero coefficients of *A*.

    Coefficients with absolute value ≤ :data:`SMALL` are suppressed.

    Parameters
    ----------
    A : Accum
        The multivector to print.
    """
    for n in range(Clif.bases()):
        if abs(A.Reg[n]) > SMALL:
            print(_FMT1.format(n, A.Reg[n]))


def print2_accum(A: Accum, B: Accum) -> None:
    """Print all coefficients of two multivectors side by side.

    Parameters
    ----------
    A : Accum
        Left multivector.
    B : Accum
        Right multivector.
    """
    for n in range(Clif.bases()):
        print(_FMT2.format(n, A.Reg[n], B.Reg[n]))


def print2_sparse(A: Accum, B: Accum) -> None:
    """Print side-by-side coefficients, suppressing rows where both are small.

    A row is printed only when ``|A.Reg[n]| > SMALL`` **or**
    ``|B.Reg[n]| > SMALL`` — i.e. when at least one of the two values is
    non-negligible.

    Parameters
    ----------
    A : Accum
        Left multivector.
    B : Accum
        Right multivector.

    Notes
    -----
    The original implementation had a parenthesisation error that caused it to
    suppress rows incorrectly.  This version uses the intended logic.
    """
    for n in range(Clif.bases()):
        if abs(A.Reg[n]) > SMALL or abs(B.Reg[n]) > SMALL:
            print(_FMT2.format(n, A.Reg[n], B.Reg[n]))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def random(signature: int, dimensions: int) -> Accum:
    """Construct a random multivector in the given algebra.

    Calls :func:`clifford.context.Initialize` to configure the algebra, then
    fills all ``2^d`` coefficients with independent standard-normal samples.

    Parameters
    ----------
    signature : int
        Signature bitmask (see :func:`clifford.context.Initialize`).
    dimensions : int
        Number of basis vectors.

    Returns
    -------
    Accum
        A random multivector whose coefficients are drawn from N(0, 1).

    Examples
    --------
    ::

        A = random(signature=0, dimensions=6)   # random element of Cl(6,0)
    """
    Clif.Initialize(dimensions, signature)
    A = Accum()
    A.Reg = np.random.normal(0.0, 1.0, Clif.bases())
    return A


def grade_restricted_random(grades: list) -> Accum:
    """Construct a random multivector supported on the given grades.

    Only coefficients of the specified grades are filled; all others remain
    zero.

    Parameters
    ----------
    grades : list of int
        The grades to populate, e.g. ``[1]`` for a random vector,
        ``[0, 2]`` for a random even multivector.

    Returns
    -------
    Accum
        A random grade-restricted multivector.

    Examples
    --------
    ::

        Clif.Cl(4)
        v = grade_restricted_random([1])   # random 4D vector
    """
    A = Accum()
    samples = np.random.normal(size=Clif.bases())
    for i in range(Clif.bases()):
        if Clif.Grade[i] in grades:
            A.Reg[i] = samples[i]
    return A


# ---------------------------------------------------------------------------
# Grade operations
# ---------------------------------------------------------------------------

def grade(S: Accum, grades: list) -> Accum:
    """Return the projection of *S* onto the given grades.

    Parameters
    ----------
    S : Accum
        The multivector to project.
    grades : list of int
        The grades to retain.

    Returns
    -------
    Accum
        A new multivector with all components not in *grades* set to zero.

    Examples
    --------
    Extract the vector (grade-1) part::

        v = grade(A, [1])

    Extract the even sub-algebra::

        even = grade(A, [0, 2, 4, 6])
    """
    T = S.copy()
    for n in range(Clif.bases()):
        if Clif.Grade[n] not in grades:
            T.Reg[n] = 0.0
    return T


def involve(S: Accum, grades: list) -> Accum:
    """Negate the components of *S* belonging to the given grades.

    Used to construct involutions selectively by grade.  For example,
    negating grades {0} implements the scalar reflection used in the Jones
    inverse iteration.

    Parameters
    ----------
    S : Accum
        The multivector to involute.
    grades : list of int
        The grades to negate.

    Returns
    -------
    Accum
        A new multivector with the selected grades negated.
    """
    T = S.copy()
    for n in range(Clif.bases()):
        if Clif.Grade[n] in grades:
            T.Reg[n] = -T.Reg[n]
    return T


def invert_by_grade(B: Accum, grades: list) -> Accum:
    """Alias for :func:`involve` — negate components of *B* in *grades*.

    Parameters
    ----------
    B : Accum
        The multivector to modify.
    grades : list of int
        Grades to negate.

    Returns
    -------
    Accum
        A new multivector.
    """
    return involve(B, grades)


# ---------------------------------------------------------------------------
# Bit-level utilities
# ---------------------------------------------------------------------------

def invert_by_bit(d: int, B: Accum) -> Accum:
    """Negate coefficients of *B* whose ordinal index has bit *d* set.

    Parameters
    ----------
    d : int
        The bit position to test (0-based; bit 0 corresponds to *e1*).
    B : Accum
        Input multivector.

    Returns
    -------
    Accum
        A new multivector.
    """
    A = B.copy()
    for i in range(len(A.Reg)):
        if i & (1 << d):
            A.Reg[i] = -A.Reg[i]
    return A


def invert_by_bits(d: int, B: Accum) -> Accum:
    """Negate coefficients of *B* whose ordinal shares any bit with mask *d*.

    Parameters
    ----------
    d : int
        Bitmask template.
    B : Accum
        Input multivector.

    Returns
    -------
    Accum
        A new multivector.
    """
    A = B.copy()
    for i in range(len(A.Reg)):
        if i & d:
            A.Reg[i] = -A.Reg[i]
    return A


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def test_clear_grade(A: Accum) -> list:
    """Return a Boolean list indicating which grades are zero in *A*.

    Parameters
    ----------
    A : Accum
        The multivector to test.

    Returns
    -------
    list of bool
        ``result[k]`` is ``True`` when the grade-*k* part of *A* is
        effectively zero (all coefficients below :data:`SMALL`).
    """
    V = [True] * (1 + Clif.dimensions)
    for n in range(Clif.bases()):
        if abs(A.Reg[n]) > SMALL:
            V[Clif.Grade[n]] = False
    return V


def test_equality(A: Accum, B: Accum) -> bool:
    """Test whether two multivectors are equal within tolerance.

    Also verifies that the algebra contexts match.

    Parameters
    ----------
    A : Accum
        First multivector.
    B : Accum
        Second multivector.

    Returns
    -------
    bool
        ``True`` if contexts match and all coefficients agree within
        :data:`SMALL`.
    """
    if A.dimensions != B.dimensions:
        return False
    if A.signature != B.signature:
        return False
    for n in range(A.bases):
        if abs(A.Reg[n] - B.Reg[n]) > SMALL:
            return False
    return True


# ---------------------------------------------------------------------------
# Subspace tests
# ---------------------------------------------------------------------------

def closed_test(SS: list) -> bool:
    """Test whether *SS* is closed under XOR (i.e. is a subgroup of ordinals).

    Parameters
    ----------
    SS : list of int
        A list of ordinal indices.

    Returns
    -------
    bool
        ``True`` if ``SS[i] ^ SS[j]`` is in *SS* for all pairs ``i, j``.
    """
    ss_set = set(SS)
    for i in range(len(SS)):
        for j in range(i + 1, len(SS)):
            if (SS[i] ^ SS[j]) not in ss_set:
                return False
    return True


def xor_test(SS: list) -> bool:
    """Test whether *SS* is closed and XOR-consistent as an indexed list.

    Specifically, tests ``SS[i ^ j] == SS[i] ^ SS[j]`` for all ``i, j``.
    This verifies that the list ordering respects XOR structure.

    Parameters
    ----------
    SS : list of int
        A list beginning with 0 whose index structure should mirror XOR.

    Returns
    -------
    bool
    """
    for i in range(len(SS)):
        for j in range(i + 1, len(SS)):
            if SS[i ^ j] != SS[i] ^ SS[j]:
                return False
    return True


def complement_test(SS: list) -> bool:
    """Test whether *SS* is closed under complement within its span.

    Parameters
    ----------
    SS : list of int
        A list of ordinals beginning with 0.

    Returns
    -------
    bool
        ``True`` if ``m ^ SS[i] == SS[n ^ i]`` for all ``i``, where ``m``
        is the OR of all elements and ``n = len(SS) - 1``.
    """
    m = 0
    n = 0
    for i in range(len(SS)):
        m |= SS[i]
        n |= i
    for i in range(len(SS)):
        if m ^ SS[i] != SS[n ^ i]:
            return False
    return True


def all_test(SS: list) -> bool:
    """Run all three subspace tests and return ``True`` if all pass.

    Parameters
    ----------
    SS : list of int
        A candidate subspace expressed as a list of ordinals.

    Returns
    -------
    bool
    """
    return closed_test(SS) and xor_test(SS) and complement_test(SS)


def print_all_tests(SS: list) -> None:
    """Print the result of each subspace test for diagnostic purposes.

    Parameters
    ----------
    SS : list of int
        A candidate subspace.
    """
    print("closed         =", closed_test(SS))
    print("xor test       =", xor_test(SS))
    print("complement test=", complement_test(SS))


# ---------------------------------------------------------------------------
# Dimension-specific closed-form inverses
# (general inverses live in clifford/inverse/)
# ---------------------------------------------------------------------------

def I4(A: Accum) -> Accum:
    """Inverse of a generic 4D multivector.

    Uses a three-step iteration that reduces *A* to a scalar via repeated
    multiplication with its reverse.

    Parameters
    ----------
    A : Accum
        A non-singular multivector in a 4-dimensional algebra.

    Returns
    -------
    Accum
        The multiplicative inverse of *A*, satisfying ``I4(A) * A ≈ 1``.

    Raises
    ------
    ZeroDivisionError
        If *A* is singular (the scalar denominator is effectively zero).

    Notes
    -----
    Algorithm: let ``B = A * ~A`` (grades 0, 1, 4).  Then negate the scalar
    part of *B* to get ``G``, multiply ``G * B`` to obtain a scalar ``D``.
    The inverse is ``~A * G * (1/D)``.
    """
    B = A * ~A
    G = B.copy()
    G.Reg[0] *= -1
    D = G * B
    return ~A * G.scale(1.0 / D.Reg[0])


def I5(A: Accum) -> Accum:
    """Inverse of a generic 5D multivector.

    Parameters
    ----------
    A : Accum
        A non-singular multivector in a 5-dimensional algebra.

    Returns
    -------
    Accum
        The multiplicative inverse of *A*.

    Notes
    -----
    ``B = A * ~A`` has grades {0, 1, 4, 5}.  The algorithm reduces
    *B* to a scalar through two further multiplications with selective
    grade negations.
    """
    B = A * ~A
    C = B * involve(B, [0, 5])
    D = C * involve(C, [0])
    iC = involve(C, [0]).scale(1.0 / D.Reg[0])
    iB = involve(B, [0, 5]) * iC
    return ~A * iB


def I6(A: Accum) -> Accum:
    """Inverse of a generic 6D multivector.

    Parameters
    ----------
    A : Accum
        A non-singular multivector in a 6-dimensional algebra.

    Returns
    -------
    Accum
        The multiplicative inverse of *A*, satisfying ``I6(A) * A ≈ 1``.

    Notes
    -----
    ``B = A * ~A`` is self-reverse and has grades {0, 1, 4, 5}.  Three
    further multiplications and scalar adjustments reduce *B* to a scalar
    denominator.  This is the Jones inverse for 6D — see
    :mod:`clifford.inverse.jones` for the general-dimension version.

    The number of multivector multiplications is 5 (including ``A * ~A``).
    """
    B = A * ~A
    G = B.copy()
    G.Reg[0] *= -3.0
    G *= B
    G.Reg[0] *= -1.0
    G *= B
    G.Reg[0] *= (-1.0 / 3.0)
    I = G.copy()
    G *= B    # G is now a scalar
    return ~A * I.scale(1.0 / G.Reg[0])
