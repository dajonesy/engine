# file: clifford/multivector.py
"""
The :class:`Accum` multivector class.

A multivector in Cl(p, q, r) is a linear combination of all ``2^d`` basis
blades, where ``d = p + q + r``.  :class:`Accum` stores the ``2^d``
real coefficients in a NumPy array :attr:`Accum.Reg` indexed by ordinal
(bitmask) order — see :mod:`clifford.__init__` for the indexing convention.

The name *Accum* (accumulator) reflects the design intent: the object holds
the running sum of blade contributions during a computation, much as a hardware
accumulator register holds the running result of arithmetic operations.

Arithmetic operators
--------------------
The standard Python arithmetic operators are overloaded:

* ``A + B``, ``A + scalar`` — addition
* ``A - B``, ``A - scalar`` — subtraction
* ``scalar * A`` — scalar left-multiplication  (via ``__rmul__``)
* ``A * B`` — geometric product               (via ``__mul__``)
* ``~A``  — reversion                         (via ``__invert__``)

Involutions
-----------
Following Lounesto (2001), three involutions are provided as methods:

* :meth:`reverse` (``~A``) — reverses the order of vectors in each blade;
  negates grades 2 and 3 mod 4.
* :meth:`conjugate` — Clifford conjugation; negates grades 1 and 2 mod 4.
* :meth:`automorph` — grade involution; negates odd grades.

References
----------
Lounesto, P. (2001). *Clifford Algebras and Spinors* (2nd ed.). Cambridge.
"""

import math
import numpy as np
import clifford.context as Clif


class Accum:
    """A real multivector in the current Clifford algebra.

    Parameters
    ----------
    None
        The constructor takes no arguments.  The algebra context is read from
        :mod:`clifford.context` at the time of construction.

    Attributes
    ----------
    Reg : numpy.ndarray
        Coefficient array of shape ``(2**dimensions,)`` and dtype ``float64``.
        ``Reg[i]`` is the coefficient of the basis blade with ordinal index
        ``i``.  Initialised to all zeros.
    dimensions : int
        Algebra dimension recorded at construction time.
    bases : int
        Number of basis blades (``2 ** dimensions``) recorded at construction.
    signature : int
        Signature bitmask recorded at construction time.
    degenerate : int
        Degeneracy bitmask recorded at construction time.

    Examples
    --------
    ::

        import clifford.context as Clif
        from clifford.multivector import Accum

        Clif.Cl(3)
        A = Accum()
        A.Reg[1] = 1.0   # set the e1 component
        A.Reg[2] = 2.0   # set the e2 component
        print(A)         # 001  1.00000000
                         # 010  2.00000000
    """

    #: Threshold below which a coefficient is treated as zero for display.
    SMALL: float = 1e-8

    #: Format string for a single non-zero coefficient line.
    FORMAT: str = '{0:08b} {1:16.8f}\n'

    def __init__(self) -> None:
        self.Reg        = np.zeros(Clif.bases(), dtype=np.float64)
        self.dimensions = Clif.dimensions
        self.bases      = Clif.bases()
        self.signature  = Clif.signature
        self.degenerate = Clif.degenerate

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """Return a sparse string representation showing only non-zero blades.

        Each non-zero coefficient is printed on its own line as::

            <ordinal in binary>  <coefficient>

        Coefficients with absolute value below :attr:`SMALL` are suppressed.
        """
        s = ''
        for i in range(self.bases):
            if abs(self.Reg[i]) >= Accum.SMALL:
                s += Accum.FORMAT.format(i, self.Reg[i])
        return s

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        """Add a multivector or a real scalar.

        Parameters
        ----------
        other : Accum or float or int
            Right-hand operand.  A scalar is added to the grade-0 component.

        Returns
        -------
        Accum
            A new multivector; neither operand is modified.
        """
        if isinstance(other, (int, float)):
            A = self.copy()
            A.Reg[0] += other
        elif isinstance(other, Accum):
            A = Accum()
            for i in range(self.bases):
                A.Reg[i] = self.Reg[i] + other.Reg[i]
        else:
            return NotImplemented
        return A

    def __radd__(self, other):
        """Support ``scalar + multivector``."""
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract a multivector or a real scalar.

        Parameters
        ----------
        other : Accum or float or int
            Right-hand operand.

        Returns
        -------
        Accum
            A new multivector; neither operand is modified.
        """
        if isinstance(other, (int, float)):
            A = self.copy()
            A.Reg[0] -= other
        elif isinstance(other, Accum):
            A = Accum()
            for i in range(self.bases):
                A.Reg[i] = self.Reg[i] - other.Reg[i]
        else:
            return NotImplemented
        return A

    def __rsub__(self,other):
        if isinstance(other, (int, float)):
            A = self.copy()
            for i in range(self.bases):
                A.Reg[i] = -A.Reg[i]
            A.Reg[0] += other
        elif isinstance(other, int[:], float[:]):
            A = Accum()
            for i in range(self.bases):
                A.Reg[i] = other[i] - self.Reg[i]
        else:
            return NotImplemented
        return A
    
    def __rmul__(self, other):
        """Left-multiply by a real scalar: ``scalar * A``.

        Parameters
        ----------
        other : float or int or list or tuple
            Scalar or per-component weight array.

        Returns
        -------
        Accum
            A new multivector; ``self`` is not modified.

        Notes
        -----
        If *other* is a list or tuple it is treated as a per-component weight
        vector (element-wise multiplication).  This is used internally by some
        algorithms and is not part of the standard algebraic interface.
        """
        A = Accum()
        if isinstance(other, (int, float)):
            for i in range(self.bases):
                A.Reg[i] = other * self.Reg[i]
        elif isinstance(other, (list, tuple, np.ndarray)):
            for i in range(self.bases):
                A.Reg[i] = other[i] * self.Reg[i]
        else:
            return NotImplemented
        return A

    def __mul__(self, other):
        """Geometric product ``self * other``.

        Parameters
        ----------
        other : Accum
            Right-hand multivector.

        Returns
        -------
        Accum
            The geometric product; neither operand is modified.

        Raises
        ------
        NotImplementedError
            If the current algebra has dimension ≥ 8 (fast multiplier not yet
            available for those dimensions).
        """
        if not isinstance(other, Accum):
            return NotImplemented
        A = Accum()
        if Clif._ActiveTable and Clif._ActiveTable.fast_mul:
            A.Reg = Clif._ActiveTable.fast_mul(self.Reg, other.Reg)
        else:
            raise NotImplementedError(
                "Dimensions > 8 do not yet have an optimised multiplier. "
                "Contribute one via clifford/sign_table.py."
            )
        return A

    def inverse(self):
        """Returns the multiplicative inverse of the multivector."""
        # Your inversion logic here (e.g., using shirokov's method or reversion)
        pass
        return None

    def __truediv__(self, other):
        # A / B is mathematically A * B.inverse()
        if isinstance(other, Multivector):
            return self * other.inverse()
        # Handle scalar division
        return self * (1 / other)

    def __rtruediv__(self, other):
        # Handle cases like 1 / A
        return other * self.inverse()
    
    # ------------------------------------------------------------------
    # Equality
    # ------------------------------------------------------------------

    def __eq__(self, other) -> bool:
        """Test coefficient-wise equality within tolerance :attr:`SMALL`.

        Returns
        -------
        bool
            ``True`` if all corresponding coefficients agree within
            :attr:`SMALL`.
        """
        if not isinstance(other, Accum):
            return NotImplemented
        for i in range(self.bases):
            if abs(self.Reg[i] - other.Reg[i]) >= Accum.SMALL:
                return False
        return True

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Set all coefficients to zero in place."""
        self.Reg[:] = 0.0

    def copy(self) -> 'Accum':
        """Return a new :class:`Accum` with the same coefficients.

        Returns
        -------
        Accum
            An independent copy; modifying the copy does not affect ``self``.
        """
        other = Accum()
        other.Reg[:] = self.Reg
        return other

    def scale(self, scalar: float) -> 'Accum':
        """Return a new multivector scaled by *scalar*.

        Equivalent to ``scalar * self`` but available as a named method for
        clarity in algorithm code.

        Parameters
        ----------
        scalar : float
            The scaling factor.

        Returns
        -------
        Accum
            A new scaled multivector; ``self`` is not modified.
        """
        other = Accum()
        other.Reg = scalar * self.Reg
        return other

    def mag(self) -> float:
        """Return the Euclidean magnitude of the coefficient vector.

        This is the square root of the sum of squares of all coefficients.
        It is *not* the geometric norm of the multivector (which depends on
        the metric signature), but a useful measure of overall magnitude.

        Returns
        -------
        float
            ``sqrt(sum(Reg[i]**2))``.
        """
        return math.sqrt(float(np.dot(self.Reg, self.Reg)))

    def normalize(self) -> float:
        """Normalise ``self`` in place by its Euclidean coefficient magnitude.

        Divides every coefficient by :meth:`mag` and returns the original
        magnitude.

        Returns
        -------
        float
            The magnitude before normalisation.
        """
        r = self.mag()
        if r > 0.0:
            self.Reg *= (1.0 / r)
        return r

    # ------------------------------------------------------------------
    # Involutions  (Lounesto 2001, §3)
    # ------------------------------------------------------------------

    def __invert__(self) -> 'Accum':
        """Reversion ``~A``.

        Reverses the order of basis vectors in each blade.  Negates
        coefficients whose grade satisfies ``grade mod 4 ∈ {2, 3}``.

        Returns
        -------
        Accum
            The reverse of ``self``; ``self`` is not modified.

        Notes
        -----
        Equivalent to Lounesto's ``Ã`` notation.
        """
        other = self.copy()
        for i in range(self.bases):
            if (Clif.Grade[i] & 3) > 1:   # grade mod 4 in {2, 3}
                other.Reg[i] = -other.Reg[i]
        return other

    def reverse(self) -> 'Accum':
        """Reversion — alias for ``~self``.

        Returns
        -------
        Accum
            The reverse of ``self``.
        """
        return ~self

    def conjugate(self) -> 'Accum':
        """Clifford conjugation.

        Negates coefficients whose grade satisfies ``grade mod 4 ∈ {1, 2}``.

        Returns
        -------
        Accum
            The Clifford conjugate of ``self``; ``self`` is not modified.

        Notes
        -----
        Equivalent to Lounesto's ``A̅`` notation.
        """
        other = self.copy()
        for i in range(self.bases):
            n = Clif.Grade[i] & 3
            if n == 1 or n == 2:
                other.Reg[i] = -other.Reg[i]
        return other

    # ------------------------------------------------------------------
    # Scalar test
    # ------------------------------------------------------------------

    def is_scalar(self) -> bool:
        """Return ``True`` if all non-scalar coefficients are effectively zero.

        Returns
        -------
        bool
            ``True`` when ``|Reg[i]| < SMALL`` for all ``i > 0``.
        """
        for i in range(1, self.bases):
            if abs(self.Reg[i]) >= Accum.SMALL:
                return False
        return True

    def random(self):
        """Return random values

        Returns
        -------
        Accum
            Random values

        Notes
        -----
        With this method, choice of signature and dimension have already been decided.
        We just get some random value.
        Principly used for testing.
        
        """
        self.Reg = np.random.normal(0.0, 1.0, self.bases)
        return
