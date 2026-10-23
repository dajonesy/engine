# clif

A research toolkit for numerical computations in Clifford / geometric algebras, 
with an emphasis on **fast multivector inverses**, including a 6D inverse and 
an emerging dimension-independent approach.

> **Status:** research prototype, not yet a packaged release.
> The API is unstable and the test suite is still being built out.
> Star the repo if you want to follow along.

---

## What's here

The package implements Clifford algebra with explicit ordinal indexing of 
multivector components, efficient sign-table-based multiplication, and support 
for arbitrary metric signature and degeneracy. 
It's organized as four cooperating layers:

- `clif.context` — global algebra state (dimension, signature, degeneracy, grade table).
- `clif.sign_table` — builds/caches the multiplication sign table and a Numba-jitted fast
  multiplier for dimensions < 8.
- `clif.multivector` — the `Accum` class, a multivector as a NumPy array of real coefficients indexed by ordinal.
- `clif.util` — construction, printing, grade projection, involutions, and inverse models for specific dimensions.
- `clif.tools` — primitive signtable support functions
 
Inverse algorithms live in `clif.inverse`:

- `clif.inverse.fls` (and `clif.inverse.sparse_fls`) — Faddeev–LeVerrier–Souriau based inverses, with even/odd-graded variants.
- `clif.inverse.euclidean` — inverse for Euclidean (positive-definite) signatures.
- `clif.inverse.newton` — iterative refinement via Newton–Schulz.

## A worked example

```python
from clif.context import Cl, Layout
from clif.multivector import Accum

Cl(1, 1, 1)        # algebra with 1 positive, 1 negative, 1 null basis vector
A = Accum.random()
print(A)

Cl(1, 3)                  # spacetime algebra  Cl(1,3)
Layout([1, -1, -1, -1])   # equivalent, explicit
```

## Documentation

Built locally with Sphinx:

```bash
cd docs && make html
```

Then open `docs/_build/html/index.html` in a browser.

## License

MIT. See [LICENSE](LICENSE).
