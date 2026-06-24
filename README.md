# clifford

A research toolkit for computations in Clifford / geometric algebras, with an emphasis on
**fast multivector inverses**, including a 6D inverse and an emerging dimension-independent
approach.

> **Status:** research prototype, not yet a packaged release.
> The API is unstable and the test suite is still being built out.
> Star the repo if you want to follow along.

---

## What's here

The package implements Clifford algebra with explicit ordinal indexing of multivector
components, efficient sign-table-based multiplication, and support for arbitrary metric
signature and degeneracy. It's organized as four cooperating layers:

- `clifford.context` — global algebra state (dimension, signature, degeneracy, grade table).
- `clifford.sign_table` — builds/caches the multiplication sign table and a Numba-jitted fast
  multiplier for dimensions < 8.
- `clifford.multivector` — the `Accum` class, a multivector as a NumPy array of real
  coefficients indexed by ordinal.
- `clifford.util` — construction, printing, grade projection, involutions, and inverses for
  specific dimensions.

Inverse algorithms live in `clifford.inverse`:

- `clifford.inverse.jones` — the Jones inverse, valid for dimensions ≤ 6.
- `clifford.inverse.shirokov` — the Shirokov–Lounesto algorithm, valid for any dimension but
  more expensive.
- `clifford.inverse.fls`, `clifford.inverse.sparse_fls`, `clifford.inverse.euclidean`,
  `clifford.inverse.newton` — faster, JIT-compiled inverse routines under active development.

## A worked example

```python
import clifford.context as Clif
from clifford.multivector import Accum
import clifford.util as util

Clif.Cl(3)                     # Euclidean 3D algebra  Cl(3,0)
A = util.random(signature=0, dimensions=3)
print(A)

Clif.Cl(1, 3)                  # spacetime algebra  Cl(1,3)
Clif.Layout([1, -1, -1, -1])   # equivalent, explicit
```

## Documentation

Built locally with Sphinx:

```bash
cd docs && make html
```

Then open `docs/_build/html/index.html` in a browser.

## License

MIT. See [LICENSE](LICENSE).
