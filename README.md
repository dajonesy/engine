# clif

A lightweight Python toolkit for computations in Clifford / geometric
algebras, with an emphasis on **fast inverses in high dimensions**.

> **Status:** research prototype, not yet a packaged release.
> The API is unstable and the test suite is still being built out.
> Star the repo if you want to follow along.

---

## What's here

The current focus is on multivector inversion in 7D Clifford algebras:

- `clif.inverse.self_reverse_7d.sri7_inverse` — inner-kernel inverse for
  multivectors already in the form `W = U · b₂,₃,₆(U)` (supported on
  a 56-slot subspace of the algebra).
- `clif.inverse.self_reverse_7d.i7_inverse` — full Abdulkhaev–Shirokov
  pipeline for inverting a general 7D multivector.  Calls the inner
  kernel under the hood with bookend multiplies before and after.

Both kernels are JIT-compiled with Numba and run in the tens of
microseconds on commodity hardware.

## A worked example

```python
import clif.util as util
import clif.multivector as mv
from clif.inverse.self_reverse_7d import i7_inverse

# 1. Build a random multivector in Cl(7, 0, 0).
A = util.random(signature=0, dimensions=7)

# 2. Invert.  i7_inverse takes the 128-coefficient array and returns
#    the 128-coefficient array of A^{-1}.
iA = mv.Accum()
iA.Reg = i7_inverse(A.Reg)

# 3. Verify: iA * A should be a clean unit scalar.
print(iA * A)
```

## Documentation

Built locally with Sphinx:

```bash
cd docs && make html
```

Then open `docs/_build/html/index.html` in a browser.

## License

MIT.  See [LICENSE](LICENSE).
