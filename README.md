# correlatoranalyser

A Python library for statistical analysis of Lattice QCD correlator data. It provides a unified fitting interface over multiple backends and first-class support for resampling-based error estimation (bootstrap and jackknife).

---

## Installation

Requires Python ≥ 3.10. Install via [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/your-org/correlatoranalyser.git
cd correlatoranalyser
poetry install
```

---

## Quick start

```python
import numpy as np
from correlatoranalyser import Data
from correlatoranalyser.fit import fit

# 1. Wrap raw gauge configurations in a bootstrap ensemble
ordinate = Data(resample_type="bst", data=raw, Nresample=NBST)

# 2. Define a model (must be a named, module-level function)
def model_2exp(t, p):
    return p["A0"] * np.exp(-p["E0"] * t) + p["A1"] * np.exp(-p["E1"] * t)

# 3. Fit
fit_result = fit(
    backend                      = "lsqfit",
    abscissa                     = t,
    ordinate                     = ordinate,
    model                        = model_2exp,
    # prior                      = prior,   # optional: supply priors instead of p0
    p0                           = p0,
    central_value_fit            = True,
    central_value_fit_correlated = False,
    resample_fit                 = True,
    resample_fit_correlated      = False,
)

# 4. Inspect the result
print(fit_result)
```

```
FitResult[(1.0, 24.0), Ndata=24, resample:bst]:
  χ²/dof [dof]   = 0.958 [20]
  χ²/<χ²> [<χ²>] = 1.59 [12]
  p-value        = 0.512
  AIC            = -18.7
    A0: 1.505(23)
    E0: 0.3009(21)
    A1: 0.748(19)
    dE1: 0.505(18)
```

Parameter values and uncertainties are available as `Data` objects:

```python
fit_result.params["E0"].mean   # central-value best fit
fit_result.params["E0"].serr   # bootstrap standard error
fit_result.params["E0"].rspl   # full array of per-resample values, shape (NBST,)
```

---

## Backends

All backends are accessed through the single `fit(backend=..., ...)` entry point.

| `backend` string | Algorithm | Best for |
|---|---|---|
| `"lsqfit"` | Levenberg–Marquardt via [lsqfit](https://github.com/gplepage/lsqfit) (P. Lepage) | Production fits; Gaussian and log-normal priors; |
| `"iminuit"` | MIGRAD via [iminuit](https://github.com/scikit-hep/iminuit) | General nonlinear fits; analytic gradient support |
| `"iminuit"` + `linear_params` | MIGRAD with variable projection (Golub–Pereyra) | Models with mixed linear/nonlinear parameters; reduces to nonlinear search space |
| `"linear regression"` | Closed-form weighted least squares | Strictly linear models; no iterative minimiser |
| `"hybrid:adam+iminuit"` | ADAM pre-optimisation → MIGRAD warm start | Highly non-convex or flat χ² landscapes (e.g. multi-exponential fits) where MIGRAD struggles from a cold start; also supports variable projection |
| `"thc"` | Truncated Hankel Correlator Method | Two-point function fits, i.e. data represented by a linear combination of exponential |

### Priors

Gaussian and log-normal priors are supported by the `lsqfit` and `iminuit` backends:

```python
from correlatoranalyser.prior import Prior

prior = {
    "A0": Prior(1.5,           2.0, dist="normal"),
    "E0": Prior(np.log(0.3),   0.5, dist="log-normal"),  # keeps E0 > 0
}
```

---

## Resampling

`Data` objects represent/perform resampleing ensemble. Two resampling schemes are supported.

```python
# Bootstrap
data = Data(resample_type="bst", data=raw, Nresample=500)

# Jackknife (leave-one-out; Nresample is inferred automatically)
data = Data(resample_type="jkn", data=raw)
```

Arithmetic operations between `Data` objects propagate the resample structure automatically, so derived quantities carry correct statistical uncertainties without any extra bookkeeping.

---

## Examples

Ready-to-run scripts are provided in `examples/`:

| File | Backend | Model |
|---|---|---|
| `example_lsqfit_2exp.py` | lsqfit | Two-exponential correlator |
| `example_iminuit_2exp.py` | iminuit | Two-exponential correlator |
| `example_iminuit_variable_projection_2exp.py` | iminuit + varproj | Two-exponential correlator |
| `example_hybrid_2exp.py` | hybrid ADAM+iminuit | Two-exponential correlator |
| `example_hybrid_variable_projection_2exp.py` | hybrid ADAM+iminuit + varproj | Two-exponential correlator |
| `example_linear_regression.py` | linear regression | Linear model |

---

## Running the tests

```bash
poetry run pytest
```

---

## Authors

- Lea Kutsch — [l.kutsch@fz-juelich.de](mailto:l.kutsch@fz-juelich.de)
- Giovanni Pederiva — [g.pederiva@fz-juelich.de](mailto:g.pederiva@fz-juelich.de)
- Marcel Rodekamp — [marcel.rodekamp@ur.de](mailto:marcel.rodekamp@ur.de)
- Emilio Taggi — [e.taggi@fz-juelich.de](mailto:e.taggi@fz-juelich.de)

## Thanks

Special thanks goes out to my collaborators  [Lea Kutsch](https://github.com/Lea-Antonia), [Giovanni Pederiva](https://github.com/GioPede), and [Emilio Taggi](https://github.com/Tag-E) who have put a lot of work into this.
