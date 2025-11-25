# Correlator Analyser

This repository contains a set of convenience functions to perform analysis of (Lattice QCD) correlator data. 


## Features
### Fit results

The heart of the library is the `FitResult` class that computes/collects various fit statistics such as 
- $\chi^2$ `chi2`
- AIC  `AIC` (with and without small sample correction)
- p-value `Q_value`
and allows for resample (bootstrap/jackknife) fits to be added.

Further, a collection of `FitResult` is implemented via the class `FitState`. It conveniently collects the results and allows for AIC model averages over all added `FitResults`.

### Interfaces

We created a simple (primary) interface function 
```python
fit(
    *,
    abscissa: np.ndarray,
    ordinate_est: np.ndarray | None = None,
    ordinate_std: np.ndarray | None = None,
    ordinate_cov: np.ndarray | None = None,
    resample_ordinate_est: np.ndarray | None = None,
    resample_ordinate_std: np.ndarray | None = None,
    resample_ordinate_cov: np.ndarray | None = None,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    resample_fit_resample_prior: bool = True,
    resample_type: str | None = None,
    # args for lsqfit:
    model: Callable | None = None,
    prior: dict | None = None,
    p0: dict | None = None,
    svdcut: float | None = None,
    maxiter: int = 10_000,
)
```
which serves as a "single line" call to execute non-linear fits over central values and resamples.

A simple example to execute a correlated, single exponential, fit over bootstraps can look as simple as
```python
import numpy as np
import correlatoranalyser as ca
import gvar as gv # since lsqfit works on gvar, it is usful to continue to work with it

t = np.arange(t_start,t_end)
C2pt_bst = np.zeros( (N_bst,T), dtype = float )
C2pt_est = np.zeros( (T,) )
C2pt_cov = np.zeros( (T,T) )

# Read bootstraps (and associate standard deviations per bootstrap), as well as central values from file into C2pt_* 
# or construct it o nthe fly

fitResult = fit(
    # Data to fit against
    # provide the axis the model depends on, here euclidean time in lattice units
    abscissa = t,
    # provide the central value data to fit against
    ordinate_est = C2pt_est,
    # provide the covariance between the data points
    ordinate_cov = C2pt_cov,
    # provide the bootstrap data to fit against
    resample_ordinate_est = C2pt_bst,
    # provide the covariance between the data points
    # we do not need to use a frozen covariance for all bootstraps, but usually this is much more stable
    resample_ordinate_cov = C2pt_cov, 
    
    # fit strategy
    # do a fit to the central values themself
    central_value_fit = True,
    # and do it using the provided covariance
    central_value_fit_correlated = True,
    # perform a fit to the resamples (e.g. bootstraps)
    resample_fit = True,
    # and do it using the provided covariance
    resample_fit_correlated = True,
    # if priors are used, one may want to resample a prior for each bootstrap to reduce bias
    # here we don't use priors, so we don't need it
    resample_fit_resample_prior = False,
    # this can be 'bst' for bootstrap, 'jkn' for jackknife, None if only central value fits are desired.
    resample_type = 'bst',
    
    # Model definition and staring values/priors
    # model definition
    model = lambda t,p: p["A0"]*np.exp(-t*p["E0"])  ,
    # provide some start values
    p0 = {
        "A0": 1e-11,
        "E0": 0.3
    },
    # if priors are desired use the following dictionary instead:
    # prior = {
    #     "A0": gv.gvar(A0_prior_mean, A0_prior_sdev), # gaussian prior
    #     "E0": gv.gvar(E0_prior_mean, E0_prior_sdev), # gaussian prior
    # }
)

print(fitResult)

```

Alternatively, one can execute the fit using `lsqfit` (other backends are planned in the future) explicitly and then import it into the fit result.

```python
import numpy as np
import correlatoranalyser as ca
import gvar as gv # since lsqfit works on gvar, it is usful to continue to work with it

t = np.arange(t_start,t_end)
C2pt_est = np.zeros( (T,) )
C2pt_cov = np.zeros( (T,T) )

# Read bootstraps (and associate standard deviations per bootstrap), as well as central values from file into C2pt_* 
# or construct it on the fly

C2pt_gvar = gv.gvar( C2pt_est, C2pt_cov )


# initialize fit result
fit_result = FitResult(
    # start point of the fit interval
    ts=t[0], 
    # end point of the fit interval
    te=t[-1],
    # number of data points
    Ndata=len(t),
    # abscissa used in the fit
    abscissa=t,
    # number of resamples
)

# execute lsqfit; on the central values only here:
nlf = lsqfit.nonlinear_fit(
    data = (t,C2pt_gvar),
    fcn = lambda t,p: p["A0"]*np.exp(-t*p["E0"])  ,
    # provide some start values
    p0 = {
        "A0": 1e-11,
        "E0": 0.3
    },
) 

fit_result.import_from_lsqfit(nlf=nlf)

print(fit_result)

```


An explicit implementation for linear regression including correlated data points is implemented via the `linear_regression` method
```python
linear_regression(     
    *,
    abscissa: np.ndarray,
    ordinate_est: np.ndarray | None = None,
    ordinate_std: np.ndarray | None = None,
    ordinate_cov: np.ndarray | None = None,
    resample_ordinate_est: np.ndarray | None = None,
    resample_ordinate_std: np.ndarray | None = None,
    resample_ordinate_cov: np.ndarray | None = None,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    resample_type: str | None = None,
    has_intercept: bool = True,
    parameter_names: tuple | None = None 
)
```

Here an example would be quite similar to the fit call above except of the slightly changed arguments.

### Serialisation

Serialisation of `FitResult` (and `FitState`) is implemented via h5py, with fit model being pickled using dill.

You can simply call

```python
import h5py as h5 

with h5.File(filename, 'a') as h5f: 
    fitResult.serialize(h5f, node='fit') # node is optional but may be used to destinguish different fits.
```

## Installation 

The code is setup using the `pyproject`. Thus we can simply clone

```sh 
git clone https://github.com/Marcel-Rodekamp/CorrelatorAnalyser.git && cd CorrelatorAnalyser
```

and install via pip

```sh
pip install .
```

## Dependencies 

We heavily utilities

- `numpy` [pip](https://pypi.org/project/numpy/)
- `gvar` [pip](https://pypi.org/project/gvar/)
- `lsqfit` [pip](https://pypi.org/project/lsqfit/)
- `scipy` [pip](https://pypi.org/project/scipy/)
- `dill` [pip](https://pypi.org/project/dill/)

## List of contributions

This interface was used in the following publications:

- P. Sinilkov et.al., Search for Stable States in Two-Body Excitations of the Hubbard Model on the Honeycomb Lattice [doi@PoS](https://doi.org/10.22323/1.466.0075), [arXiv:2502.04015](https://arxiv.org/abs/2502.04015)
- M. Rodekamp et. al.,Single-particle spectrum of doped $C_{20}H_{12}$-perylene [doi@EPJ B](https://doi.org/10.1140/epjb/s10051-024-00859-1), [arXiv:2406.06711](https://arxiv.org/abs/2406.06711)


## Thanks

Special thanks go out to my collaborators  [Lea Kutsch](https://github.com/Lea-Antonia), [Giovanni Pederiva](https://github.com/GioPede), and [Emilio Taggi](https://github.com/Tag-E) who have put a lot of work into this and are actively using it.
