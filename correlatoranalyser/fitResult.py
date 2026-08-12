from __future__ import annotations

import inspect
import textwrap
import warnings
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
from scipy.special import gammaincc
import gvar as gv

from .data import Data
from .prior import Prior

from .fit_helper import _jacobian_fd


# =============================================================================
# Model serialisation helpers
# =============================================================================

def _serialise_fcn(fcn: Callable) -> bytes:
    """
    Serialise a model function as its UTF-8 source code.

    Using ``dill`` to pickle closures produces memory-layout-dependent
    byte strings that break when a file is loaded on a different machine
    or Python version.  Storing the raw source text is portable and
    human-readable.  The function is reconstructed with ``exec`` on load
    (see ``_deserialise_fcn``).

    Raises ``ValueError`` if the source cannot be retrieved (e.g. for
    lambda functions defined at the REPL).
    """
    try:
        if hasattr(fcn, '__class__') and not inspect.isclass(fcn) and not inspect.isroutine(fcn):
            src = inspect.getsource(fcn.__class__)
        else:
            src = inspect.getsource(fcn)
    except (OSError, TypeError) as exc:

        raise ValueError(
            f"Cannot serialise model function '{getattr(fcn, '__name__', fcn)}': "
            "source code is not available.  Define the model in a .py file "
            "so that inspect.getsource() can retrieve it."
        ) from exc

    # Dedent so that methods / nested functions round-trip correctly.
    return textwrap.dedent(src).encode("utf-8")

def _deserialise_fcn(src_bytes: bytes, name: str) -> Callable | None:
    """
    Reconstruct a callable from its source bytes.

    Returns ``None`` and emits a warning on failure so that the rest of
    the FitResult is still usable even without the model function.
    """
    src = src_bytes.decode("utf-8") if isinstance(src_bytes, (bytes, bytearray)) else src_bytes
    
    import numpy as np
    ns: dict = {"np": np} 

    try:
        exec(compile(src, "<fitResult>", "exec"), ns)
    except Exception as exc:
        warnings.warn(
            f"Could not reconstruct model function from stored source "
            f"(name='{name}'): {exc}.  FitResult.fcn will be None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    # 1. First look for class instances 
    classes = [v for v in ns.values() if isinstance(v, type)]

    if classes:
        # using the last instance to instantiate 
        model_class = classes[-1]
        try:
            return model_class() 
        except TypeError:
            # Case the model class requires arguments
            return model_class
            
    # 2. Fallback: looking for callable functions
    candidates = [v for v in ns.values() if callable(v) and not isinstance(v, type)]
    if candidates:
        return candidates[-1]

    warnings.warn(
        "No callable found in stored model source.  FitResult.fcn will be None.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None

# =============================================================================
# Internal helpers
# =============================================================================

def _make_param_data(resample_type: str | None, Nresample: int | None) -> Data:
    """Return an empty, locked-mean Data object for a scalar fit parameter."""
    return Data.empty(
        resample_type=resample_type,
        shape=None,
        Nresample=Nresample,
        locked_mean=True,
    )


def _strip_log_key(key: str) -> tuple[str, bool]:
    """
    Detect lsqfit's ``log(param)`` key convention.

    Returns ``(plain_key, is_log)``.
    """
    if key.startswith("log(") and key.endswith(")"):
        return key[4:-1], True
    return key, False


def _deserialise_scalar_or_data(grp: h5py.Group, name: str) -> float | Data | None:
    """Read a field that was stored as either a plain dataset or a Data group."""
    if name not in grp:
        return None
    node = grp[name]
    if isinstance(node, h5py.Dataset):
        return float(node[()])
    return Data.deserialize(grp, node=name)


# =============================================================================
# FitResult
# =============================================================================

class FitResult:
    """
    Container for the result of a single fit.

    Populated by one of the ``import_from_*`` methods after a fit backend
    has run.  All per-resample quantities are stored in ``Data`` objects
    with ``locked_mean=True`` so that the central-value estimate is never
    silently overwritten by a resample mean.

    Parameters
    ----------
    abscissa : np.ndarray | Data | None
        Independent variable(s) used in the fit.
    Nresample : int | None
        Number of resamples.  ``None`` means no resample fits were done.
    resample_type : str | None
        ``'bst'`` (bootstrap) or ``'jkn'`` (jackknife).
    AIC_small_sample_correction : bool
        Apply the finite-sample AIC correction (AICc).  Default ``True``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        abscissa: np.ndarray | Data | None = None,
        Nresample: int | None = None,
        resample_type: str | None = None,
        AIC_small_sample_correction: bool = True,
    ) -> None:

        if resample_type not in (None, "bst", "jkn"):
            raise ValueError(
                f"resample_type must be None, 'bst', or 'jkn'; got '{resample_type}'."
            )
        if resample_type is not None and Nresample is None:
            raise ValueError(
                f"Nresample must be set when resample_type='{resample_type}'."
            )

        self.abscissa: np.ndarray | Data | None = abscissa
        self.Nresample: int | None              = Nresample
        self.resample_type: str | None          = resample_type
        self.AIC_small_sample_correction: bool  = AIC_small_sample_correction

        # Set by import_from_* methods
        self.fcn: Callable | None           = None
        self.dof: int | None                = None
        self.params: dict[str, Data]        = {}
        self.priors: dict[str, Prior]       = {}

        # Hessian-based (propagated) parameter errors, keyed like self.params.
        # Populated by import_from_iminuit / import_from_lsqfit when
        # Hessian information is available.  mean holds the central-value
        # propagated error; rspl[nres] holds the propagated error of resample nres.
        self.params_hessian_err: dict[str, Data] = {}

        # Fit-quality scalars / Data objects (scalar for CV-only, Data when
        # resample fits were also run).
        self.chi2:    Data | float | None = None
        self.p_value: Data | float | None = None
        self.AIC:     Data | float | None = None
        self.expected_chi2:Data | float | None = None
        self.expected_p_value: Data | float | None = None
        self.expected_AIC: Data | float | None = None

        # Initialise Data containers when resample fits are requested.
        if self.has_resamples:
            self.chi2    = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.p_value = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.AIC     = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.expected_chi2 = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.expected_p_value = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.expected_AIC = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)

        # Raw backend objects (not serialised).
        self.fit_output: Any = None                          # central value
        self.fit_output_rspl: list[Any] | None = None        # per-resample

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_resamples(self) -> bool:
        """True if resample fits were (or will be) performed."""
        return self.resample_type is not None

    @property
    def Ndata(self) -> int:
        """Number of fit data points, inferred from the abscissa shape."""
        if self.abscissa is None:
            raise RuntimeError(
                "Cannot determine Ndata: abscissa has not been set."
            )
        #return int(np.prod(self.abscissa.shape))
        return int( self.abscissa.shape[-1] )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def eval(self, abscissa: np.ndarray | Data | None = None) -> Data:
        """
        Evaluate the fitted model on *abscissa*.

        Parameters
        ----------
        abscissa : np.ndarray | Data | None
            Where to evaluate the model.  Defaults to the fit abscissa.

        Returns
        -------
        Data
            Model prediction with the same resample structure as the fit.
        """
        if self.fcn is None:
            raise RuntimeError("No model function (fcn) stored in this FitResult.")
        if abscissa is None:
            abscissa = self.abscissa

        if self.has_resamples:

            cv_params = {k: v.mean for k, v in self.params.items()}
            mean = self.fcn(abscissa, cv_params).astype(float)

            out = Data.empty(
                resample_type=self.resample_type,
                shape=mean.shape,
                Nresample=self.Nresample,
                locked_mean=True,
            )
            out.mean = mean

            for nres in range(self.Nresample):
                rs_params = {k: v.rspl[nres] for k, v in self.params.items()}
                out.rspl[nres] = self.fcn(abscissa, rs_params)

        else:
            cv_params = {k: gv.gvar(v.mean, self.params_hessian_err[k].mean  ) for k, v in self.params.items()}
            res = self.fcn(abscissa, cv_params)

            out = Data.import_gvar(
                g = res,
                locked_mean=True,
            )

        return out

    # ------------------------------------------------------------------
    # AIC
    # ------------------------------------------------------------------

    def _aicc_correction(self, k: int, dK: int) -> float:
        """Small-sample (AICc) correction term, shared by _compute_AIC and _compute_expected_AIC."""
        denominator = dK - k - 1
        if denominator <= 0:
            raise RuntimeError(
                f"AICc correction requires Ndata > Nparam + 1, "
                f"but Ndata={dK}, Nparam={k}."
            )
        return (2.0 * k**2 + 2.0 * k) / denominator

    def _compute_AIC(self, chi2: float) -> float:
        """
        Akaike information criterion.

        Formula from https://arxiv.org/abs/2305.19417 (eq. 3):

            AIC  = chi2 + 2k - 2d_K
            AICc = AIC  + (2k² + 2k) / (d_K - k - 1)

        where k = Nparam and d_K = Ndata.
        """
        if not self.params:
            raise RuntimeError(
                "Cannot compute AIC: no fit parameters have been stored yet."
            )
        k  = len(self.params)
        dK = self.Ndata

        aic = chi2 + 2.0 * (k - dK)

        if self.AIC_small_sample_correction:
            aic += self._aicc_correction(k, dK)

        return aic

    def _compute_expected_AIC(self, chi2: float, expected_chi2: float) -> float:
        r"""
        Expected AIC.

        Same as :meth:`_compute_AIC` (arXiv:2305.19417, eq. 3), but with the
        naive dof = d_K - k replaced by the Bruno & Sommer expected
        chi-squared :math:`\langle\chi^2\rangle`, eq. (2.13) of
        arXiv:2209.14188 (see :meth:`compute_expected_chi2`):

            <AIC>  = chi2 - 2*<chi2>

        """
        if not self.params:
            raise RuntimeError(
                "Cannot compute expected AIC: no fit parameters have been stored yet."
            )
        k  = len(self.params)
        dK = self.Ndata

        aic = chi2 - 2.0 * expected_chi2

        return aic


    # ------------------------------------------------------------------
    # Expected chi2
    # ------------------------------------------------------------------
    def compute_expected_chi2(
        self,
        cov: np.ndarray,
        W: np.ndarray,
        J: np.ndarray,
        Npriors: int = 0,
    ) -> float:
        r"""
        Compute the expected chi-squared for a fit with weight matrix W
        on data with covariance C.

        .. math::
            \langle \chi^2 \rangle = \mathrm{Tr}[W (I - H) C] + N_\mathrm{priors}

        where the hat matrix

        .. math::
            H = J^T (J W J^T)^{-1} J W

        projects residuals onto the model sensitivity subspace.
        J[i, j] = d model_j / d theta_i  (shape: Nparams × Nx).

        Reduces to dof = Nx - Nparams when W = C^{-1}.

        This is inspired by 
            M. Bruno and R. Sommer, 
            On fits to correlated and auto-correlated data 
            arXiv:2209.14188
        The form is the same as equation 2.13 but the weight matrix follows 
        the square (cov^-1 -> dof, 1/σ²)
        If cov is exact, this has an error of order 1/N. 
        If cov is estimated with error (1/sqrt(N)), this error is inhereted here.
        """
        try:
            Nx = cov.shape[0]
            M  = J @ W @ J.T                          # (Nparams, Nparams)
            H  = J.T @ np.linalg.solve(M, J @ W)     # (Nx, Nx)
            expected = np.trace(W @ (np.eye(Nx) - H) @ cov) + Npriors
            return float(expected)
        except np.linalg.LinAlgError as e:
            print(e)
            return self.dof
        except Exception as e:
            raise e

    # ------------------------------------------------------------------
    # p-value (eq. 2.18)
    # ------------------------------------------------------------------
    def compute_p_value(
        self,
        cov: np.ndarray,
        W: np.ndarray,
        J: np.ndarray,
        chi2_obs: float,
        Npriors: int = 0,
        Nmc: int = 10_000,
        seed: int = 0,
    ) -> float:
        r"""
        Compute the quality-of-fit p-value for a fit with weight matrix W
        on data with covariance C.

        .. math::
            Q(\chi^2_\mathrm{obs}, \nu) = P\Big(\sum_{j=1}^{N_\nu} \lambda_j(\nu)\, z_j^2 \ge \chi^2_\mathrm{obs}\Big),
            \qquad z_j \sim \mathcal{N}(0,1)\ \text{i.i.d.}

        where the :math:`\lambda_j(\nu)` are the strictly positive eigenvalues of

        .. math::
            \nu = C^{1/2} W (I - H) C^{1/2}

        and H is the same hat matrix used in :meth:`compute_expected_chi2`
        (H = J^T (J W J^T)^{-1} J W). J[i, j] = d model_j / d theta_i
        (shape: Nparams x Nx). Note W here is the same bilinear weight
        matrix as elsewhere in this class (chi2 = r^T W r); W(I-H) is
        symmetric because W H is symmetric (H is self-adjoint w.r.t. the
        W-inner product), so a single factor of W is needed here — unlike
        the two factors of the weight operator in eq. (2.17) of the paper,
        which is written in terms of a matrix square root W_paper with
        W = W_paper^2.

        This is inspired by
            M. Bruno and R. Sommer,
            On fits to correlated and auto-correlated data
            arXiv:2209.14188
        The form is equation 2.18. Q reduces to the standard incomplete-gamma
        p-value (self.dof degrees of freedom) when W = C^{-1/2} exactly, but
        is the correct quality-of-fit for arbitrary weight matrices W, e.g.
        for uncorrelated fits.

        Each prior contributes an extra unit eigenvalue, matching the
        ``+Npriors`` term of :meth:`compute_expected_chi2`.

        Q is estimated by Monte Carlo (as proposed in sect. 2.2 of the
        paper): the statistical precision on Q is of order 1/sqrt(Nmc).
        """
        try:
            Nx = cov.shape[0]
            M  = J @ W @ J.T
            H  = J.T @ np.linalg.solve(M, J @ W)

            # eig(C^{1/2} X C^{1/2}) == eig(C X) for any square root of C,
            # so the (symmetric) Cholesky factor avoids forming C^{1/2}.
            L  = np.linalg.cholesky(cov)
            nu = L.T @ (W @ (np.eye(Nx) - H)) @ L
            nu = 0.5 * (nu + nu.T)

            eigvals = np.linalg.eigvalsh(nu)
            tol = eigvals.max() * 1e-10 if eigvals.size else 0.0
            lambdas = eigvals[eigvals > tol]

            if Npriors:
                lambdas = np.concatenate([lambdas, np.ones(Npriors)])

            if lambdas.size == 0:
                return 1.0 if chi2_obs <= 0 else 0.0

            rng     = np.random.default_rng(seed)
            z       = rng.standard_normal(size=(Nmc, lambdas.size))
            chi2_mc = (z**2) @ lambdas
            return float(np.mean(chi2_mc >= chi2_obs))

        except np.linalg.LinAlgError as e:
            print(e)
            return float(gammaincc(self.dof / 2.0, chi2_obs / 2.0))
        except Exception as e:
            raise e

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        abscissa_range = (
            f"({self.abscissa[0]}, {self.abscissa[-1]})"
            if self.abscissa is not None
            else "?"
        )
        
        if hasattr(self, "truncation_dimension"):
            truncation_report = f"THC @ truncation dimension:{self.truncation_dimension}, "
        else:
            truncation_report = ""

        resample_tag = self.resample_type if self.has_resamples else False
        if self.has_resamples:
            lines = [
                f"FitResult[{truncation_report}{abscissa_range}, Ndata={self.Ndata}, resample:{resample_tag}]:",
                f"  χ²/dof [dof]   = {self.chi2.mean / self.dof:.3g} [{self.dof}]",
                f"  χ²/<χ²> [<χ²>] = {self.chi2.mean / self.expected_chi2.mean:.3g} [{self.expected_chi2.mean:.3g}]",
                f"  p-value        = {self.p_value.mean:.3g}",
                f"  <p-value>      = {self.expected_p_value.mean:.3g}",
                f"  AIC            = {self.AIC.mean:.3g}",
                f"  <AIC>          = {self.expected_AIC.mean:.3g}",
            ]
        else:
            lines = [
                f"FitResult[{truncation_report}{abscissa_range}, Ndata={self.Ndata}, resample:{resample_tag}]:",
                f"  χ²/dof [dof] = {self.chi2 / self.dof:.3g} [{self.dof}]",
                f"  χ²/<χ²> [<χ²>] = {self.chi2 / self.expected_chi2:.3g} [{self.expected_chi2:.3g}]",
                f"  p-value      = {self.p_value:.3g}",
                f"  <p-value>    = {self.expected_p_value:.3g}",
                f"  AIC          = {self.AIC:.3g}",
                f"  <AIC>        = {self.expected_AIC:.3g}",
            ]
    
        for key, p in self.params.items():
            prior_tag = f"  [{self.priors[key]}]" if key in self.priors else ""

            if self.has_resamples:
                lines.append(f"    {key}: {p.gvar()}{prior_tag}")
            else:
                lines.append(f"    {key}: {gv.gvar(p.mean, self.params_hessian_err[key].mean)}{prior_tag}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # HDF5 serialisation
    # ------------------------------------------------------------------

    def serialize(self, h5_handle: h5py.File, node: str | None = None) -> None:
        """
        Write this FitResult into an HDF5 group.

        Parameters
        ----------
        h5_handle : h5py.File
            Open HDF5 file handle.
        node : str | None
            Path of the group to create.  If ``None``, writes into *h5_handle*
            directly.
        """
        if not self.params:
            raise ValueError(
                "No fit results to serialise.  Run a fit first."
            )

        grp = h5_handle if node is None else h5_handle.create_group(node)

        # --- abscissa ---
        if isinstance(self.abscissa, Data):
            self.abscissa.serialize(grp, node="abscissa")
        elif self.abscissa is not None:
            grp.create_dataset("abscissa", data=self.abscissa)

        # --- scalar metadata ---
        if self.dof is not None:
            grp.create_dataset("dof", data=self.dof)
        if self.Nresample is not None:
            grp.create_dataset("Nresample", data=self.Nresample)
        if self.resample_type is not None:
            grp.create_dataset("resample_type", data=self.resample_type)

        # --- model function (source code, not pickle) ---
        if self.fcn is not None:
            try:
                src_bytes = _serialise_fcn(self.fcn)
                grp.create_dataset("fcn_source", data=np.void(src_bytes))
                grp.create_dataset("fcn_name",   data=getattr(self.fcn, "__name__", "unknown"))
            except ValueError as exc:
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)

        # --- fit parameters ---
        for key, param in self.params.items():
            param.serialize(grp, node=f"params/{key}")

        # --- Hessian parameter errors ---
        for key, herr in self.params_hessian_err.items():
            herr.serialize(grp, node=f"params_hessian_err/{key}")

        # --- priors ---
        for key, prior in self.priors.items():
            prior.serialize(grp, node=f"priors/{key}")

        # --- fit quality ---
        for name, val in (("chi2", self.chi2), ("expected_chi2", self.expected_chi2), ("p_value", self.p_value), ("expected_p_value", self.expected_p_value), ("AIC", self.AIC), ("expected_AIC", self.expected_AIC)):
            if isinstance(val, Data):
                val.serialize(grp, node=name)
            elif val is not None:
                grp.create_dataset(name, data=val)

        # --- Lambda eigenvalues (THC backend only) ---
        for _attr in ("lambda_eigs_real", "lambda_eigs_imag"):
            _val = getattr(self, _attr, None)
            if _val is not None:
                if isinstance(_val, Data):
                    _val.serialize(grp, node=_attr)
                else:
                    grp.create_dataset(_attr, data=_val)

    @staticmethod
    def deserialize(h5_handle: h5py.File, node: str | None = None) -> FitResult:
        """
        Reconstruct a FitResult from an HDF5 group.

        Parameters
        ----------
        h5_handle : h5py.File
            Open HDF5 file handle.
        node : str | None
            Path of the group to read from.

        Returns
        -------
        FitResult
        """
        grp = h5_handle if node is None else h5_handle[node]

        # Determine resample configuration first so the constructor can
        # allocate the right Data containers.
        Nresample     = int(grp["Nresample"][()]) if "Nresample" in grp else None
        resample_type = (
            grp["resample_type"][()].decode("utf-8")
            if "resample_type" in grp else None
        )

        out = FitResult(Nresample=Nresample, resample_type=resample_type)

        # abscissa
        if "abscissa" in grp:
            if isinstance(grp["abscissa"], h5py.Dataset):
                out.abscissa = grp["abscissa"][()]
            else:
                out.abscissa = Data.deserialize(grp, node="abscissa")

        # scalar metadata
        if "dof" in grp:
            out.dof = int(grp["dof"][()])

        # model function
        if "fcn_source" in grp:
            src_bytes = bytes(grp["fcn_source"][()])
            name      = (
                grp["fcn_name"][()].decode("utf-8")
                if "fcn_name" in grp else "unknown"
            )
            out.fcn = _deserialise_fcn(src_bytes, name)

        # fit parameters
        if "params" in grp:
            out.params = {
                key: Data.deserialize(grp, node=f"params/{key}")
                for key in grp["params"].keys()
            }

        # Hessian parameter errors
        if "params_hessian_err" in grp:
            out.params_hessian_err = {
                key: Data.deserialize(grp, node=f"params_hessian_err/{key}")
                for key in grp["params_hessian_err"].keys()
            }

        # priors
        if "priors" in grp:
            out.priors = {
                key: Prior.deserialize(grp, node=f"priors/{key}")
                for key in grp["priors"].keys()
            }

        # fit quality
        for name in ("chi2", "expected_chi2", "p_value", "expected_p_value", "AIC", "expected_AIC"):
            val = _deserialise_scalar_or_data(grp, name)
            if val is not None:
                setattr(out, name, val)

        # Lambda eigenvalues (THC backend only)
        for _attr in ("lambda_eigs_real", "lambda_eigs_imag"):
            if _attr in grp:
                _node = grp[_attr]
                if isinstance(_node, h5py.Dataset):
                    setattr(out, _attr, _node[()])
                else:
                    setattr(out, _attr, Data.deserialize(grp, node=_attr))

        return out

    # ------------------------------------------------------------------
    # Private helpers shared by all importers
    # ------------------------------------------------------------------

    def _ensure_param(self, key: str) -> None:
        """Create the Data entry for *key* if it does not yet exist."""
        if key not in self.params:
            self.params[key] = _make_param_data(self.resample_type, self.Nresample)

    def _ensure_hessian_err(self, key: str) -> None:
        """Create the Hessian-error Data entry for *key* if it does not yet exist."""
        if key not in self.params_hessian_err:
            self.params_hessian_err[key] = _make_param_data(self.resample_type, self.Nresample)

    def _ensure_lambda_store(self,max_k: int) -> None:
        """Create the Data entry for eigenvaues of the THC (exponential energies)"""
        if not hasattr(self, "lambda_eigs") or getattr(self, "lambda_eigs") is None:
            if self.has_resamples:
                store = Data.empty(
                    resample_type = self.resample_type,
                    shape         = (max_k,),
                    Nresample     = self.Nresample,
                    locked_mean   = True,
                    dtype         = complex
                )
                self.lambda_eigs = store
            else:
                # set as plain array on CV path
                self.lambda_eigs = None

    def _store_fit_quality(
        self,
        chi2: float,
        expected_chi2: float | None,
        expected_p_value: float | None,
        nres: int | None,
    ) -> None:
        """
        Write chi2, expected chi2 (defaults to dof), p-value, expected
        p-value (eq. 2.18 of arXiv:2209.14188; defaults to p-value), AIC,
        and expected AIC (dof -> expected chi2 in the AIC formula;
        defaults to AIC) for a central-value or resample fit.
        """
        exp_chi2_val = expected_chi2 if expected_chi2 is not None else self.dof

        p_val     = float(gammaincc(self.dof / 2.0, chi2 / 2.0))
        aic       = self._compute_AIC(chi2)
        exp_p_val = expected_p_value if expected_p_value is not None else p_val
        exp_aic   = self._compute_expected_AIC(chi2, exp_chi2_val)

        if nres is None:
            if self.has_resamples:
                self.chi2.mean            = chi2
                self.p_value.mean         = p_val
                self.AIC.mean             = aic
                self.expected_p_value.mean = exp_p_val
                self.expected_chi2.mean   = exp_chi2_val
                self.expected_AIC.mean    = exp_aic
            else:
                self.chi2              = chi2
                self.p_value           = p_val
                self.AIC               = aic
                self.expected_p_value  = exp_p_val
                self.expected_chi2     = exp_chi2_val
                self.expected_AIC      = exp_aic
        else:
            self.chi2.rspl[nres]              = chi2
            self.p_value.rspl[nres]           = p_val
            self.AIC.rspl[nres]               = aic
            self.expected_p_value.rspl[nres]  = exp_p_val
            self.expected_chi2.rspl[nres]     = exp_chi2_val
            self.expected_AIC.rspl[nres]      = exp_aic

    def _init_resample_store(self) -> None:
        """Lazily create the per-resample raw-output list."""
        if self.fit_output_rspl is None:
            self.fit_output_rspl = [None] * self.Nresample

    # ------------------------------------------------------------------
    # Importers
    # ------------------------------------------------------------------

    def import_from_adam(
        self,
        cost_history: list[float],
        nres: int | None = None,
    ): 
        r"""
            import cost history of ADAM. 

            Due to the stoppping criterion it is not guaranteed that all fits (resamples) have
            a cost history of the same length. Therefore, we simply store a collection of the 
            cost histories. 
            These are stored as numpy arrays 
        """
        if not hasattr(self,"cost_history"):
            self.cost_history = None
            
            if self.has_resamples:
                self.cost_history_rspl = [None] * self.Nresample

        if nres is None:
            self.cost_history = np.asarray(cost_history)           
        else:
            self.cost_history_rspl[nres] = np.asarray(cost_history)

    def _build_jacobian(
        self,
        model: Callable,
        cv: dict[str, float],
        variable_projection: dict[str, float] | None,
    ) -> np.ndarray:
        r"""
        Full parameter Jacobian ``J[i, j] = d model_j / d theta_i`` at *cv*.

        ``compute_expected_chi2`` / ``compute_p_value`` need the sensitivity
        directions of **all** fitted parameters — the hat matrix
        :math:`H = J^T (J W J^T)^{-1} J W` must project onto the full model
        subspace.

        By convention ``model.grad`` returns the derivatives with respect to
        the *nonlinear* parameters only, because that is what the
        variable-projection cost function needs.  So when variable projection
        is in use, the rows belonging to the linear parameters have to be added
        back here; they are simply the design-matrix columns
        ``d model / d A_k = model(x, {..., A_k: 1, A_l != k: 0})``.

        Without this the hat matrix would project onto too small a subspace and
        the expected chi² (and hence the expected p-value and AIC) would be
        systematically too large.
        """
        if not hasattr(model, "grad"):
            return _jacobian_fd(model, self.abscissa, cv)

        rows = np.asarray(model.grad(self.abscissa, cv), dtype=float)
        if rows.size == 0:
            rows = rows.reshape(0, np.asarray(model(self.abscissa, cv)).size)

        if variable_projection:
            lin_keys = list(variable_projection)
            lin_rows = np.empty((len(lin_keys), rows.shape[-1]), dtype=float)
            for i, key in enumerate(lin_keys):
                test = {**cv, **{k: 0.0 for k in lin_keys}, key: 1.0}
                lin_rows[i] = np.asarray(model(self.abscissa, test), dtype=float)
            rows = np.vstack([rows, lin_rows]) if rows.shape[0] else lin_rows

        return rows

    def import_from_iminuit(
        self,
        minuit: Any,
        Ndata: int,
        model: Callable,
        cov: np.ndarray | None = None,
        W: np.ndarray | None = None,
        variable_projection: dict[str, float] | None = None,
        variable_projection_hessians: dict[str, float] | None = None,
        prior: dict[str, Prior] | None = None,
        nres: int | None = None,
    ) -> None:
        """
        Import parameters and fit quality from a ``iminuit.Minuit`` object.

        Parameters
        ----------
        minuit : iminuit.Minuit
            Completed Minuit instance (after ``migrad()``).
        Ndata : int
            Number of fit data points used for χ² and AIC.
        model : callable
            The model function (stored on the first call).
        cov: np.ndarray 
            The data covariance. This is used to compute the expected chi²
        W: np.ndarray
            The weight matrix of the chi² definition. This is used to compute the expected chi². 
        variable_projection : dict[str, float] | None
            Linear-parameter values from variable-projection fits.
        prior : dict[str, Prior] | None
            Priors used in the fit (central-value fit only).
        nres : int | None
            Resample index.  ``None`` → central-value fit.
        """
        if self.fcn is None:
            self.fcn = model

        if self.dof is None:
            Nnonlin = len(minuit.params)
            Nlin    = len(variable_projection) if variable_projection is not None else 0
            Nprior  = len(prior) if prior is not None else 0
            self.dof = Ndata - Nnonlin - Nlin + Nprior

        # --- parameter values ---
        for param in minuit.params:
            key = param.name
            self._ensure_param(key)
            if nres is None:
                self.params[key].mean = param.value
            else:
                self.params[key].rspl[nres] = param.value

        # --- variable-projection linear parameters ---
        if variable_projection is not None:
            for key, value in variable_projection.items():
                self._ensure_param(key)
                if nres is None:
                    self.params[key].mean = value
                else:
                    self.params[key].rspl[nres] = value

        # --- Hessian (propagated) errors from Minuit / Recomputed ---
        minuit.hesse()
        if variable_projection_hessians:
            for key,error in variable_projection_hessians.items():
                self._ensure_hessian_err(key)
                if nres is None:
                    self.params_hessian_err[key].mean = error
                else:
                    self.params_hessian_err[key].rspl[nres] = error

        else:
            # the hessian errors are compute within iminuit. we can just
            # import them:
            for param in minuit.params:
                key = param.name
                self._ensure_hessian_err(key)
                if nres is None:
                    self.params_hessian_err[key].mean = param.error
                else:
                    self.params_hessian_err[key].rspl[nres] = param.error

        # --- priors (central-value fit only) ---
        if nres is None and prior is not None:
            for key, p in prior.items():
                self.priors[key] = p

        if cov is not None and W is not None:
            cv = {
                k: self.params[k].mean if nres is None else self.params[k].rspl[nres]
                for k in self.params
            }

            J = self._build_jacobian(model, cv, variable_projection)

            Npriors  = len(prior) if prior else 0
            exp_chi2 = self.compute_expected_chi2(cov, W, J, Npriors=Npriors)
            exp_pval = self.compute_p_value(cov, W, J, minuit.fval, Npriors=Npriors)
        else:
            exp_chi2 = self.dof
            exp_pval = None

        # --- fit quality ---
        self._store_fit_quality(minuit.fval, exp_chi2, exp_pval, nres)

        # --- raw backend object (not serialised) ---
        if nres is None:
            self.fit_output             = minuit
            self.number_function_calls  = minuit.nfcn   # for testing
        else:
            self._init_resample_store()
            self.fit_output_rspl[nres] = minuit

    def import_from_lsqfit(
        self,
        nlf: Any,
        cov: np.ndarray | None = None,
        W: np.ndarray | None = None,
        nres: int | None = None,
    ) -> None:
        """
        Import parameters and fit quality from a ``lsqfit.nonlinear_fit`` object.

        lsqfit uses ``gvar.GVar`` objects for its parameters; we extract the
        mean value here.  Log-normal priors are encoded by lsqfit as
        ``"log(key)"`` entries — these are unwrapped and the physical (linear)
        value is stored under the plain ``key``.

        The ``gvar`` standard deviation from the lsqfit output is stored in
        ``params_hessian_err`` as the propagated Gaussian error estimate.

        Parameters
        ----------
        nlf : lsqfit.nonlinear_fit
            Completed lsqfit fit object.
        cov: np.ndarray 
            The data covariance. This is used to compute the expected chi²
        W: np.ndarray
            The weight matrix of the chi² definition. This is used to compute the expected chi². 
        nres : int | None
            Resample index.  ``None`` → central-value fit.
        """
        import gvar as gv

        if self.fcn is None:
            self.fcn = nlf.fcn

        if self.dof is None:
            self.dof = nlf.dof

        for raw_key, gvar_val in nlf.p.items():
            key, is_log = _strip_log_key(raw_key)
            mean  = float(np.exp(gv.mean(gvar_val)) if is_log else gv.mean(gvar_val))
            # For log-normal params the propagated error in linear space is
            # mean * sdev_of_log, but we store the gvar sdev directly so the
            # user can interpret it in the natural parameterisation.
            hess_err = float(gv.sdev(gvar_val))

            self._ensure_param(key)
            self._ensure_hessian_err(key)

            if nres is None:
                self.params[key].mean              = mean
                self.params_hessian_err[key].mean  = hess_err
            else:
                self.params[key].rspl[nres]             = mean
                self.params_hessian_err[key].rspl[nres] = hess_err

        # --- priors (central-value fit only) ---
        if nres is None and nlf.prior is not None:
            imported = Prior.import_from_lsqfit(nlf.prior)
            for key, prior in imported.items():
                self.priors[key] = prior

        if cov is not None and W is not None:
            cv = {
                k: self.params[k].mean if nres is None else self.params[k].rspl[nres]
                  for k in self.params
            }

            J  = (nlf.fcn.grad(self.abscissa, cv) if hasattr(nlf.fcn, "grad") else _jacobian_fd(nlf.fcn, self.abscissa, cv))

            Npriors  = len(self.priors) if self.priors else 0
            exp_chi2 = self.compute_expected_chi2(cov, W, J, Npriors=Npriors)
            exp_pval = self.compute_p_value(cov, W, J, float(nlf.chi2), Npriors=Npriors)
        else:
            exp_chi2 = self.dof
            exp_pval = None

        # --- fit quality ---
        # lsqfit stores p-value in nlf.Q; we re-compute from chi2 for
        # consistency with the other backends.
        self._store_fit_quality(float(nlf.chi2), exp_chi2, exp_pval, nres)

        # --- raw backend object (not serialised) ---
        if nres is None:
            self.fit_output = nlf
            self.number_function_calls  = nlf.nit   # for testing
        else:
            self._init_resample_store()
            self.fit_output_rspl[nres] = nlf

    def import_from_linear_regression(
        self,
        target_data: np.ndarray,
        result_params: np.ndarray,
        design_matrix: np.ndarray,
        weight_matrix: np.ndarray,
        parameter_names: tuple[str, ...],
        cov: np.ndarray,
        nres: int | None = None,
    ) -> None:
        """
        Import parameters and fit quality from the closed-form linear solver.

        Gaussian error propagation for linear regression is available in
        closed form via the covariance C = (X^T W X)^{-1}.  The square
        root of the diagonal of C is stored in ``params_hessian_err``.

        Parameters
        ----------
        target_data : np.ndarray
            The ordinate values that were fitted (mean or one resample).
        result_params : np.ndarray
            Best-fit parameter vector, ordered like *parameter_names*.
        design_matrix : np.ndarray, shape (N, p)
            Design matrix X used in the fit.
        weight_matrix : np.ndarray, shape (N, N)
            Weight matrix W = C^{-1}.
        parameter_names : tuple[str, ...]
            Names of the fit parameters.
        cov: np.ndarray 
            The data covariance. This is used to compute the expected chi²
        nres : int | None
            Resample index.  ``None`` → central-value fit.
        """
        has_intercept = len(result_params) == 2

        if self.fcn is None:
            if has_intercept:
                m_key, b_key = parameter_names
                self.fcn = lambda x, p: x * p[m_key] + p[b_key]
            else:
                (m_key,) = parameter_names
                self.fcn = lambda x, p: x * p[m_key]

        if self.dof is None:
            self.dof = self.Ndata - len(result_params)

        # Closed-form parameter covariance: C = (X^T W X)^{-1}
        try:
            param_cov  = np.linalg.inv(design_matrix.T @ weight_matrix @ design_matrix)
            param_sdev = np.sqrt(np.diag(param_cov))
        except np.linalg.LinAlgError:
            param_sdev = np.full(len(result_params), np.nan)

        for idx, key in enumerate(parameter_names):
            self._ensure_param(key)
            self._ensure_hessian_err(key)
            if nres is None:
                self.params[key].mean             = result_params[idx]
                self.params_hessian_err[key].mean = param_sdev[idx]
            else:
                self.params[key].rspl[nres]              = result_params[idx]
                self.params_hessian_err[key].rspl[nres]  = param_sdev[idx]

        # Priors are not used in linear regression.
        self.priors = {}

        # chi2 = r^T W r
        residuals = target_data - design_matrix @ result_params
        chi2      = float(residuals.T @ weight_matrix @ residuals)

        exp_chi2 = self.compute_expected_chi2(
            cov=cov,      # pass this in as a new argument
            W=weight_matrix,
            J=design_matrix.T,     # shape (Nparams, Nx) — already available
            Npriors=0,
        )
        exp_pval = self.compute_p_value(
            cov=cov,
            W=weight_matrix,
            J=design_matrix.T,
            chi2_obs=chi2,
            Npriors=0,
        )

        self._store_fit_quality(chi2, exp_chi2, exp_pval, nres)

    def import_from_constant_fit(
        self,
        target_data: np.ndarray,
        result_constant: np.ndarray,
        weight_matrix: np.ndarray,
        constant_name: str,
        cov: np.ndarray,
        nres: int | None = None,
    ) -> None:
        """
        Import parameters and fit quality from the constant fit.

        Gaussian error propagation is available.

        Parameters
        ----------
        target_data : np.ndarray
            The ordinate values that were fitted (mean or one resample).
        result_constant : np.ndarray
            Best-fit parameter, ordered like.
        weight_matrix : np.ndarray, shape (N, N)
            Weight matrix W = C^{-1}.
        constant_name : str
            Names of the fit parameters.
        cov: np.ndarray 
            The data covariance. This is used to compute the expected chi²
        nres : int | None
            Resample index.  ``None`` → central-value fit.
        """

        if self.fcn is None:
            self.fcn = lambda x, p: np.full_like(x,p[constant_name])

        if self.dof is None:
            self.dof = self.Ndata - 1

        # closed form gaussian error:
        numerator_weights = np.sum(weight_matrix, axis=0) # This is 1^T @ W
        variance_theta = (numerator_weights @ cov @ numerator_weights.T) / (np.sum(weight_matrix)**2)
        sigma_theta = np.sqrt(variance_theta)
        self._ensure_param(constant_name)
        self._ensure_hessian_err(constant_name)
        
        if nres is None:
            self.params[constant_name].mean             = result_constant
            self.params_hessian_err[constant_name].mean = sigma_theta
        else:
            self.params[constant_name].rspl[nres]              = result_constant
            self.params_hessian_err[constant_name].rspl[nres]  = sigma_theta

        # Priors are not used in constant fits.
        self.priors = {}

        # chi2 = r^T W r
        residuals = target_data - result_constant
        chi2      = float(residuals.T @ weight_matrix @ residuals)

        J = np.asarray([ np.ones_like(target_data) ])  # shape (Nparams, Nx) — already available
        exp_chi2 = self.compute_expected_chi2(
            cov=cov,      # pass this in as a new argument
            W=weight_matrix,
            J=J,
            Npriors=0,
        )
        exp_pval = self.compute_p_value(
            cov=cov,
            W=weight_matrix,
            J=J,
            chi2_obs=chi2,
            Npriors=0,
        )

        self._store_fit_quality(chi2, exp_chi2, exp_pval, nres)

    def import_from_thc(
        self,
        energies,
        overlaps,
        lambda_eigvals,
        chi2_val: float,
        max_k: int,
        W_chi2,
        cov,
        abscissa_fit,
        truncation_dimension:int,
        nres=None,
    ) -> None:
        """
        Import results from a THC (Truncated Hankel Correlator) run.
    
        Unlike the other importers, there is no backend object: the relevant
        quantities are passed as plain numpy arrays.
    
        Parameters
        ----------
        energies : np.ndarray, shape (Nstates,)
            Physical energies E_l (ascending), np.nan for unresolved states.
            **Physical result** — stored in ``self.params`` as ``E0, E1, ...``.
        overlaps : np.ndarray, shape (Nstates,)
            Amplitude coefficients A_l, np.nan for unresolved states.
            **Physical result** — stored in ``self.params`` as ``A0, A1, ...``.
        lambda_real : np.ndarray, shape (k_used,)
            Real parts of the raw (unfiltered) eigenvalues Lambda of X,
            where Lambda_l ≈ exp(-E_l * delta_t).
            **Algorithmic result** — stored in ``self.lambda_eigs_real``,
            NOT in ``self.params``.  Padded to ``max_k`` with np.nan.
        lambda_imag : np.ndarray, shape (k_used,)
            Imaginary parts of Lambda.  Stored in ``self.lambda_eigs_imag``.
            A physical energy has |Im(Lambda)| ≈ 0 and 0 < |Lambda| < 1.
        chi2_val : float
            chi²_obs = r^T W r evaluated by the thc() driver.
        max_k : int
            Length of the lambda_eigs_real / lambda_eigs_imag storage arrays.
        W_chi2 : np.ndarray, shape (Ndata, Ndata)
            Weight matrix used for chi2_obs (also used for expected_chi2).
        cov : np.ndarray, shape (Ndata, Ndata)
            Full data covariance (used for compute_expected_chi2).
        abscissa_fit : np.ndarray, shape (Ndata,)
            Timeslice values t0, t0+1, ..., T used in chi2.
        truncation_dimension :
            The truncation dimension of the thc method.
        nres : int | None
            Resample index.  None -> central-value result.
        """
        import numpy as np
        from .data import Data as _Data
    
        Nstates = len(energies)

        self.truncation_dimension = truncation_dimension
    
        # ------------------------------------------------------------------
        # Physical parameters: energies E0..E{N-1} and amplitudes A0..A{N-1}
        # ------------------------------------------------------------------
        for k in range(Nstates):
            for prefix, arr in (("E", energies), ("A", overlaps)):
                key = f"{prefix}{k}"
                self._ensure_param(key)
                val = float(arr[k])   # may be nan; stored as-is
                if nres is None:
                    self.params[key].mean = val
                else:
                    self.params[key].rspl[nres] = val
    
        # ------------------------------------------------------------------
        # Algorithmic result: Lambda eigenvalues, split into real and imag.
        # Stored as two separate Data objects to avoid complex dtype issues
        # in Data internals and HDF5 serialisation.  Lazily initialised on
        # the first call so the storage size (max_k) is known.
        # ------------------------------------------------------------------
        self._ensure_lambda_store(max_k=max_k)

        extended_evals = np.full(max_k, np.nan, dtype=complex)
        n_lam =  min(len(lambda_eigvals), max_k)
        extended_evals[:n_lam] = lambda_eigvals[:n_lam]

        if nres is None:
            if self.has_resamples:
                self.lambda_eigs.mean = extended_evals
            else:
                self.lambda_eigs = extended_evals
        else:
            self.lambda_eigs.rspl[nres] = extended_evals
    
        # ------------------------------------------------------------------
        # expected_chi2 / expected_p_value: compute if model.grad is
        # available and no nan params
        # ------------------------------------------------------------------
        exp_chi2 = self.dof
        exp_pval = None
        if (
            self.fcn is not None
            and hasattr(self.fcn, "grad")
            and not np.any(np.isnan(energies))
            and cov is not None
            and W_chi2 is not None
        ):
            params_cur = {}
            for k in range(Nstates):
                params_cur[f"E{k}"] = (
                    self.params[f"E{k}"].mean if nres is None
                    else self.params[f"E{k}"].rspl[nres]
                )
                params_cur[f"A{k}"] = (
                    self.params[f"A{k}"].mean if nres is None
                    else self.params[f"A{k}"].rspl[nres]
                )

            J        = self.fcn.grad(abscissa_fit, params_cur)   # (2*Nstates, Ndata)
            exp_chi2 = self.compute_expected_chi2(
                cov=cov, W=W_chi2, J=J, Npriors=0,
            )
            exp_pval = self.compute_p_value(
                cov=cov, W=W_chi2, J=J, chi2_obs=chi2_val, Npriors=0,
            )

        # ------------------------------------------------------------------
        # chi2, p-value, AIC via the standard _store_fit_quality helper
        # ------------------------------------------------------------------
        self._store_fit_quality(chi2_val, exp_chi2, exp_pval, nres)
