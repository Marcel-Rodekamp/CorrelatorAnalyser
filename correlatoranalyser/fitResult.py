from __future__ import annotations

import inspect
import textwrap
import warnings
from collections.abc import Callable
from typing import Any

import h5py
import numpy as np
from scipy.special import gammaincc

from .data import Data
from .prior import Prior


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
    ns: dict = {}
    try:
        exec(compile(src, "<fitResult>", "exec"), ns)   # noqa: S102
    except Exception as exc:
        warnings.warn(
            f"Could not reconstruct model function from stored source "
            f"(name='{name}'): {exc}.  FitResult.fcn will be None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    # The function we want is the last callable defined in the source.
    candidates = [v for v in ns.values() if callable(v) and not isinstance(v, type)]
    if not candidates:
        warnings.warn(
            "No callable found in stored model source.  FitResult.fcn will be None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return candidates[-1]


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

        # Initialise Data containers when resample fits are requested.
        if self.has_resamples:
            self.chi2    = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.p_value = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)
            self.AIC     = Data.empty(resample_type=resample_type, shape=None, Nresample=Nresample, locked_mean=True)

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
        return int(np.prod(self.abscissa.shape))

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

        cv_params = {k: v.mean for k, v in self.params.items()}
        mean = self.fcn(abscissa, cv_params).astype(float)

        out = Data.empty(
            resample_type=self.resample_type,
            shape=mean.shape,
            Nresample=self.Nresample,
            locked_mean=True,
        )
        out.mean = mean

        if self.has_resamples:
            for nres in range(self.Nresample):
                rs_params = {k: v.rspl[nres] for k, v in self.params.items()}
                out.rspl[nres] = self.fcn(abscissa, rs_params)

        return out

    # ------------------------------------------------------------------
    # AIC
    # ------------------------------------------------------------------

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
            denominator = dK - k - 1
            if denominator <= 0:
                raise RuntimeError(
                    f"AICc correction requires Ndata > Nparam + 1, "
                    f"but Ndata={dK}, Nparam={k}."
                )
            aic += (2.0 * k**2 + 2.0 * k) / denominator

        return aic

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        abscissa_range = (
            f"({self.abscissa[0]}, {self.abscissa[-1]})"
            if self.abscissa is not None
            else "?"
        )
        resample_tag = self.resample_type if self.has_resamples else False
        if self.has_resamples:
            lines = [
                f"FitResult[{abscissa_range}, Ndata={self.Ndata}, resample:{resample_tag}]:",
                f"  χ²/dof [dof] = {self.chi2.mean / self.dof:.3g} [{self.dof}]",
                f"  p-value      = {self.p_value.mean:.3g}",
                f"  AIC          = {self.AIC.mean:.3g}",
            ]
        else:
            lines = [
                f"FitResult[{abscissa_range}, Ndata={self.Ndata}, resample:{resample_tag}]:",
                f"  χ²/dof [dof] = {self.chi2 / self.dof:.3g} [{self.dof}]",
                f"  p-value      = {self.p_value:.3g}",
                f"  AIC          = {self.AIC:.3g}",
            ]
    
        for key, p in self.params.items():
            prior_tag = f"  [{self.priors[key]}]" if key in self.priors else ""
            lines.append(f"    {key}: {p.gvar()}{prior_tag}")
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
        for name, val in (("chi2", self.chi2), ("p_value", self.p_value), ("AIC", self.AIC)):
            if isinstance(val, Data):
                val.serialize(grp, node=name)
            elif val is not None:
                grp.create_dataset(name, data=val)

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
        for name in ("chi2", "p_value", "AIC"):
            val = _deserialise_scalar_or_data(grp, name)
            if val is not None:
                setattr(out, name, val)

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

    def _store_fit_quality(
        self, chi2: float, nres: int | None
    ) -> None:
        """Write chi2, p-value, and AIC for a central-value or resample fit."""
        p_val = float(gammaincc(self.dof / 2.0, chi2 / 2.0))
        aic   = self._compute_AIC(chi2)

        if nres is None:
            if self.has_resamples:
                self.chi2.mean    = chi2
                self.p_value.mean = p_val
                self.AIC.mean     = aic
            else:
                self.chi2         = chi2
                self.p_value      = p_val
                self.AIC          = aic               
        else:
            self.chi2.rspl[nres]    = chi2
            self.p_value.rspl[nres] = p_val
            self.AIC.rspl[nres]     = aic

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
        if nres is None:
            # central value fit

            if not hasattr(self,"cost_history"):
                self.cost_history = cost_history
            else:
                raise RuntimeError(f"FitResult {self}, already hast cost_history")
        else:

            if not hasattr(self,"cost_history_rspl"):
                self.cost_history_rspl = [None] * self.Nresample 
            self.cost_history_rspl[nres] = cost_history


    def import_from_iminuit(
        self,
        minuit: Any,
        Ndata: int,
        model: Callable,
        variable_projection: dict[str, float] | None = None,
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

        # --- Hessian (propagated) errors from Minuit ---
        # minuit.valid is True only after a successful migrad; errors are
        # meaningful only after hesse() has been called (or migrad has
        # estimated them).  We store them unconditionally and let the user
        # decide whether to trust them.
        for param in minuit.params:
            key = param.name
            if param.error is not None:
                self._ensure_hessian_err(key)
                if nres is None:
                    self.params_hessian_err[key].mean = param.error
                else:
                    self.params_hessian_err[key].rspl[nres] = param.error

        # --- variable-projection linear parameters ---
        if variable_projection is not None:
            for key, value in variable_projection.items():
                self._ensure_param(key)
                if nres is None:
                    self.params[key].mean = value
                else:
                    self.params[key].rspl[nres] = value

        # --- priors (central-value fit only) ---
        if nres is None and prior is not None:
            for key, p in prior.items():
                self.priors[key] = p

        # --- fit quality ---
        self._store_fit_quality(minuit.fval, nres)

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

        # --- fit quality ---
        # lsqfit stores p-value in nlf.Q; we re-compute from chi2 for
        # consistency with the other backends.
        self._store_fit_quality(float(nlf.chi2), nres)

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
        self._store_fit_quality(chi2, nres)
