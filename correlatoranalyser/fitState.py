from __future__ import annotations

import re
import warnings
from typing import Literal

import h5py
import numpy as np

from .data import Data
from .fitResult import FitResult


# =============================================================================
# HDF5 helper
# =============================================================================

def _read_str_dataset(ds: h5py.Dataset) -> list[str]:
    """
    Read an HDF5 string dataset and return a plain list of str.

    Handles both h5py ≥ 3 (native str) and older byte-string datasets.
    """
    try:
        raw = ds.asstr()[()]
    except Exception:
        raw = ds[()]

    if isinstance(raw, (bytes, bytearray)):
        return [raw.decode("utf-8")]
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, np.ndarray):
        return [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in raw.flat
        ]
    return list(raw)


# =============================================================================
# FitState
# =============================================================================

class FitState:
    """
    Ordered collection of FitResult objects with model-averaging support.

    After each call to :meth:`append` the internal list is re-sorted so
    that the best-fit model is always at index 0.  Two sort criteria are
    supported:

    ``sort_by``
        ``'chi2_dof'`` — sort ascending by χ²/dof (default).
        ``'AIC'``      — sort ascending by AIC mean.

    Parameters
    ----------
    sort_by : {'chi2_dof', 'AIC'}
        Criterion used to order the stored fits.
    """

    def __init__(self, sort_by: Literal["chi2_dof", "AIC"] = "chi2_dof") -> None:
        if sort_by not in ("chi2_dof", "AIC"):
            raise ValueError(f"sort_by must be 'chi2_dof' or 'AIC', got '{sort_by}'.")

        self.sort_by: str              = sort_by
        self.fit_results: list[FitResult] = []
        self.keys_all:    list[str]    = []

    # ------------------------------------------------------------------
    # Collection interface
    # ------------------------------------------------------------------

    def append(self, fit: FitResult) -> None:
        """
        Add a FitResult and re-sort the collection.

        New parameter keys are merged into :attr:`keys_all` so that
        :meth:`model_average` can iterate over the full parameter set.
        """
        self.fit_results.append(fit)

        for key in fit.params:
            if key not in self.keys_all:
                self.keys_all.append(key)

        self._sort()

    def _sort(self) -> None:
        if self.sort_by == "chi2_dof":
            self.fit_results.sort(key=lambda r: r.chi2.mean / r.dof)
        else:
            self.fit_results.sort(key=lambda r: r.AIC.mean)

    def __len__(self) -> int:
        return len(self.fit_results)

    def __getitem__(self, index: int) -> FitResult:
        return self.fit_results[index]

    def __iter__(self):
        return iter(self.fit_results)

    def __repr__(self) -> str:
        lines = [f"FitState ({len(self)} fits, sort_by='{self.sort_by}'):"]
        for i, fit in enumerate(self.fit_results):
            chi2_dof = fit.chi2.mean / fit.dof if fit.dof else float("nan")
            keys     = ", ".join(fit.params.keys())
            lines.append(
                f"  [{i}]  χ²/dof={chi2_dof:.3g}  AIC={fit.AIC.mean:.3g}"
                f"  params=[{keys}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers shared by model_average and eval_model_avg
    # ------------------------------------------------------------------

    def _resample_config(self) -> tuple[str | None, int | None]:
        """Return the (resample_type, Nresample) of the first stored fit."""
        if not self.fit_results:
            raise RuntimeError("FitState is empty.")
        first = self.fit_results[0]
        return first.resample_type, first.Nresample

    def _aic_weights(
        self,
        aic_stack: Data,
    ) -> Data:
        """
        Compute normalised AIC weights from a (Nfits,) Data object.

        For the central value:
            w_i = exp(-0.5 * (AIC_i - min(AIC))) / sum_j w_j

        For resamples the same formula is applied per resample so that
        the weight distribution reflects resample-level uncertainty.

        Parameters
        ----------
        aic_stack : Data, shape (Nfits,)
            AIC values collected across all contributing fits.

        Returns
        -------
        Data, shape (Nfits,)
            Normalised weights with the same resample structure.
        """
        # Central-value weights
        aic_min_cv    = np.min(aic_stack.mean)
        weights_cv    = np.exp(-0.5 * (aic_stack.mean - aic_min_cv))
        weights_cv   /= weights_cv.sum()

        resample_type, Nresample = self._resample_config()
        weights = Data.zeros(
            resample_type=resample_type,
            shape=aic_stack.mean.shape,
            Nresample=Nresample,
            locked_mean=True,
        )
        weights.mean = weights_cv

        if resample_type is not None:
            for nres in range(Nresample):
                aic_rs   = aic_stack.rspl[nres]
                w_rs     = np.exp(-0.5 * (aic_rs - aic_rs.min()))
                weights.rspl[nres] = w_rs / w_rs.sum()

        return weights

    # ------------------------------------------------------------------
    # Model average of scalar parameters
    # ------------------------------------------------------------------

    def model_average(
        self,
        keys: str | list[str] | None = None,
        abscissa: np.ndarray | None = None,
    ) -> Data | dict[str, Data]:
        """
        Compute the AIC-weighted model average of fit parameters or model
        evaluations.

        Passing ``keys="fcn"`` (or calling :meth:`model_average_eval`
        directly) triggers function-evaluation averaging: every stored
        model is evaluated on *abscissa* and the results are combined with
        AIC weights, returning a single ``Data`` object of shape
        ``(len(abscissa),)``.

        For scalar parameters, fits that do not contain a requested key
        are silently excluded from that key's average.

        Parameters
        ----------
        keys : str | list[str] | None
            Parameter name(s) to average.  Use ``"fcn"`` to average model
            evaluations instead.  If ``None``, all parameters in
            :attr:`keys_all` are averaged and returned as a dict.
        abscissa : np.ndarray | None
            Required when ``keys="fcn"``; the points at which each model
            is evaluated before averaging.

        Returns
        -------
        Data
            If *keys* is a single string (including ``"fcn"``).
        dict[str, Data]
            If *keys* is a list or ``None``.
        """
        if not self.fit_results:
            raise RuntimeError("FitState is empty — nothing to average.")

        # ---- function-evaluation branch ----
        if keys == "fcn":
            if abscissa is None:
                raise ValueError(
                    "abscissa must be provided when keys='fcn'."
                )
            return self._eval_model_avg_impl(abscissa)

        # ---- scalar-parameter branch ----
        if isinstance(keys, str):
            self._check_key(keys)
            return self._avg_single_param(keys)

        if isinstance(keys, list):
            for key in keys:
                self._check_key(key)
            return {key: self._avg_single_param(key) for key in keys}

        # keys is None → average everything
        return {key: self._avg_single_param(key) for key in self.keys_all}

    def _check_key(self, key: str) -> None:
        if key not in self.keys_all:
            raise KeyError(
                f"'{key}' is not a known fit parameter.  "
                f"Available keys: {self.keys_all}"
            )

    def _avg_single_param(self, key: str) -> Data:
        """AIC-weighted average of a single parameter across all fits that contain it."""
        resample_type, Nresample = self._resample_config()

        contributing = [fit for fit in self.fit_results if key in fit.params]
        if not contributing:
            raise RuntimeError(
                f"No fit in this FitState contains the parameter '{key}'."
            )
        Nfits = len(contributing)

        # Collect parameter values and AIC into (Nfits,) Data objects.
        param_stack = Data.zeros(
            resample_type=resample_type,
            shape=(Nfits,),
            Nresample=Nresample,
            locked_mean=True,
        )
        aic_stack = Data.zeros(
            resample_type=resample_type,
            shape=(Nfits,),
            Nresample=Nresample,
            locked_mean=True,
        )

        for fit_id, fit in enumerate(contributing):
            param_stack.mean[fit_id] = fit.params[key].mean
            aic_stack.mean[fit_id]   = fit.AIC.mean
            if resample_type is not None:
                for nres in range(Nresample):
                    param_stack.rspl[nres][fit_id] = fit.params[key].rspl[nres]
                    aic_stack.rspl[nres][fit_id]   = fit.AIC.rspl[nres]

        weights = self._aic_weights(aic_stack)

        # Weighted average: scalar result.
        avg_cv = float(np.dot(weights.mean, param_stack.mean))

        out = Data.zeros(
            resample_type=resample_type,
            shape=None,
            Nresample=Nresample,
            locked_mean=True,
        )
        out.mean = avg_cv

        if resample_type is not None:
            for nres in range(Nresample):
                out.rspl[nres] = float(
                    np.dot(weights.rspl[nres], param_stack.rspl[nres])
                )

        return out

    # ------------------------------------------------------------------
    # Model average of function evaluations
    # ------------------------------------------------------------------

    def _eval_model_avg_impl(self, abscissa: np.ndarray) -> Data:
        """
        Evaluate each stored model on *abscissa* and return the
        AIC-weighted model average as a :class:`Data` object.

        Called by ``model_average(keys='fcn', abscissa=...)`` and by the
        convenience alias :meth:`model_average_eval`.

        The weighting follows the same scheme as :meth:`model_average`:
        for the central value (and independently for each resample) the
        weights are

            w_i ∝ exp(-0.5 * (AIC_i - min(AIC)))

        All fits must share the same ``resample_type`` and ``Nresample``.

        Parameters
        ----------
        abscissa : np.ndarray
            Points at which to evaluate the models.

        Returns
        -------
        Data, shape (len(abscissa),)
            Model-averaged function values with full resample information.
        """
        resample_type, Nresample = self._resample_config()
        Nfits = len(self.fit_results)
        Nout  = len(abscissa)

        # ---- collect per-fit evaluations and AIC into stacked arrays ----
        # eval_stack : shape (Nfits, Nout)  — central-value model curves
        # aic_stack  : shape (Nfits,)       — central-value AIC
        eval_stack_cv = np.empty((Nfits, Nout), dtype=float)
        aic_stack_cv  = np.empty((Nfits,),      dtype=float)

        # For resample fits we need (Nresample, Nfits, Nout) and (Nresample, Nfits).
        if resample_type is not None:
            eval_stack_rs = np.empty((Nresample, Nfits, Nout), dtype=float)
            aic_stack_rs  = np.empty((Nresample, Nfits),       dtype=float)

        for fit_id, fit in enumerate(self.fit_results):
            prediction = fit.eval(abscissa)      # returns Data, shape (Nout,)

            eval_stack_cv[fit_id] = prediction.mean
            aic_stack_cv[fit_id]  = fit.AIC.mean

            if resample_type is not None:
                for nres in range(Nresample):
                    eval_stack_rs[nres, fit_id] = prediction.rspl[nres]
                    aic_stack_rs[nres, fit_id]  = fit.AIC.rspl[nres]

        # ---- central-value weighted average ----
        w_cv = np.exp(-0.5 * (aic_stack_cv - aic_stack_cv.min()))
        w_cv /= w_cv.sum()
        # w_cv: (Nfits,)  ×  eval_stack_cv: (Nfits, Nout)  →  (Nout,)
        avg_cv = w_cv @ eval_stack_cv

        # ---- build output Data object ----
        out = Data.zeros(
            resample_type=resample_type,
            shape=(Nout,),
            Nresample=Nresample,
            locked_mean=True,
        )
        out.mean = avg_cv

        if resample_type is not None:
            for nres in range(Nresample):
                aic_rs  = aic_stack_rs[nres]               # (Nfits,)
                w_rs    = np.exp(-0.5 * (aic_rs - aic_rs.min()))
                w_rs   /= w_rs.sum()
                out.rspl[nres] = w_rs @ eval_stack_rs[nres]   # (Nout,)

        return out

    def model_average_eval(self, abscissa: np.ndarray) -> Data:
        """
        Convenience alias for ``model_average(keys='fcn', abscissa=abscissa)``.

        Parameters
        ----------
        abscissa : np.ndarray
            Points at which to evaluate the models.

        Returns
        -------
        Data, shape (len(abscissa),)
            Model-averaged function values with full resample information.
        """
        if not self.fit_results:
            raise RuntimeError("FitState is empty — nothing to evaluate.")
        return self._eval_model_avg_impl(abscissa)

    # ------------------------------------------------------------------
    # HDF5 serialisation
    # ------------------------------------------------------------------

    def serialize(self, h5_file: h5py.File) -> None:
        """
        Write the full FitState into an open HDF5 file.

        Layout
        ------
        ``FitResults/Fit0``, ``FitResults/Fit1``, …
            One group per FitResult, written by ``FitResult.serialize``.
        ``ModelAverage/KeysList``
            List of all parameter keys seen across all stored fits.
        ``FitState/sort_by``
            The sort criterion so the object reconstructs identically.

        Parameters
        ----------
        h5_file : h5py.File
            Open, writable HDF5 file handle.
        """
        if not self.fit_results:
            raise ValueError(
                "FitState is empty.  Add fits with .append() before serialising."
            )

        for fit_id, fit in enumerate(self.fit_results):
            fit.serialize(h5_handle=h5_file, node=f"FitResults/Fit{fit_id}")

        h5_file.create_dataset("ModelAverage/KeysList", data=self.keys_all)
        h5_file.create_dataset("FitState/sort_by",      data=self.sort_by)

    @staticmethod
    def deserialize(h5_file: h5py.File) -> FitState:
        """
        Reconstruct a FitState from an open HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            Open HDF5 file handle (read mode is sufficient).

        Returns
        -------
        FitState
        """
        # --- sort criterion ---
        sort_by = "chi2_dof"
        if "FitState/sort_by" in h5_file:
            raw = h5_file["FitState/sort_by"][()]
            sort_by = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        state = FitState(sort_by=sort_by)

        # --- fit results ---
        if "FitResults" not in h5_file:
            raise RuntimeError(
                "HDF5 file contains no 'FitResults' group.  "
                "Was it written by FitState.serialize()?"
            )

        fr_grp = h5_file["FitResults"]

        def _fit_index(name: str) -> int:
            m = re.search(r"(\d+)$", name)
            return int(m.group(1)) if m else -1

        for node_name in sorted(fr_grp.keys(), key=_fit_index):
            fit = FitResult.deserialize(h5_handle=h5_file, node=f"FitResults/{node_name}")
            # Append without re-sorting during load; sort once at the end.
            state.fit_results.append(fit)
            for key in fit.params:
                if key not in state.keys_all:
                    state.keys_all.append(key)

        # --- keys list (may contain keys from fits that lacked resamples) ---
        if "ModelAverage/KeysList" in h5_file:
            loaded_keys = _read_str_dataset(h5_file["ModelAverage/KeysList"])
            for key in loaded_keys:
                if key not in state.keys_all:
                    state.keys_all.append(key)
        else:
            warnings.warn(
                "No 'ModelAverage/KeysList' dataset found; keys_all rebuilt "
                "from the loaded FitResults.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not state.fit_results:
            warnings.warn(
                "No FitResults were loaded.  Check that the HDF5 file was "
                "written by FitState.serialize().",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            state._sort()

        return state
