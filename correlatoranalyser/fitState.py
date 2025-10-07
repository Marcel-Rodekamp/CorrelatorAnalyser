import numpy as np

import gvar as gv

import h5py

from dataclasses import dataclass, field, fields

from pathlib import Path

from typing import Self, List, Dict

from .fitResult import FitResult

import re

import warnings

from typing import Any, Dict

def _read_ds(ds: h5py.Dataset):
    """Read an h5py dataset, decoding strings when needed."""
    # h5py>=3: this returns str instead of bytes for string datasets
    try:
        return ds.asstr()[()]
    except Exception:
        data = ds[()]
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8")
        if isinstance(data, np.ndarray) and data.dtype.kind in ("S", "O"):
            return np.array(
                [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in data],
                dtype=object,
            )
        return data


@dataclass
class FitState:
    
    # list of fit results: Defining the state of the fitter
    fit_results: List[FitResult] = field(default_factory=lambda: [])  
    
    # collection of keys in the state
    keys_all: list[str] = field(default_factory=list[str])
    
    # a dictionary to store the model averaged params
    # This dictionary will be filled when model_average is called
    param_avg: dict = field(default_factory=dict)

    def append(self, new_fit: FitResult) -> None:
        self.fit_results.append(new_fit)
        # ToDO: add all parameters from the fit to the keys list
        try:
            new_keys = list(new_fit.best_fit_param.keys())
            self.keys_all += [key for key in new_keys if key not in self.keys_all]
            self.fit_results.sort(key=lambda x: x.AIC)  # sort by AIC
        except:
            pass
        try:
            new_keys = list(new_fit.best_fit_param_res.keys())
            self.keys_all += [key for key in new_keys if key not in self.keys_all]
            self.fit_results.sort(key=lambda x: x.AIC_res)  # sort by AIC
        except:
            pass

        return

    def model_average(self, keys: str | list[str] = None) -> dict:
        N = len(self.fit_results)

        def avg_single_key(key: str) -> None:
            # prepare arrays for data:
            param_est = np.empty(N)
            param_err = np.empty(N)
            AIC_est = np.empty(N)

            resamples_available = self.fit_results[0].has_resamples()

            if resamples_available:
                Nres = self.fit_results[0].Nres
                AIC_res = np.empty((Nres, N))
                param_res = np.empty((Nres, N))
            
            # if a key is not available in a fit, we need to remove it from the average
            not_available_fits = []

            # collect the data from the state
            for fitID, fit in enumerate(self.fit_results):
                params = fit.result_params()

                if key not in params["est"].keys():
                    not_available_fits.append(fitID)
                    continue

                param_est[fitID] = params["est"][key]
                param_err[fitID] = params["err"][key]
                AIC_est[fitID]   = fit.AIC

                if resamples_available:
                    param_res[:,fitID] = gv.mean(params["res"][key])
                    AIC_res[:,fitID]   = fit.AIC_res
            
            # Delete the entries for which there is no parameter
            param_est = np.delete(param_est, not_available_fits)
            param_err = np.delete(param_err, not_available_fits)
            AIC_est = np.delete(AIC_est, not_available_fits)

            if resamples_available:
                param_res = np.delete(param_res, not_available_fits, axis = 1)
                AIC_res = np.delete(AIC_res, not_available_fits, axis = 1)

            if not self.param_avg:
                self.param_avg["est"] = {}
                self.param_avg["err"] = {}

            # do model average for key for central value fit results
            if self.fit_results[0].has_central_value():
                # 1) calculate weights from the AIC, where the AIC are normalized s.t. the smallest AICs = 0
                # This has no effect on the outcome but can prevent overflows for very negative AICs
                weights_est = np.exp(-0.5 * (AIC_est - np.min(AIC_est)))
                # we normalize here already in order to have the normalized weight in the uncertainty computation
                weights_est /= np.sum(weights_est)
                # 2) the model average is now simply the weighted average of the parameters
                # \bar{p} = (\sum_{i=0}^{num_fits} p_i w_i) / (\sum_{i=0} w_i)
                modelAvg_est = np.average(param_est, weights=weights_est)
                # 3) finally we express the error of parameters by its weighted average according to
                # Δ\bar{p}^2 = \sum_{i=0}^{num_fits} (Δp_i w_i / (\sum_{j}^{num_fits} w_j) )^2
                # TODO: This ignores correlation between the fit results
                modelAvg_err = np.sqrt(np.sum(param_err**2 * weights_est**2))

                # finally store the result in the output array
                self.param_avg["est"][key] = modelAvg_est
                self.param_avg["err"][key] = modelAvg_err

            # model averaging for bootstrap fit results
            if self.fit_results[0].has_resamples():
                if not self.param_avg:
                    self.param_avg["res"] = {}
                elif "res" not in self.param_avg.keys():
                    self.param_avg["res"] = {}
                
                # Steps are similar as above, for the central value results, but extended to every sample result:
                # 1) calculate weights
                weights_res = np.exp(-0.5 * (AIC_res - np.min(AIC_res,axis=1)[:,None]))
                weights_res /= np.sum(weights_res, axis = 1)[:,None]
                # 2) model average
                modelAvg_res = np.average(param_res, weights=weights_res, axis=1)
                # 3) calculate the uncertainty. This is changed as we don't rely on error propagation but instead 
                #    compute the combined bootstrap + model average error.
                #    This is in principle a very conservative estimate as it captures uncertainties on the model.
                # This potentially overwrites the error above. This IS intended as by default the bootstrap uncertainty is
                # more reliable than the simple error propagation!
                if self.fit_results[0].resample_type == 'bst':
                    self.param_avg["err"][key] = np.std(modelAvg_res,axis=0)
                elif self.fit_results[0].resample_type == 'jkn':
                    self.param_avg["err"][key] = np.sqrt(Nres-1) * np.std(modelAvg_res,axis=0)
                # if no central value fit is done we simply compute the mean over bootstrap fits
                # these two values are equal provided, same fitting strategy!
                if self.fit_results[0].AIC is None:  # check if central value fit
                    self.param_avg["est"][key] = np.mean(modelAvg_res, axis = 0)
                # finally the bootstrap results will be stored too:
                self.param_avg["res"][key] = modelAvg_res
        # end def avg_single_key

        # go through all parameters and do model averaging for each of them:
        if isinstance(keys, list):  # if argument was given
            for key in keys:
                if key not in self.keys_all:
                    raise KeyError(
                        f'The given key "{key}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                    )
                avg_single_key(key=key)

            out = {}
            for res_type_key in ["est","err","res"]:
                if res_type_key not in self.param_avg.keys(): continue
                out[res_type_key] = {key:self.param_avg[res_type_key][key] for key in keys} 
            return out

        elif isinstance(keys, str):  # if argument was given
            if keys not in self.keys_all:
                raise KeyError(
                    f'The given key "{keys}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                )
            
            avg_single_key(key=keys)

            out = {}
            for res_type_key in ["est","err","res"]:
                if res_type_key not in self.param_avg.keys(): continue
                out[res_type_key] = self.param_avg[res_type_key][keys] 
            return out
        else:
            for key in self.keys_all:
                avg_single_key(key=key)
            return self.param_avg

    def __getitem__(self, index: int) -> FitResult:
        """method that gets a fit result based on the index in the fit_result list, list is sorted by AIC"""
        return self.fit_results[index]

    def __len__(self):
        """
        Returns the number of fits which are being tracked.
        """

        return len(self.fit_results)

    # dumps all information of the FitState in a h5 File
    def serialize_all(
        self, h5_file: h5py.File
    ) -> None:
        if not self.fit_results:
            raise Warning(f"Import FitResults first with {type(self).__name__}.append(new_fit : FitResult) before saving in an h5 file.")

        for field in fields(self):
            field_value = getattr(self, field.name)
            # print(field.name,field_value)
            # save the fit results using the serialize method from FitResult class
            if field.name == "fit_results":
                for i, fit in enumerate(field_value):
                    fit.serialize(h5_handle=h5_file, node=f"FitResults/Fit{i}")

            # save the keys
            elif field.name == "keys_all":
                h5_file.create_dataset(f"ModelAverage/KeysList", data=field_value)

            # save the averaged parameters
            elif field.name == "param_avg":
                for item in field_value:
                    for key in field_value[item]:
                        h5_file.create_dataset(
                            f"ModelAverage/Parameters/{item}/{key}",
                            data=field_value[item][key],
                        )
        return

    def deserialize_all(self, h5_file: h5py.File):
        if "FitResults" in h5_file:
            fr_grp = h5_file["FitResults"]

            def _fit_idx(name: str) -> int:
                m = re.search(r"(\d+)$", name)
                return int(m.group(1)) if m else -1

            for node_name in sorted(fr_grp.keys(), key=_fit_idx):
                node_path = f"FitResults/{node_name}"
                self.fit_results.append(FitResult.deserialize(h5_handle=h5_file, node=node_path))
        else:
            raise RuntimeError("No 'FitResults' group found")

        # ---------- ModelAverage/KeysList ----------
        self.keys_all = []
        try:
            keys = _read_ds(h5_file["ModelAverage/KeysList"])
            if isinstance(keys, np.ndarray):
                self.keys_all = keys.tolist()
            elif np.isscalar(keys):
                self.keys_all = [keys.item() if hasattr(keys, "item") else keys]
            else:
                # iterable but not str/bytes -> list; str -> wrap
                self.keys_all = list(keys) if not isinstance(keys, (str, bytes)) else [keys]
        except KeyError:
            warnings.warn("No 'ModelAverage/KeysList' dataset found; leaving `keys_all` empty.", RuntimeWarning)

        # ---------- ModelAverage/Parameters/<item>/<key> ----------
        self.param_avg: Dict[str, Dict[str, Any]] = {}
        try:
            params_grp = h5_file["ModelAverage/Parameters"]
            for item in params_grp.keys():
                inner: Dict[str, Any] = {}
                for key in params_grp[item].keys():
                    inner[key] = _read_ds(params_grp[item][key])
                self.param_avg[item] = inner
        except KeyError:
            warnings.warn("No 'ModelAverage/Parameters' group found; leaving `param_avg` empty.", RuntimeWarning)

        # Optional: mirror your save-time guard as a load-time sanity note
        if not self.fit_results:
            warnings.warn(
                f"No FitResults loaded. Ensure the HDF5 file contains groups like 'FitResults/Fit0', 'FitResults/Fit1', ...",
                RuntimeWarning,
            )


