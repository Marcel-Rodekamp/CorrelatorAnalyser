from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Self, List, Dict
from fit import FitResult
import numpy as np
import gvar as gv
import h5py


@dataclass
class FitState:
    fit_results: List[FitResult] = field(
        default_factory=lambda: []
    )  # list of fit results
    keys_all: list[str] = field(default_factory=list[str])
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
            new_keys = list(new_fit.best_fit_param_bst.keys())
            self.keys_all += [key for key in new_keys if key not in self.keys_all]
            self.fit_results.sort(key=lambda x: x.AIC_bst)  # sort by AIC
        except:
            pass

        return

    def model_average(
        self,
        keys: str | list[str] = None,
    ) -> dict:
        N = len(self.fit_results)

        def avg_single_key(key: str) -> None:
            # prepare arrays for data:
            param_est = np.zeros(N)
            param_err = np.zeros(N)
            AIC_est = np.zeros(N)
            if self.fit_results[0].Nbst is not None:
                Nbst = self.fit_results[0].Nbst
                AIC_bst = np.zeros((Nbst, N))
                param_bst = np.zeros((Nbst, N))
                param_bst_err = np.zeros((Nbst, N))

            for fitID, fit in enumerate(self.fit_results):
                # check if data is bootstraped or/and from central value fit:
                try:
                    bestParam = fit.best_fit_param[key]
                    param_est[fitID] = gv.mean(bestParam)
                    param_err[fitID] = gv.sdev(bestParam)
                    AIC_est[fitID] = fit.AIC
                    # cv_flag = True
                except TypeError:  # no central value fit, parameter is None
                    pass
                except KeyError:  # key is not in fit
                    pass
                try:
                    bestParam_bst = fit.best_fit_param_bst[key]
                    param_bst[:, fitID] = gv.mean(bestParam_bst)
                    param_bst_err[:, fitID] = gv.sdev(bestParam_bst)
                    AIC_bst[:, fitID] = fit.AIC_bst
                except TypeError:  # no bootstrap fit, parameter is None
                    pass
                except KeyError:  # key is not in fit
                    pass

            if not self.param_avg:
                self.param_avg["est"] = {}
                self.param_avg["err"] = {}

            # do model average for key for central value fit results
            if self.fit_results[0].AIC is not None:  # check if central value fit
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
            if self.fit_results[0].Nbst is not None:  # check if bootstrap fit
                if not self.param_avg:
                    self.param_avg["bst"] = {}
                elif "bst" not in self.param_avg:
                    self.param_avg["bst"] = {}

                # Steps are similar as above, for the central value results, but extended to every sample result:
                # 1) calculate weights
                weights_bst = np.exp(
                    -0.5 * (AIC_bst - np.min(AIC_bst, axis=1)[:, None])
                )
                weights_bst /= np.sum(weights_bst, axis=1)[:, None]
                # 2) model average
                modelAvg_bst = np.average(param_bst, weights=weights_bst, axis=1)
                # 3) calculate the uncertainty. This is changed as we don't rely on error propagation but instead
                #    compute the combined bootstrap + model average error.
                #    This is in principle a very conservative estimate as it captures uncertainties on the model.
                # This potentially overwrites the error above. This IS intended as by default the bootstrap uncertainty is
                # more reliable than the simple error propagation!
                self.param_avg["err"][key] = np.std(modelAvg_bst, axis=0)
                # if no central value fit is done we simply compute the mean over bootstrap fits
                # these two values are equal provided, same fitting strategy!
                if self.fit_results[0].AIC is None:  # check if central value fit
                    self.param_avg["est"][key] = np.mean(modelAvg_bst, axis=0)
                # finally the bootstrap results will be stored too:
                self.param_avg["bst"][key] = modelAvg_bst

        # end avg_single_key

        # go through all parameters and do model averaging for each of them:
        if isinstance(keys, list):  # if argument was given
            for key in keys:
                if key not in self.keys_all:
                    raise KeyError(
                        f'The given key "{key}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                    )
                avg_single_key(key=key)

            out = {}
            for res_type_key in ["est", "err", "bst"]:
                if res_type_key not in self.param_avg.keys():
                    continue
                out[res_type_key] = {
                    key: self.param_avg[res_type_key][key] for key in keys
                }
            return out

        elif isinstance(keys, str):  # if argument was given
            if keys not in self.keys_all:
                raise KeyError(
                    f'The given key "{keys}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                )
            avg_single_key(key=keys)

            out = {}
            for res_type_key in ["est", "err", "bst"]:
                if res_type_key not in self.param_avg.keys():
                    continue
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
        self, h5_file: h5py.File = h5py.File("../Report/FitState.h5", "w")
    ) -> None:
        # print(getattr(self,fit_results) )
        if not self.fit_results:
            raise Warning(
                f"Import FitResults first with {type(self).__name__}.append(new_fit : FitResult) before saving in an h5 file."
            )
        if not self.param_avg:
            raise Warning(
                f"Do model averaging first with {type(self).__name__}.model_average() before saving in an h5 file"
            )
        for field in fields(self):
            field_value = getattr(self, field.name)
            # print(field.name,field_value)
            # save the fit results using the serialize method from FitResult class
            if field.name == "fit_results":
                for i, fit in enumerate(field_value):
                    fit.serialize(h5_handle=h5_file, node=f"/FitResults/Fit{i}")
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
