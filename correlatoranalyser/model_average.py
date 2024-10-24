from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, List, Dict
from fit import FitResult
import numpy as np
import gvar as gv


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
    ) -> None:
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

            # do model average for key for central value fit results
            if self.fit_results[0].AIC is not None:  # check if central value fit
                weights_est = np.exp(-0.5 * (AIC_est - np.min(AIC_est)))
                modelAvg_est = np.average(param_est, weights=weights_est)
                modelAvg_err = np.average(param_err, weights=weights_est)
                self.param_avg[key + "_est"] = gv.gvar(modelAvg_est, modelAvg_err)
            # model averaging for bootstrap fit results
            if self.fit_results[0].Nbst is not None:  # check if bootstrap fit
                weights_bst = np.exp(-0.5 * (AIC_bst - np.min(AIC_bst)))
                modelAvg_bst = np.average(param_bst, weights=weights_bst, axis=1)
                modelAvg_bst_err = Nbst * [np.std(modelAvg_bst, axis=0)]
                self.param_avg[key + "_bst"] = gv.gvar(modelAvg_bst, modelAvg_bst_err)
            return

        # go through all parameters and do model averaging for each of them:
        if type(keys) is list:
            for key in keys:
                if key not in self.keys_all:
                    raise KeyError(
                        f'The given key "{key}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                    )
                avg_single_key(key=key)
        elif type(keys) is str:
            if keys not in self.keys_all:
                raise KeyError(
                    f'The given key "{keys}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                )
            avg_single_key(key=keys)
        else:
            for key in self.keys_all:
                avg_single_key(key=key)
        return

    def __get_item(index: int) -> FitResult:
        """method that gets a fit result based on the index in the fit_result list, list is sorted by AIC"""
        return
