from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, List, Dict
from fit import FitResult
import numpy as np
import gvar as gv


@dataclass
class FitState:
    """Stores all FitResults in a list and averages over the
    parameters in the list keys, the FitResult list can be updated after a new find with the append method
    """

    # list of fit results
    fit_results: List[FitResult] = field(default_factory=lambda: [])
    # keys_all: dict#includes all parameters of the Fit
    param_avg: gv.BufferDict = field(default_factory=dict)
    param_avg_bst: gv.BufferDict = field(default_factory=dict)
    # num_states: int
    # ts: int
    # te: int

    def append(self, new_res: FitResult):
        self.fit_results.append(new_res)
        return

    def model_average(
        self, keys: List[str]
    ) -> (
        None
    ):  # keys argument can be set optional if it should not be averaged over all parameters
        def avg_single_key(key: str) -> None:
            N = len(self.fit_results)  # number of FitResults
            # prepare arrays for data:
            param_est = np.zeros(N)
            param_err = np.zeros(N)
            AIC_est = np.zeros(N)
            # read in the data to average:
            for fitID, fit in enumerate(self.fit_results):
                # check if the parameter is actually in the fit result
                try:
                    bestParam = fit.best_fit_param[key]
                except KeyError as e:
                    continue
            param_est[fitID] = gv.mean(bestParam)
            param_err[fitID] = gv.sdev(bestParam)
            AIC_est[fitID] = fit.AIC
            # compute the weights
            weights_est = np.exp(-0.5 * (AIC_est - np.min(AIC_est)))
            # compute model average and store it
            modelAvg_est = np.average(param_est, weights=weights_est)
            modelAvg_err = np.average(param_err, weights=weights_est)
            self.param_avg[key] = gv.gvar(modelAvg_est, modelAvg_err)
            return

        def avg_single_key_bst(key: str) -> None:
            N = len(self.fit_results)  # number of FitResults
            for fitID, fit in enumerate(self.fit_results):
                try:  # check if the parameter is actually in the fit result
                    bestParam_bst = fit.best_fit_param_bst[key]
                except KeyError as e:
                    continue
            Nbst = self.fit_results[0].Nbst
            AIC_bst = np.zeros((Nbst, N))
            param_bst = np.zeros((Nbst, N))
            param_bst_err = np.zeros((Nbst, N))

            bestParam_bst = fit.best_fit_param_bst[key]
            param_bst[:, fitID] = gv.mean(bestParam_bst)
            param_bst_err[:, fitID] = gv.sdev(bestParam_bst)
            AIC_bst[:, fitID] = fit.AIC_bst
            print(AIC_bst, np.min(AIC_bst))
            # compute weights:
            weights_bst = np.exp(-0.5 * (AIC_bst - np.min(AIC_bst)))
            print(weights_bst)
            # compute model average and store it
            modelAvg_bst = np.average(param_bst, weights=weights_bst, axis=1)
            modelAvg_err = np.std(modelAvg_bst, axis=0)
            self.param_avg_bst[key] = gv.gvar(modelAvg_bst, modelAvg_err)

        for key in keys:
            avg_single_key(key=key)
            if self.fit_results[0].Nbst is not None:
                avg_single_key_bst(key=key)
        return

    """method that gets a fit result based on the index in the list"""

    def __get_item(index: int) -> FitResult:
        return
