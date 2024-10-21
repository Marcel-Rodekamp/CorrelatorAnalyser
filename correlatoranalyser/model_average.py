from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, List, Dict
from fit import FitResult


@dataclass
class Utility:
    fit_results: List[FitResult] = field(
        default_factory=lambda: []
    )  # list of fit results
    keys_all: list[str] = field(default_factory=list[str])
    param_avg: dict = field(default_factory=dict)
    # param_avg_bst: dict = field(default_factory=dict)

    def append(self, new_fit: FitResult) -> None:
        self.fit_results.append(new_fit)
        # ToDO: add all parameters from the fit to the keys list
        try:
            new_keys = list(new_fit.best_fit_param.keys())
            # print(new_keys)
            self.keys_all += [key for key in new_keys if key not in self.keys_all]
            self.fit_results.sort(key=lambda x: x.AIC)  # sort by AIC
        except:
            pass
        try:
            new_keys = list(new_fit.best_fit_param_bst.keys())
            # print(new_keys)
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

            return

        # go through all parameters and do model averaging for each of them:
        if type(keys) is list:
            for key in keys:
                avg_single_key(key=key)
        elif type(keys) is str:
            avg_single_key(key=key)
        else:
            for key in self.keys_all:
                avg_single_key(key=key)
        return

    def __get_item(index: int) -> FitResult:
        """method that gets a fit result based on the index in the fit_result list, list is sorted by AIC"""
        return
