from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, List, Dict
from fit import FitResult


@dataclass
class Utility:
    # list of fit results
    fit_results: List[FitResult] = field(default_factory=lambda: [])
    # num_states: int
    # keys_a
    param_avg: dict = field(default_factory=dict)
    param_avg_bst: dict = field(default_factory=dict)

    def append(self, new_fit: FitResult) -> None:
        self.fit_results.append(new_fit)
        # ToDo: sort by AIC
        self.fit_results.sort(key=lambda x: x.AIC)
        return

    def model_average(self, key: str = None) -> None:
        def avg_single_key(key: str) -> None:
            return

        def avg_single_key_bst(key: str) -> None:
            return

        return

    """method that gets a fit result based on the index in the list"""

    def __get_item(index: int) -> FitResult:
        return
