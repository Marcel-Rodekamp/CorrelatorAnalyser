from dataclasses import dataclass, field
from pathlib import Path
from typing import Selfm, List, Dict
from .fit import FitResult


@dataclass
class Utility:
    # list of fit results
    fit_results: List[FitResult] = field(default_factory=lambda: [])
    num_states: int

    def model_average(key: str = None) -> Dict:
        # do fit for different timeslices, different nstates with fit.py
        # sort by AIC value
        # calculate the final parameter
        return

    """method that gets a fit result based on the index in the list"""

    def __get_item(index: int) -> FitResult:
        return
