import numpy as np
import gvar as gv
from .fitState import FitState
from .fit import fit
import matplotlib.pyplot as plt
from .fitmodels import (
    SimpleSumOfExponentialsModel,
    SimpleSumOfExponentialsFlatPrior,
    SimpleSumOfExponentialsP0,
)

from .plotting.plotting_new import plot_best_fits, plot_best_fits_resample_mean

# from .ExpExample import data
Nt = 16
Nbst = 200
Nconf = 200
num_states = 2  # number of states
abscissa: np.ndarray = np.arange(0, Nt)
data: np.ndarray[gv.GVar] = gv.gvar(
    np.random.normal(
        np.exp(-0.2 * abscissa),  # + 0.1 * np.exp(-0.35 * abscissa),
        0.01 * np.exp(0.1 * abscissa),
        size=(Nt),
    ),
    0.05 * np.exp(0.1 * abscissa),
)


#import Fitstate from h5file:
res = FitState()
import h5py
with h5py.File("./Report/FitResult.h5", "r") as h5f:
    res.deserialize_all(h5_file=h5f)

plot_best_fits_resample_mean(fit_state=res, num_fits=2, C=data)