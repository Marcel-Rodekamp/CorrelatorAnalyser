from pathlib import Path
from typing import Self, List, Dict
import fit
from model_average import FitState
import numpy as np
import gvar as gv


"""Do model averaging with an exponential model up to n states"""
Nt = 16
Nbst = 100
Nconf = 100
num_states = 1
priors = {"E0": 0.5, "A0": 0.5}
exp_model = lambda t, p: p["A0"] * np.exp(-t * p["E0"])

# test data:
abscissa: np.ndarray = np.arange(0, Nt)
data: np.ndarray[gv.GVar] = gv.gvar(
    np.exp(-0.2 * abscissa), 0.1 * np.exp(0.001 * abscissa)
)
data2 = np.random.normal(
    np.exp(-0.2 * abscissa), 0.1 * np.exp(0.001 * abscissa), size=(Nconf, Nt)
)
data_bst = np.zeros((Nbst, Nt))
for nbst in range(Nbst):
    data_bst[nbst] = np.mean(data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0)

# ---------------------------------------------------------------------------------------------
res = FitState()  # res contains the FitResults and the model average results


for ts in np.arange(1, 4):  # abscissa[-1] - 2 * num_states - 2):
    fit_res = fit.fit(
        abscissa=abscissa[ts:-1],
        ordinate_est=gv.mean(data[ts:-1]),
        ordinate_var=gv.var(data[ts:-1]),
        bootstrap_ordinate_est=data_bst[:, ts:-1],
        bootstrap_ordinate_cov=np.cov(data_bst[:, ts:-1], rowvar=False),
        p0=priors,
        model=exp_model,
        # bootstrap_fit=True,
        # bootstrap_fit_resample_prior=False,
        # bootstrap_fit_correlated=True,
        central_value_fit=True,
    )
    res.append(fit_res)
    res.model_average(["A0"])


print(res)

# for i,fitre in enumerate(res.fit_results):
#     print(i,fitre,"/n")
