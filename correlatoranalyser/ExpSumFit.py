import numpy as np
import gvar as gv
from model_average import Utility
from fit import fit
import matplotlib.pyplot as plt
from fitModels import SimpleSumOfExponentialsModel, SimpleSumOfExponentialsFlatPrior

Nt = 16
Nbst = 100
Nconf = 100
num_states = 2  # number of states
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

# plt.errorbar(abscissa,gv.mean(data2[0]),gv.sdev(data2[0]))
# plt.plot(abscissa, 1*np.exp(-abscissa*0.2)+0.1)
# plt.show()
res = Utility()
te = abscissa[-1]
for ns in range(1, num_states + 1):
    model = SimpleSumOfExponentialsModel(
        Nstates=ns
    )  # lambda t, p: p["A0"] * np.exp(-t * p["E0"])
    # ToDo: refresh priors
    prior_new = SimpleSumOfExponentialsFlatPrior(Nstates=ns)()
    # print(prior_new)
    # prior_new = {"E0": gv.gvar(0.5,100), #flat prior
    #                 "A0": gv.gvar(0.5,100),
    #             "E1": gv.gvar(0.5,100), #flat prior
    #                 "A1": gv.gvar(0.5,100)}
    for ts in np.arange(0, te - 2 * ns - 2):
        update = fit(
            abscissa=abscissa[ts:te],
            ordinate_est=gv.mean(data[ts:te]),
            ordinate_var=gv.var(data[ts:te]),
            bootstrap_ordinate_est=data_bst[:, ts:te],
            bootstrap_ordinate_cov=np.cov(data_bst[:, ts:te], rowvar=False),
            prior=prior_new,
            # p0={"E0": 0.5, "A0": 0.5},
            model=model,
            # bootstrap_fit=True,
            bootstrap_fit_resample_prior=False,
            bootstrap_fit_correlated=True,
            # central_value_fit=False,
        )
        res.append(update)
        print("all keys for averaging:", res.keys_all)
        res.model_average()

for i, fit in enumerate(res.fit_results):
    print(
        f"Fit {i}: range[{fit.ts},{fit.te}] with parameters {fit.best_fit_param} and AIC {fit.AIC}\n"
    )
# print(res)
