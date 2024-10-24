import numpy as np
import gvar as gv
from model_average import FitState
from fit import fit
import matplotlib.pyplot as plt
from fitModels import SimpleSumOfExponentialsModel, SimpleSumOfExponentialsFlatPrior


"""
ToDo: change model s.t. the order of parameters is sorted, with current Model only central value model averaging is working

 """
Nt = 16
Nbst = 100
Nconf = 100
num_states = 1  # number of states
abscissa: np.ndarray = np.arange(0, Nt)

real_abs = np.exp(-0.2 * abscissa)

data: np.ndarray[gv.GVar] = gv.gvar(
    np.random.normal(
        real_abs,  # + 0.1 * np.exp(-0.35 * abscissa),
        0.01 * np.exp(0.01 * abscissa),
        size=(Nt),
    ),
    0.05 * np.exp(0.1 * abscissa),
)
data2 = np.random.normal(
    real_abs,  # + 0.1 * np.exp(-0.35 * abscissa),
    0.01 * np.exp(0.1 * abscissa),
    size=(Nconf, Nt),
)

data_bst = np.zeros((Nbst, Nt))
for nbst in range(Nbst):
    data_bst[nbst] = np.mean(data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0)

plt.plot(abscissa, gv.mean(data), marker=".", ls="", color="b")
# plt.errorbar(abscissa,gv.mean(data2[0]),gv.sdev(data2[0]))
# plt.plot(abscissa, 1*np.exp(-abscissa*0.2)+0.1)
# plt.show()


res = FitState()
te = abscissa[-1]
for ns in range(1, num_states + 1):
    model = SimpleSumOfExponentialsModel(
        Nstates=ns
    )  # lambda t, p: p["A0"] * np.exp(-t * p["E0"])
    # ToDo: refresh priors
    prior_new = SimpleSumOfExponentialsFlatPrior(Nstates=ns)()
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
        """Make sure that A0>A1>...>An s.t. over the same parameter is averaged"""
        print("Before sorting:", update.best_fit_param)
        pairs = [
            (update.best_fit_param[f"E{i}"], update.best_fit_param[f"A{i}"])
            for i in range(ns)
        ]
        sorted_pairs = sorted(pairs, key=lambda x: gv.mean(x[0]), reverse=False)
        for i, (E, A) in enumerate(sorted_pairs):
            update.best_fit_param[f"E{i}"] = E
            update.best_fit_param[f"A{i}"] = A
        print("After sorting:", update.best_fit_param)

        # print(update.best_fit_param_bst)
        res.append(update)
        # print("all keys for averaging:", res.keys_all)
        print(f"...Averaging over {res.keys_all}")
        res.model_average()


print(f"Result Model Averaging:")
for key in res.keys_all:
    value = res.param_avg[key + "_est"]
    print(f"{key}_est: {value}")


print("Top 10 Fit Results:")
for i, fit in enumerate(res.fit_results):
    if False:  # i > 10:
        pass
    else:
        print(
            f"Fit {i}: range[{fit.ts},{fit.te}] with parameters {fit.best_fit_param} (cv)  and AIC {fit.AIC}\n"
        )

plt.plot(
    abscissa,
    real_abs,
    color="green",
    label="real test data",
)
plt.plot(
    abscissa,
    gv.mean(res.param_avg["A0_est"])
    * np.exp(-gv.mean(res.param_avg["E0_est"]) * abscissa),
    color="red",
    label="central value model average",
)

plt.legend()
plt.show()
