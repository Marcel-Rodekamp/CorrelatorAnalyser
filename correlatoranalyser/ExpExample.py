import numpy as np
import gvar as gv
from .fitState import FitState
from .fit import fit
import matplotlib.pyplot as plt
# from .fitmodels import (
#     SimpleSumOfExponentialsModel,
#     SimpleSumOfExponentialsFlatPrior,
#     SimpleSumOfExponentialsP0,
# )

from .plotting.plotting_new import plot_best_fits

#======================================================================================================
# Fit Model:
# C(t) = A0 * exp(-t E0) + ... + An * exp(-t (E0+..+ΔEn))
# with ΔE_n = E_n - E_{n-1}  =>  E_n = E_{n-1} + ΔE_n
#
class MultiState:
    def __init__(self, Nstates):
        self.Nstates = Nstates
        self.Nparam = 2 * Nstates

    def prior(self):
        #ToDO

        p = gv.BufferDict()

        p["A0"] = gv.gvar(1e-9, 1e-8)
        p["log(E0)"] = gv.log(gv.gvar(1,10))
        
        for n in range(1,self.Nstates):
            p[f"A{n}"] = gv.gvar(1e-9, 1e-8)
            p[f"log(ΔE{n})"] = gv.log(gv.gvar(1,10))

        return p
    
    def __call__(self, t:np.ndarray, p:gv.BufferDict) -> np.ndarray:
        E = p["E0"]
        out = p[f"A{0}"] * np.exp( -t*E )
        
        for n in range(1,self.Nstates):
            #    ΔE_n = E_n - E_{n-1}
            # =>  E_n = E_{n-1} + ΔE_n
            E += p[f"ΔE{n}"]
            out += p[f"A{n}"] * np.exp( -t*E )

        return out
#=======================================================================================================

Nt = 16
Nbst = 100
Nconf = 100
num_states = 2  # number of states
abscissa: np.ndarray = np.arange(0, Nt)
# data: np.ndarray[gv.GVar] = gv.gvar(
#     np.exp(-0.2 * abscissa) + 0.1 * np.exp(-0.35 * abscissa),
#     0.05 * np.exp(0.001 * abscissa),
# )

data: np.ndarray[gv.GVar] = gv.gvar(
    np.random.normal(
        np.exp(-0.2 * abscissa),  # + 0.1 * np.exp(-0.35 * abscissa),
        0.01 * np.exp(0.01 * abscissa),
        size=(Nt),
    ),
    0.05 * np.exp(0.1 * abscissa),
)
data2 = np.random.normal(
    np.exp(-0.2 * abscissa),  # + 0.1 * np.exp(-0.35 * abscissa),
    0.01 * np.exp(0.01 * abscissa),
    size=(Nconf, Nt),
)
data_bst = np.zeros((Nbst, Nt))
for nbst in range(Nbst):
    data_bst[nbst] = np.mean(data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0)


# plt.plot(abscissa, gv.mean(data), marker=".", ls="", color="b")
for nbst in range(Nbst):
    plt.plot(abscissa,gv.mean(data_bst[nbst]),marker="x",ls='')

# plt.errorbar(x=abscissa,y=gv.mean(data),yerr=gv.sdev(data2[0]),linestyle='')
plt.plot(abscissa, 1*np.exp(-abscissa*0.2))
plt.savefig("./Report/initial_data.png")

res = FitState()
te = abscissa[-1]

#Do fits for all different nstates:
for ns in range(1, num_states + 1):
    # model = SimpleSumOfExponentialsModel(
    #     Nstates=ns
    # )  # lambda t, p: p["A0"] * np.exp(-t * p["E0"])
    # prior_new = SimpleSumOfExponentialsFlatPrior(Nstates=ns)()
    # p0_new = SimpleSumOfExponentialsP0(Nstates=ns)()
    model = MultiState(Nstates=ns)
    # ToDo: refresh priors
    prior = model.prior()

    for ts in np.arange(0, te - 2 * ns - 2):
        print(f"...Fitting {ns} states, timeframe: [{ts},{te}]")
        update = fit(
            abscissa=abscissa[ts:te],
            ordinate_est=gv.mean(data[ts:te]),
            ordinate_std=gv.var(data[ts:te]),
            resample_ordinate_est=data_bst[:, ts:te],
            resample_ordinate_std=np.cov(data_bst[:, ts:te], rowvar=False),
            prior=prior,
            # p0={"E0": 0.5, "A0": 0.5},
            model=model,
            # p0=p0_new,
            resample_fit = True,
            resample_type='bst',
            # bootstrap_fit_resample_prior=False,
            # resample_fit_correlated=True
            central_value_fit=False,
        )
        # """Make sure that A0>A1>...>An s.t. over the same parameter is averaged"""
        # print("Before sorting:", update.best_fit_param)
        # pairs = [
        #     (update.best_fit_param[f"E{i}"], update.best_fit_param[f"A{i}"])
        #     for i in range(ns)
        # ]
        # sorted_pairs = sorted(pairs, key=lambda x: gv.mean(x[0]), reverse=False)
        # for i, (E, A) in enumerate(sorted_pairs):
        #     update.best_fit_param[f"E{i}"] = E
        #     update.best_fit_param[f"A{i}"] = A
        # print("After sorting:", update.best_fit_param)

        # print(update.best_fit_param_bst)
        res.append(update)
        # print("all keys for averaging:", res.keys_all)
        print(f"...Averaging over {res.keys_all}")
        res.model_average()
        # for key in res.keys_all:
        #     # print("central value:",res.param_avg[key+"_est"])
        #     print("bootstrap:", np.mean(res.param_avg[key + "_bst"]))
        # value=np.mean(res.param_avg[key+"_bst"])
        # print(f"np.mean({key}_bst)={np.mean(value)}")
        # print(f" {value[i]}" for enuemerate i, key in res.keys_all)
        # print(f"after averaging:{res.param_avg}")

import h5py
# with h5py.File("./Report/FitResult.h5", "w") as h5f:
#             res.serialize_all(h5_file=h5f)

# res.serialize_all(h5file=h5py.File("../Report/TestData.h5", "w"))

# print(f"Result Model Averaging:")
# for key in res.keys_all:
#     value = res.param_avg[key + "_est"]
#     print(f"{key}_est: {value}")

# print bootsrap results
# averaged_param = {}
# for key in res.keys_all:
#     value = np.mean(res.param_avg[key + "_bst"])
#     print(f"{key}_bst: {value}")
#     averaged_param[key]=gv.mean(value)

# print("Top 5 Fit Results:")
# for i, fit in enumerate(res.fit_results):
#     if i > 4:
#         pass
#     else:
#         print(
#             f"Fit {i+1}: range[{fit.ts},{fit.te}] with parameters {fit.best_fit_param} (cv)  and AIC {fit.AIC}\n"
#         )
# print(res)

# plt.plot(
#     abscissa,
#     1 * np.exp(-0.2 * abscissa) + 0.1 * np.exp(-0.35 * abscissa),
#     color="green",
#     label="real test data",
# )  # + 0.1 * np.exp(-0.35 * abscissa))
# for nbst in range(Nbst):
#     E0_bst = res.param_avg['E0_bst'][nbst]
#     # E1_bst = res.param_avg['E1_bst'][nbst]
#     A0_bst = res.param_avg['A0_bst'][nbst]
# A1_bst = res.param_avg['A1_bst'][nbst]
# plt.plot(abscissa,gv.mean(A0_bst)*np.exp(-gv.mean(E0_bst)*abscissa))#+ gv.mean(A1_bst)*  np.exp(-gv.mean(E1_bst)* abscissa))
# plt.plot(abscissa,averaged_param['A0']*np.exp(-averaged_param['E0']*abscissa),color='blue',label='mean of bootstrap model average')#+ averaged_param['A1']*  np.exp(-averaged_param['E1']* abscissa))
# plt.plot(
#     abscissa,
#     gv.mean(res.param_avg["A0_est"])
#     * np.exp(-gv.mean(res.param_avg["E0_est"]) * abscissa),
#     color="red",
#     label="central value model average",
# )
# plt.legend()
# plt.show()


#=======================================
# Test Plotting:
#=======================================
# old:
# plots = DataPlotter(data_complete=res)
# fig,ax = DataPlotter.plotTopFits(plots,C= data,no_fits=5)

#new:
#multidim:
# plot = plot_best_fits(fit_state=res,C=np.stack((data,data),axis=0))
#one-dim:
plot = plot_best_fits(fit_state=res,num_fits=3,C=data)
print(res.fit_results[0].best_fit_param, res.fit_results[0].best_fit_param)
# plt.savefig('./Report/plotTopFits.png', dpi=300)

# print(data[None,:])



