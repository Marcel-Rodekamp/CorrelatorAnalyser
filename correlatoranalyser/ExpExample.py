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
import h5py

# ======================================================================================================
# Fit Model:
# C(t) = A0 * exp(-t E0) + ... + An * exp(-t (E0+..+ΔEn))
# with ΔE_n = E_n - E_{n-1}  =>  E_n = E_{n-1} + ΔE_n
#
class MultiState:
    def __init__(self, Nstates):
        self.Nstates = Nstates
        self.Nparam = 2 * Nstates

    def prior(self):
        # ToDO

        p = gv.BufferDict()

        # p["A0"] = gv.gvar(1e-9, 1e-8)
        p["A0"] = gv.gvar(0.5, 10)
        p["log(E0)"] = gv.log(gv.gvar(1, 10))

        for n in range(1, self.Nstates):
            p[f"A{n}"] = gv.gvar(0.5, 10)
            # p[f"A{n}"] = gv.gvar(1e-9, 1e-8)
            p[f"log(ΔE{n})"] = gv.log(gv.gvar(1, 10))

        return p

    def __call__(self, t: np.ndarray, p: gv.BufferDict) -> np.ndarray:
        E = p["E0"]
        out = p[f"A{0}"] * np.exp(-t * E)

        for n in range(1, self.Nstates):
            #    ΔE_n = E_n - E_{n-1}
            # =>  E_n = E_{n-1} + ΔE_n
            E += p[f"ΔE{n}"]
            out += p[f"A{n}"] * np.exp(-t * E)

        return out


# =======================================================================================================

Nt = 16
Nbst = 200
Nconf = 200
num_states = 2  # number of states
abscissa: np.ndarray = np.arange(0, Nt)
# data: np.ndarray[gv.GVar] = gv.gvar(
#     np.exp(-0.2 * abscissa) + 0.1 * np.exp(-0.35 * abscissa),
#     0.05 * np.exp(0.001 * abscissa),
# )

data: np.ndarray[gv.GVar] = gv.gvar(
    np.random.normal(
        np.exp(-0.2 * abscissa),  # + 0.1 * np.exp(-0.35 * abscissa),
        0.01 * np.exp(0.1 * abscissa),
        size=(Nt),
    ),
    0.05 * np.exp(0.1 * abscissa),
)
data2 = np.random.normal(
    np.exp(-0.2 * abscissa),  # + 0.1 * np.exp(-0.35 * abscissa),
    0.01 * np.exp(0.1 * abscissa),
    size=(Nconf, Nt),
)
data_bst = np.zeros((Nbst, Nt))
for nbst in range(Nbst):
    data_bst[nbst] = np.mean(data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0)


plt.plot(abscissa, gv.mean(data), marker=".", ls="", color="b")
for nbst in range(Nbst):
    plt.plot(abscissa, gv.mean(data_bst[nbst]), marker="x", ls="")

# plt.errorbar(x=abscissa,y=gv.mean(data),yerr=gv.sdev(data2[0]),linestyle='')
plt.plot(abscissa, 1 * np.exp(-abscissa * 0.2))
plt.yscale("log")
plt.savefig("./Report/test/initial_data.png")

res = FitState()
te = abscissa[-1]

# Do fits for all different nstates:
for ns in range(1, num_states + 1):
    model = MultiState(Nstates=ns)
    prior = model.prior()
    # ToDo: refresh priors
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
            resample_fit=True,
            resample_type="bst",
            # bootstrap_fit_resample_prior=False,
            # resample_fit_correlated=True,
            central_value_fit=False
        )
        # print(update.prior_res)
        res.append(update)
res.model_average()


#dummp into h5 file:
with h5py.File("./Report/FitResult.h5", "w") as h5f:
            res.serialize_all(h5_file=h5f)

# res2 = FitState()
# with h5py.File("./Report/FitResult.h5", "r") as h5f:
#             # res.serialize_all(h5_file=h5f)
#     res2.deserialize_all(h5_file=h5f)

# =========================================================================================================================================================================================
# Testing the fit
if res.fit_results[0].best_fit_param is not None:
    print("Parameters from the model average (CV):", res.param_avg)
    for fit in res.fit_results:
        print("Chi2/dof:", fit.chi2 / fit.dof)

    x_fine = np.arange(0, Nt, step=0.2)
    y_data = res.param_avg["est"]["A0"] * np.exp(-x_fine * res.param_avg["est"]["E0"])
    # y_data = res.param_avg['est']['A1']*np.exp(-x_fine*res.param_avg['est']['E1'])
    y_data += res.param_avg["est"]["A1"] * np.exp(
        -x_fine * (res.param_avg["est"]["E0"] + res.param_avg["est"]["ΔE1"])
    )
    plt.plot(x_fine, y_data, label="CV")
    plt.legend()
    plt.yscale("log")
    plt.savefig("./Report/test/CV_Fit.png", dpi=300)
    # plt.show()


# check chi2 of resample fit:
if res.fit_results[0].Nres is not None:  # check if redsample fit
    bad_fits = 0
    for i, fit in enumerate(res.fit_results):
        print(np.mean(fit.AIC_res))
        # print(fit.chi2_res[0])
        # print(fit.dof)
        for nbst in np.arange(0, Nbst):
            if fit.chi2_res[nbst] / fit.dof > 10:
                if (fit.te - fit.ts) > 10:
                    # print(
                    #     f"Fit no {i} nbst {nbst} with {fit.ts}, {fit.te} dof: {fit.dof} chi2/dof: {fit.chi2_res[nbst]/fit.dof}"
                    # )
                    bad_fits += 1
    print(f"No. of fits with chi2/dof >10: {bad_fits} of {nbst*len(res.fit_results)}")

if res.fit_results[0].Nres is not None:  # check if resample fit
    # print(res.fit_results/)
    x_fine = np.arange(0, Nt, step=0.2)
    for nbst in np.arange(Nbst):  # res.param_avg['res']:
        # print(param, np.mean(res.param_avg['res'][param]))
        y_data = res.param_avg["res"]["A0"][nbst] * np.exp(
            -x_fine * res.param_avg["res"]["E0"][nbst]
        )
        y_data += res.param_avg["res"]["A1"][nbst] * np.exp(
            -x_fine
            * (res.param_avg["res"]["E0"][nbst] + res.param_avg["res"]["ΔE1"][nbst])
        )
        plt.plot(x_fine, y_data, ls="--")
    y_data = np.mean(res.param_avg["res"]["A0"]) * np.exp(
        -x_fine * np.mean(res.param_avg["res"]["E0"])
    )
    # y_data += np.mean(res.param_avg['res']['A1'])*np.exp(-x_fine*(np.mean(res.param_avg['res']['E0'])+np.mean(res.param_avg['res']['ΔE1'])))
    plt.plot(x_fine, y_data, ls="-", label="mean")
    plt.yscale("log")
    plt.legend()
    plt.savefig("./Report/test/Res_Fit.png", dpi=300)
    plt.figure()
    plt.plot(x_fine, y_data, ls="-", label="mean of bst model avg")
    plt.plot(x_fine, 1 * np.exp(-x_fine * 0.2), label="'True' data")
    plt.legend()
    plt.yscale("log")
    plt.savefig("./Report/test/Res_Mean.png", dpi=300)


# ===========================================================================================================================================================================================


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


# =======================================
# Test Plotting:
# =======================================
# old:
# plots = DataPlotter(data_complete=res)
# fig,ax = DataPlotter.plotTopFits(plots,C= data,no_fits=5)

# new:
# multidim:
# plot = plot_best_fits(fit_state=res,C=np.stack((data,data),axis=0))
# #one-dim:


# plot_fig, plot_ax = plot_best_fits(fit_state=res, num_fits=6, C=data)
# plot_best_fits_resample_mean(fit_state=res, num_fits=6, C=data)
# costumization of the fit can be done further:
# plot_ax.set_title("test")
# plt.savefig('./Report/plotTopFits.png', dpi=300)
