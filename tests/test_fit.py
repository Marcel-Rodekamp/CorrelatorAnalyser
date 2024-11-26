import numpy as np

import gvar as gv

import lsqfit

import warnings

import pytest

from correlatoranalyser import fit, FitResult

def test_defensive(Nt: int = 32, Nres: int = 100):
    abscissa: np.ndarray = np.zeros(Nt)
    ordinate_est: np.ndarray[gv.GVar] = np.zeros(Nt, dtype=gv.GVar)
    ordinate_std: np.ndarray[gv.GVar] = np.zeros(Nt, dtype=gv.GVar)
    ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nt, Nt), dtype=gv.GVar)

    resample_ordinate_est: np.ndarray[gv.GVar] = np.zeros((Nres, Nt), dtype=gv.GVar)
    resample_ordinate_std: np.ndarray[gv.GVar] = np.zeros((Nres, Nt), dtype=gv.GVar)
    resample_ordinate_cov: np.ndarray[gv.GVar] = np.zeros(
        (Nres, Nt, Nt), dtype=gv.GVar
    )

    # tesing arguments for uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(abscissa=abscissa, ordinate_est=ordinate_est, ordinate_std=ordinate_std)
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(abscissa=abscissa, ordinate_est=ordinate_est, ordinate_cov=ordinate_cov)
    # 3. partially wrong (resample arguments instaed of central value arguments)
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_std=resample_ordinate_std,
    )
    # 4. partially wrong (resample arguments instead of central value and covariance instead of variance)
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
    )
    # 5. wrong arguments
    with pytest.raises(TypeError) as E:
        fit()  # -> req. abscissa
    assert E.type is TypeError
    with pytest.raises(ValueError) as E:
        fit(abscissa=abscissa)  # -> req. ordinate_est
    assert E.type is ValueError

    # testing arguments for correlated central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        central_value_fit_correlated=True,
    )
    # 2. partially wrong (resample arguments instaed of central value arguments)
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        central_value_fit_correlated=True,
    )
    # 3. wrong argumuments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            ordinate_std=ordinate_std,
            central_value_fit_correlated=True,
        )  # -> req. ordinate_cov
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            central_value_fit_correlated=True,
        )  # -> req. ordinate_cov
    assert E.type is ValueError

    # testing arguments for uncorr. resample fit, no central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_std=resample_ordinate_std,
        central_value_fit=False,
        resample_fit=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        central_value_fit=False,
        resample_fit=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa, central_value_fit=False, resample_fit=True
        )  # -> req. resample_ordinate_est
    assert E.type is ValueError

    # testing arguments for correlated resample fit, no central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        central_value_fit=False,
        resample_fit=True,
        resample_fit_correlated=True,
    )
    # 2. wrong argumuments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            resample_ordinate_std=resample_ordinate_std,
            central_value_fit=False,
            resample_fit=True,
            resample_fit_correlated=True,
        )  # -> req. resample_ordinate_cov
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            central_value_fit=False,
            resample_fit=True,
            resample_fit_correlated=True,
        )  # req. resample_ordinate_cov
    assert E.type is ValueError

    # tesing arguments for uncorr. resample fit and uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_std=ordinate_std,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_std=resample_ordinate_std,
        resample_fit=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        resample_fit=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa, ordinate_est=ordinate_est, resample_fit=True
        )  # -> req. resample_ordinate_est
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            resample_ordinate_est=resample_ordinate_est,
            resample_fit=True,
        )  # -> req. resample_ordinate_var (resample_ordinate_cov) and ordinate_var (ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            resample_ordinate_est=resample_ordinate_est,
            ordinate_std=ordinate_std,
            resample_fit=True,
        )  # -> req. resample_ordinate_var (resample_ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            resample_fit=True,
        )  # -> req. resample_ordinate_var (resample_ordinate_cov)
    assert E.type is ValueError

    # tesing arguments for corr. resample fit and uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_std=ordinate_std,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        resample_fit=True,
        resample_fit_correlated=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance for central value fit) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        resample_fit=True,
        resample_fit_correlated=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            resample_fit=True,
            resample_fit_correlated=True,
        )  # -> req. resample_ordinate_cov
    assert E.type is ValueError

    # testing arguments for uncorr. resample fit and corr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_std=resample_ordinate_std,
        resample_fit=True,
        central_value_fit_correlated=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance for central value fit) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        resample_fit=True,
        central_value_fit_correlated=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            resample_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. resample_ordinate_std (resample_ordinate_cov) and ordinate_cov
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            ordinate_cov=ordinate_cov,
            resample_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. resample_ordinate_var (resample_ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            resample_ordinate_est=resample_ordinate_est,
            resample_ordinate_std=resample_ordinate_std,
            resample_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. ordinate_cov
    assert E.type is ValueError

    # testing arguments for corr. resample fit and corr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        central_value_fit_correlated=True,
        resample_fit=True,
        resample_fit_correlated=True,
    )
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_std=ordinate_std,
        resample_ordinate_est=resample_ordinate_est,
        resample_ordinate_cov=resample_ordinate_cov,
        central_value_fit_correlated=True,
        resample_fit=True,
        resample_fit_correlated=True,
    )
    # 2. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            ordinate_cov=ordinate_cov,
            resample_ordinate_est=resample_ordinate_est,
            resample_ordinate_std=resample_ordinate_std,
            central_value_fit_correlated=True,
            resample_fit=True,
            resample_fit_correlated=True,
        )  # -> req. resample_ordinate_cov
    assert E.type is ValueError

if __name__ == "__main__":
    Nt = 16
    Nres = 100
    Nconf = 100
    # abscissa: np.ndarray = np.zeros(Nt)
    # ordinate_est: np.ndarray[gv.GVar] = np.zeros(Nt,dtype=gv.GVar)
    # ordinate_var: np.ndarray[gv.GVar] = np.zeros(Nt,dtype=gv.GVar)
    # ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nt,Nt),dtype=gv.GVar)

    # resample_ordinate_est: np.ndarray[gv.GVar] = np.zeros((Nres,Nt),dtype=gv.GVar)
    # resample_ordinate_var: np.ndarray[gv.GVar] = np.zeros((Nres,Nt),dtype=gv.GVar)
    # resample_ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nres,Nt,Nt),dtype=gv.GVar)

    # fit(
    #     abscissa = abscissa,
    #     ordinate_est = ordinate_est,
    #     resample_ordinate_est=resample_ordinate_est,
    #     resample_ordinate_cov=resample_ordinate_cov
    #     #ordinate_cov = ordinate_cov
    # )
    # test_defensive()

    abscissa: np.ndarray = np.arange(0, Nt)
    data: np.ndarray[gv.GVar] = gv.gvar(
        np.exp(-0.2 * abscissa), 0.1 * np.exp(0.001 * abscissa)
    )
    data2 = np.random.normal(
        np.exp(-0.2 * abscissa), 0.1 * np.exp(0.001 * abscissa), size=(Nconf, Nt)
    )
    data_res = np.zeros((Nres, Nt))
    for nres in range(Nres):
        data_res[nres] = np.mean(
            data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0
        )

    # plt.errorbar(abscissa,gv.mean(data2[0]),gv.sdev(data2[0]),capsize=2)
    # plt.yscale("log")
    # plt.show()
    # print(data_res.mean(axis=0))
    # print(data_res.std(axis=0))

    # #resample fit:
    # res =fit(
    #     abscissa = abscissa,
    #     resample_ordinate_est=data_res,
    #     resample_ordinate_cov=np.cov(data_res,rowvar=False),
    #     prior = {
    #         "E0": gv.gvar(0.5,100), #flat prior
    #         "A0": gv.gvar(0.5,100)
    #     },
    #     model= lambda t,p: p["A0"]*np.exp(-t*p["E0"]),
    #     resample_fit=True,
    #     resample_fit_resample_prior=False,
    #     central_value_fit=False,
    #     resample_fit_correlated=True
    # )

    # # # uncorrelated central value fit:
    # res = fit(
    #     abscissa=abscissa,
    #     ordinate_est=gv.mean(data),
    #     ordinate_var=gv.var(data),
    #     prior={"E0": gv.gvar(0.5, 100), "A0": gv.gvar(0.5, 100)},  # flat prior
    #     model=lambda t, p: p["A0"] * np.exp(-t * p["E0"]),
    # )

    # bootrtrap and cental value fit:
    res = fit(
        abscissa=abscissa,
        ordinate_est=gv.mean(data),
        ordinate_std=gv.sdev(data),
        resample_ordinate_est=data_res,
        resample_ordinate_cov=np.cov(data_res, rowvar=False),
        # prior = {
        #     "E0": gv.gvar(0.5,100), #flat prior
        #     "A0": gv.gvar(0.5,100)
        # },
        # p0={"E0": 0.5, "A0": 0.5},
        prior={
            "log(E0)": gv.log(gv.gvar(0.5, 100)),
            "log(A0)": gv.log(gv.gvar(0.5, 100)),
        },
        model=lambda t, p: p["A0"] * np.exp(-t * p["E0"]),
        resample_fit=True,
        resample_fit_resample_prior=False,
        resample_fit_correlated=True,
        central_value_fit=True,
    )
    # print(res.best_fit_param_res)
    # plt.errorbar(abscissa,gv.mean(data2[0]),gv.sdev(data2[0]),capsize=2)
    # # # plt.plot(abscissa,gv.mean(data),gv.sdev(data))

    # plt.plot(abscissa, gv.mean(res.best_fit_param['A0']) * np.exp(-abscissa * gv.mean(res.best_fit_param["E0"])))
    # # plt.yscale("log")
    # plt.show()
    # # print dict res:
    print(
        res.best_fit_param_res["E0"],
        gv.mean(res.best_fit_param_res["E0"]),
        gv.sdev(res.best_fit_param["E0"]),
    )
    # for key, value in res.items():
    #     print(f"{key}:{value}")
    # # plt.show()

# print(res.best_fit_param,res.best_fit_param_res,res.AIC_res)
