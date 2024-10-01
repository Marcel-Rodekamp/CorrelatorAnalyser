import numpy as np
import gvar as gv
import matplotlib.pyplot as plt
import lsqfit
import warnings
import pytest
import h5py
from dataclasses import dataclass, fields


@dataclass
class FitResult:
    """ToDo"""

    ts: int  # startpoint abscissa
    te: int  # endpoint abscissa
    num_dof: int = None
    best_fit_param: gv.BufferDict | None = None
    chi2: float | None = None
    aug_chi2: float | None = None
    Q_value: float | None = None
    AIC: float | None = None
    aug_AIC: float | None = None
    # bootstrap fit results
    Nbst: int | None = None
    best_fit_param_bst: gv.BufferDict | None = None
    chi2_bst: np.ndarray | None = None
    aug_chi2_bst: np.ndarray | None = None

    Q_value_bst: np.ndarray | None = None
    AIC_bst: np.ndarray | None = None
    aug_AIC_bst: np.ndarray | None = None

    def __post_init__(self):
        if self.Nbst is None:
            return
        self.chi2_bst = np.zeros(Nbst)
        self.best_fit_param_bst = {}
        self.aug_chi2_bst = np.zeros(Nbst)
        self.Q_value_bst = np.zeros(Nbst)
        self.AIC_bst = np.zeros(Nbst)
        self.aug_AIC_bst = np.zeros(Nbst)

    def calc_AIC(self, nlf: lsqfit.nonlinear_fit, augmented: bool = False) -> float:
        if augmented:
            return nlf.chi2 + 2 * len(nlf.p) - 2 * len(nlf.x)
        correction: float = 0.0
        if nlf.prior is not None:  # if prior is none then AIC = aug_AIC
            correction = np.sum(
                [
                    (nlf.prior[key].mean - nlf.p[key].mean) ** 2
                    / nlf.prior[key].sdev ** 2
                    for key in nlf.prior.keys()
                ]
            )
        return nlf.chi2 + 2 * len(nlf.p) - 2 * len(nlf.x) - correction

    def calc_aug_chi2(self, nlf: lsqfit.nonlinear_fit) -> float:
        correction: float = 0.0
        if (
            nlf.prior is not None
        ):  # if prior is None, then the chi2 from lsqfit is already the priorless chi2
            correction = np.sum(
                [
                    (nlf.prior[key].mean - nlf.p[key].mean) ** 2
                    / nlf.prior[key].sdev ** 2
                    for key in nlf.prior.keys()
                ]
            )
        return nlf.chi2 - correction

    """function evaluates the fit model for a given abscissa and fit parameters, if no  fit parameters are given the best fit parameters from the fit are used, 
    returns an np.ndarray with the corresponding ordinate values"""
    # def eval_model(self,nlf: lsqfit.nonlinear_fit, abscissa: np.ndarray, params_dict: None | dict = None)-> np.ndarray:
    #     if params_dict is None:
    #         params_dict = self.p
    #     ordinate = nlf.fcn(abscissa,params_dict)
    #         # ordinate = np.print(nlf.fcn(10,nlf.p))
    #     print(abscissa,ordinate)
    #     return ordinate

    """save the interesting results from a lsqfit, if nbst is given then save in corresponding row nbst of the bootstrap parameters"""

    def import_from_nonlinear_fit(
        self, nlf: lsqfit.nonlinear_fit, nbst: None | int = None
    ) -> None:
        if nbst is not None:
            self.best_fit_param_bst[nbst] = nlf.p
            self.chi2_bst[nbst] = self.calc_aug_chi2(nlf)
            self.aug_chi2_bst[nbst] = nlf.chi2
            self.Q_value_bst[nbst] = nlf.Q
            self.AIC_bst[nbst] = self.calc_AIC(nlf)
            self.aug_AIC_bst[nbst] = self.calc_AIC(nlf, augmented=True)
            self.num_dof = nlf.dof
        else:
            self.num_dof = nlf.dof
            self.best_fit_param = nlf.p
            self.chi2 = self.calc_aug_chi2(nlf)
            self.aug_chi2 = nlf.chi2
            self.Q_value = nlf.Q
            self.AIC = self.calc_AIC(nlf)
            self.aug_AIC = self.calc_AIC(nlf, augmented=True)
        return

    # node: path in h5 file (specifies the ) h5_handel: h5File object
    def serialize(self, h5_handle: h5py.File, node: str) -> None:
        if self.best_fit_param is None and self.best_fit_param_bst is None:
            raise ValueError(
                f"Import data from a fit first before saving the data in an h5 file.)"
            )
        # for key, value in self.best_fit_param.items(): print(key,value)
        for field in fields(self):
            field_value = getattr(self, field.name)
            # print(f"{field.name} with value {field_value} field.type:{field.type} type(field_value): {type(field_value)} is None: {field_value is None } is dict: {type(field_value)==gv.BufferDict}")
            if type(field_value) == gv.BufferDict:  # for best_param
                for key, value in field_value.items():
                    # split gvar data in estimate(est) and error (err) for saving in h5 file
                    h5_handle.create_dataset(
                        f"{node}/{field.name}/{key}/est", data=gv.mean(value)
                    )
                    h5_handle.create_dataset(
                        f"{node}/{field.name}/{key}/err", data=gv.sdev(value)
                    )
            elif field_value is None:
                pass
                # h5_handle.create_dataset(
                #     f"{node}/{field.name}", data=field_value, shape=()
                # )
            elif (
                field.name == "best_fit_param_bst"
            ):  # type(field_value) == dict: #for best_fit_param_bst, when not None
                for nbst, nbst_param in field_value.items():
                    for key, value in nbst_param.items():
                        h5_handle.create_dataset(
                            f"{node}/{field.name}/{nbst}/{key}/est", data=gv.mean(value)
                        )
                        h5_handle.create_dataset(
                            f"{node}/{field.name}/{nbst}/{key}/err", data=gv.sdev(value)
                        )
            else:
                # print(f'save {field.name} in {h5_handle}/{node}/{field.name}')
                h5_handle.create_dataset(f"{node}/{field.name}", data=field_value)
        return

    # read in data from a fit results in form of a h5 file and save as an FitResult object
    @staticmethod
    def deserialize(h5_handle: h5py.File, node: str) -> "FitResult":
        # check if h5file contains bootstrap data:
        te = h5_handle[f"{node}/te"][()]
        ts = h5_handle[f"{node}/ts"][()]
        if "Nbst" in h5_handle[node]:
            Nbst = h5_handle[f"{node}/Nbst"][()]
            res = FitResult(te=te, ts=ts, Nbst=Nbst)
        else:
            res = FitResult(te=te, ts=ts)
        # read in the data:
        for key in h5_handle[node]:
            if isinstance(h5_handle[f"{node}/{key}"], h5py.Dataset):
                setattr(res, key, h5_handle[f"{node}/{key}"][()])
            elif isinstance(
                h5_handle[f"{node}/{key}"], h5py.Group
            ):  # best_fit_param and best__fit_param_bst
                if key == "best_fit_param":
                    key_value = gv.BufferDict()
                    for item in h5_handle[f"{node}/{key}"]:  #'A0',E0' etc.
                        # print(item)
                        est = h5_handle[f"{node}/{key}/{item}/est"][()]
                        err = h5_handle[f"{node}/{key}/{item}/err"][()]
                        key_value[item] = gv.gvar(est, err)
                    setattr(res, key, key_value)
                elif key == "best_fit_param_bst":
                    for nbst in h5_handle[f"{node}/{key}"]:  # iterate over bootstrap id
                        key_value = gv.BufferDict()
                        for item in h5_handle[f"{node}/{key}/{nbst}"]:  #'A0',E0' etc.
                            # print('bootstrap',nbst,item)
                            est = h5_handle[f"{node}/{key}/{nbst}/{item}/est"][()]
                            err = h5_handle[f"{node}/{key}/{nbst}/{item}/err"][()]
                            key_value[item] = gv.gvar(est, err)
                        # print(key_value)
                        # setattr(res,key[int(nbst)],key_value)
                        res.best_fit_param_bst[int(nbst)] = key_value
                        # print(getattr(res,key),int(nbst))
        # print('After deserialize: ',res)
        return res

        # for key in self.best_fit_param:
        #          h5_handle.create_dataset(f"{node}/best_fit_param/{key}",data = self.best_fit_param[key])
        # # for key in self.best_fit_param:
        #     #print(key)
        # for attribute, value in self.__dict__.items():
        #     # print("serialize funcition")
        #     if type(value) is dict:
        #         print(value)
        #         for key in value:
        #             print(key,value[key])
        #         # prin
        #         # t("dict")
        #         # print(attribute, '=', value)
        #         # pass

        #         #     h5_handle.create_dataset(f"{node}/{attribute}/{item}",data = item_value)
        # else:
        #     h5_handle.create_dataset(f"{node}/{attribute}",data = value)

    # def eval_fit_model(abscissa:np.ndarray, params:dict = None)->np.ndarray:


def fit(
    *,
    abscissa: np.ndarray,
    ordinate_est: np.ndarray[gv.GVar] | None = None,
    ordinate_var: np.ndarray[gv.GVar] | None = None,
    ordinate_cov: np.ndarray[gv.GVar] | None = None,
    bootstrap_ordinate_est: np.ndarray[gv.GVar] | None = None,
    bootstrap_ordinate_var: np.ndarray[gv.GVar] | None = None,
    bootstrap_ordinate_cov: np.ndarray[gv.GVar] | None = None,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    bootstrap_fit: bool = False,
    bootstrap_fit_correlated: bool = False,
    bootstrap_fit_resample_prior: bool = True,
    # args for lsqfit:
    model: callable = None,
    prior: dict = None,
    p0: dict = None,
    svdcut: float = None,
    eps: float = None,
    maxiter: int = 10_000,
) -> FitResult:
    r"""!
    @param abscissa: datapoints for the x-axis (i.e. an array containing Nt times (shape (Nt,))
    @param ordinate_est: datapoints for the y-axis (i.e. an array containing the datapoints measured at Nt times (shape (Nt,))
    @param ordinate_var: variance of the given y datapoints (i.e. an array containing the error of the measured datapoints (shape (Nt,))
    @param ordinate_cov: covariance matrix of the y datapoints to specify the correlation betweem them
    @param bootstrap_ordinate_est: datapoints for the y-axis for each bootstrap (array of shape (Nbst,Nt))
    @param bootstrap_ordinate_var: variance of the datapoints for each bootstrap (array of shape (Nbst,Nt) or (Nt,), the latter uses the given variance for all bootstraps)
    @param bootstrap_ordinate_cov: covariance matrix of the datapoints for each bootstrap (array of shape (Nbst,Nt,Nt) or (Nt,Nt), the latter uses the given covariance matrix for all bootstraps)

    @param central_value_fit: option whether a central value fit should be performed (default: True)
    @param central_value_fit_correlated: option whether correlated fit should be performed (default: False)
    @param bootstrap_fit: option whether a bootstrap fit should be performed (delfault: False)
    @param bootstrap_ft_corrrelated: option whether a correlated bootstrap fit should be performed (default: False)
    @param bootstrap_fit_resample_prior: option whether the meanvalue of the prior should be resampled for each bootstrap, without resampling (option 'False') the same meanvalue for the prior is used for all bootstraps (default: True)

    @param model: the function to be fit to the datapoints, arguments should be the abscissa and the fit parameters
    @param prior: a priori estimates for the fit parameters (default: None)
    @po: start value for the fit parameters (default:None)
    @maxiter: the maximum of iterations to perform the fit (default: 10_000)
    This function peforms a fit using lsqfit by Peter Lapage. Default is an uncorrelated central value fit.
    Optional are a correlated central value fit and a correlated (uncorrelated) bootstrap fit. The the best parameters for the fit are then returned
    """

    # check if all necessary arguments are passed:
    if not (central_value_fit or bootstrap_fit):
        raise ValueError(
            f"at least one fit strategy needs to be defined: central_value_fit or bootstrap_fit"
        )

    if central_value_fit:
        if ordinate_est is None and bootstrap_ordinate_est is None:
            raise ValueError(
                f"central value fit requires ordinate_est (alternatively bootstrap_ordinate_est)"
            )
        if (
            ordinate_var is None
            and ordinate_cov is None
            and bootstrap_ordinate_var is None
            and bootstrap_ordinate_cov is None
        ):
            raise ValueError(
                f"central value fit requires at least one: ordinate_var or ordinate_cov (alternatively bootstrap_ordinate_var or bootstrap_ordinate_cov)"
            )
        if (
            central_value_fit_correlated
            and ordinate_cov is None
            and bootstrap_ordinate_cov is None
        ):
            raise ValueError(
                f"central value fit (correlated) requires ordinate_cov (alternativly bootstrap_ordinate_cov)"
            )
        if (
            not central_value_fit_correlated
            and ordinate_var is None
            and bootstrap_ordinate_var is None
        ):
            warnings.warn(
                f"variance provided through covariance in uncorrelated central value fit"
            )

    if bootstrap_fit:
        if bootstrap_ordinate_est is None:
            raise ValueError(f"bootrstap fit requires bootstrap_ordinate_est")
        if bootstrap_ordinate_var is None and bootstrap_ordinate_cov is None:
            raise ValueError(
                f"bootstrap fit requires at least one: bootstrap_ordinate_var or bootstrap_ordinate_cov"
            )
        if bootstrap_fit_correlated and bootstrap_ordinate_cov is None:
            raise ValueError(
                f"bootstrap fit (correlated) requires bootstrap_ordinate_cov"
            )
        if not bootstrap_fit_correlated and bootstrap_ordinate_var is None:
            warnings.warn(
                f"variance provided through covariance in uncorrelated bootstrap fit"
            )

    # check if the given arguments have the correct dimensions:
    if abscissa.ndim != 1:
        raise ValueError(f"abscissa should be one dimensional")
    (Nt,) = abscissa.shape
    if bootstrap_ordinate_est is not None:
        Nbst, _ = bootstrap_ordinate_est.shape

    if ordinate_est is not None:
        if ordinate_est.shape != (Nt,):
            raise ValueError(f"abscissa and ordinate_est should have the same shape")
    if ordinate_var is not None:
        if ordinate_var.shape != (Nt,):
            raise ValueError(f"abscissa and ordinate_var should have the same shape")
    if ordinate_cov is not None:
        if ordinate_cov.shape != (Nt, Nt):
            raise ValueError(
                f"ordinate_cov should have the shape (Nt,Nt), but has {ordinate_cov.shape}"
            )
    if bootstrap_ordinate_est is not None:
        if bootstrap_ordinate_est.shape != (Nbst, Nt):
            raise ValueError(f"bootstrap_ordinate_est should have shape (Nbst,Nt)")
    if bootstrap_ordinate_var is not None:
        if bootstrap_ordinate_var.shape != (
            Nbst,
            Nt,
        ) and bootstrap_ordinate_var.shape != (Nt,):
            raise ValueError(
                f"bootstrap_ordinate_var should have shape (Nbst,Nt) or (Nt,), but has {bootstrap_ordinate_var.shape}"
            )
    if bootstrap_ordinate_cov is not None:
        if bootstrap_ordinate_cov.shape != (
            Nbst,
            Nt,
            Nt,
        ) and bootstrap_ordinate_cov.shape != (Nt, Nt):
            raise ValueError(
                f"bootstrap_ordinate_cov should have shape (Nbst,(Nt,Nt)) or (Nt,Nt), but has {bootstrap_ordinate_cov.shape}"
            )

    # prepare the arguments for lsqfit
    args = {}
    if model is None:
        raise ValueError(
            f"a model for the fit is required, the function should have abscissa and the parameters as an argument"
        )
    args["fcn"] = model
    args["maxit"] = maxiter
    if prior is None and p0 is None:
        raise ValueError(f"at least one of prior and p0 needs to be defined")
    args["prior" if prior is not None else "p0"] = prior if prior is not None else p0

    # res = {}

    if bootstrap_fit:
        res = FitResult(
            ts=abscissa[0], te=abscissa[-1], Nbst=Nbst
        )  # prepare res for saving the bootstrap and possible central value fit results
    else:
        res = FitResult(
            ts=abscissa[0], te=abscissa[-1]
        )  # prepare res for central value fit results only

    # # prepare data for the central value fit:
    if central_value_fit:
        # for the uncorrelated fit check in which way the variance is given and save in temp
        if not central_value_fit_correlated:
            if ordinate_var is not None:
                temp: np.ndarray = ordinate_var
            elif ordinate_cov is not None:
                temp: np.ndarray = np.diag(ordinate_cov)
                print(f"Debug 1")
            elif bootstrap_ordinate_var is not None:
                if bootstrap_ordinate_var.shape == (Nt,):
                    temp: np.ndarray = bootstrap_ordinate_var
                if bootstrap_ordinate_var.shape == (Nbst, Nt):
                    raise ValueError(
                        f"no variance specified for central value fit, only bootstrap_ordinate_var with shape {bootstrap_ordinate_var.shape}"
                    )
            elif bootstrap_ordinate_cov is not None:
                if bootstrap_ordinate_cov.shape == (Nt, Nt):
                    temp: np.ndarray = np.diag(bootstrap_ordinate_cov)
                else:
                    raise ValueError(
                        f"no variance given and could not be extracted from bootstrap_ordinate_cov with shape {bootstrap_ordinate_cov.shape}"
                    )
        # for the correlated fit check in which way the covariance  is given and save in temp
        else:
            if ordinate_cov is not None:
                temp: np.ndarray = ordinate_cov
            elif bootstrap_ordinate_cov is not None:
                if bootstrap_ordinate_cov.shape == (Nt, Nt):
                    temp: np.ndarray = bootstrap_ordinate_cov
                if bootstrap_ordinate_cov.shape == (Nbst, Nt, Nt):
                    raise ValueError(
                        f"no covariance specified for central value fit, only bootstrap_ordinate_cov with shape {bootstrap_ordinate_cov.shape}"
                    )
            # ToDo: if bootstrap_ordinate_est is given, calculate covariance from that with np.cov(bootstrap_ordinate_est,rowvar=False)
        ordinate_gvar = gv.gvar(
            (
                ordinate_est
                if ordinate_est is not None
                else np.mean(bootstrap_ordinate_est, axis=0)
            ),
            temp,
        )
        args["data" if central_value_fit_correlated else "udata"] = (
            abscissa,
            ordinate_gvar,
        )

        # Do the fit with lsqfit
        try:
            res.import_from_nonlinear_fit(
                nlf=lsqfit.nonlinear_fit(**args)
            )  # Save the fit results in res (object of type FitResult)
            # res["central value fit"] = lsqfit.nonlinear_fit(**args)  # FitResult()

        except Exception as e:
            msg = f"Fit Failed: :\n"
            for key, val in args.items():
                msg += f"- {key}: {val}\n"
            raise RuntimeError(f"{msg}\n{e}")

    if not bootstrap_fit:
        with h5py.File("../Report/FitResult.h5", "w") as h5f:
            res.serialize(h5_handle=h5f, node=f"timeslice_{abscissa[0]}_{abscissa[-1]}")
            FitResult.deserialize(
                h5_handle=h5f, node=f"timeslice_{abscissa[0]}_{abscissa[-1]}"
            )
        return res  # return fit results

    # prepare data for a bootstrap fit
    for nbst in range(Nbst):
        # for uncorrelated bootstrap fit check in which way the variance is given and save in temp
        if not bootstrap_fit_correlated:
            if bootstrap_ordinate_var is not None:
                if bootstrap_ordinate_var.shape == (Nt,):
                    temp = bootstrap_ordinate_var
                if bootstrap_ordinate_var.shape == (Nbst, Nt):
                    temp = bootstrap_ordinate_var[nbst]
            elif bootstrap_ordinate_cov is not None:
                temp: np.ndarray = np.diag(bootstrap_ordinate_cov)
        # for a correlated bootstrap fit check in which way the covariance is given and save in temp
        else:
            if bootstrap_ordinate_cov.shape == (Nt, Nt):
                temp = bootstrap_ordinate_cov
            if bootstrap_ordinate_cov.shape == (Nbst, Nt, Nt):
                temp = bootstrap_ordinate_cov[nbst]

        ordinate_gvar = gv.gvar(bootstrap_ordinate_est[nbst, :], temp)
        args["data" if bootstrap_fit_correlated else "udata"] = (
            abscissa,
            ordinate_gvar,
        )
        # varying the prior mean value for each bootstrap sample, to avoid bias
        if prior is not None and bootstrap_fit_resample_prior:
            prior_bst = gv.BufferDict()
            for key in prior.keys():
                prior_bst[key] = gv.gvar(gv.sample(prior[key], 1), prior[key].sdev)
            args["prior"] = prior_bst
        try:
            # pass
            res.import_from_nonlinear_fit(
                lsqfit.nonlinear_fit(**args), nbst
            )  # Do the fit with lsqfit
        except Exception as e:
            msg = f"Fit Failed: :\n"
            for key, val in args.items():
                msg += f"- {key}: {val}\n"
            msg += f"- nbst: {nbst}\n"
            raise RuntimeError(f"{msg}\n{e}")

    with h5py.File("../Report/FitResult.h5", "w") as h5f:
        res.serialize(h5_handle=h5f, node=f"timeslice_{abscissa[0]}_{abscissa[-1]}")
        # res2 = FitResult.deserialize(
        #     h5_handle=h5f, node=f"timeslice_{abscissa[0]}_{abscissa[-1]}"
        # )
    return res


def test_defensive(Nt: int = 32, Nbst: int = 100):

    abscissa: np.ndarray = np.zeros(Nt)
    ordinate_est: np.ndarray[gv.GVar] = np.zeros(Nt, dtype=gv.GVar)
    ordinate_var: np.ndarray[gv.GVar] = np.zeros(Nt, dtype=gv.GVar)
    ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nt, Nt), dtype=gv.GVar)

    bootstrap_ordinate_est: np.ndarray[gv.GVar] = np.zeros((Nbst, Nt), dtype=gv.GVar)
    bootstrap_ordinate_var: np.ndarray[gv.GVar] = np.zeros((Nbst, Nt), dtype=gv.GVar)
    bootstrap_ordinate_cov: np.ndarray[gv.GVar] = np.zeros(
        (Nbst, Nt, Nt), dtype=gv.GVar
    )

    # tesing arguments for uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(abscissa=abscissa, ordinate_est=ordinate_est, ordinate_var=ordinate_var)
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(abscissa=abscissa, ordinate_est=ordinate_est, ordinate_cov=ordinate_cov)
    # 3. partially wrong (bootstrap arguments instaed of central value arguments)
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_var=bootstrap_ordinate_var,
    )
    # 4. partially wrong (bootstrap arguments instead of central value and covariance instead of variance)
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
    )
    # 5. wrong arguments
    with pytest.raises(TypeError) as E:
        fit()  # -> req. abscissa
    assert E.type is TypeError
    with pytest.raises(ValueError) as E:
        fit(abscissa=abscissa)  # -> req. ordinate_est
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa, ordinate_est=ordinate_est
        )  # -> req. ordinate_var (ordinate_cov)
    assert E.type is ValueError

    # testing arguments for correlated central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        central_value_fit_correlated=True,
    )
    # 2. partially wrong (bootstrap arguments instaed of central value arguments)
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        central_value_fit_correlated=True,
    )
    # 3. wrong argumuments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            ordinate_var=ordinate_var,
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

    # testing arguments for uncorr. bootstrap fit, no central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_var=bootstrap_ordinate_var,
        central_value_fit=False,
        bootstrap_fit=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        central_value_fit=False,
        bootstrap_fit=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa, central_value_fit=False, bootstrap_fit=True
        )  # -> req. bootstrap_ordinate_est
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            central_value_fit=False,
            bootstrap_fit=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov)
    assert E.type is ValueError

    # testing arguments for correlated bootstrap fit, no central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        central_value_fit=False,
        bootstrap_fit=True,
        bootstrap_fit_correlated=True,
    )
    # 2. wrong argumuments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_ordinate_var=bootstrap_ordinate_var,
            central_value_fit=False,
            bootstrap_fit=True,
            bootstrap_fit_correlated=True,
        )  # -> req. bootstrap_ordinate_cov
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            central_value_fit=False,
            bootstrap_fit=True,
            bootstrap_fit_correlated=True,
        )  # req. bootstrap_ordinate_cov
    assert E.type is ValueError

    # tesing arguments for uncorr. bootstrap fit and uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_var=ordinate_var,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_var=bootstrap_ordinate_var,
        bootstrap_fit=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        bootstrap_fit=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa, ordinate_est=ordinate_est, bootstrap_fit=True
        )  # -> req. bootstrap_ordinate_est
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_fit=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov) and ordinate_var (ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            ordinate_var=ordinate_var,
            bootstrap_fit=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_fit=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov)
    assert E.type is ValueError

    # tesing arguments for corr. bootstrap fit and uncorr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_var=ordinate_var,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        bootstrap_fit=True,
        bootstrap_fit_correlated=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance for central value fit) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        bootstrap_fit=True,
        bootstrap_fit_correlated=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_fit=True,
            bootstrap_fit_correlated=True,
        )  # -> req. bootstrap_ordinate_cov
    assert E.type is ValueError

    # testing arguments for uncorr. bootstrap fit and corr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_var=bootstrap_ordinate_var,
        bootstrap_fit=True,
        central_value_fit_correlated=True,
    )
    # 2. partially wrong argumuments(covariance instead of variance for central value fit) should warn
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        bootstrap_fit=True,
        central_value_fit_correlated=True,
    )
    # 3. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov) and ordinate_cov
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            ordinate_cov=ordinate_cov,
            bootstrap_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. bootstrap_ordinate_var (bootstrap_ordinate_cov)
    assert E.type is ValueError
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_ordinate_var=bootstrap_ordinate_var,
            bootstrap_fit=True,
            central_value_fit_correlated=True,
        )  # -> req. ordinate_cov
    assert E.type is ValueError

    # testing arguments for corr. bootstrap fit and corr. central value fit:
    # 1. correct arguments should not raise
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_cov=ordinate_cov,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        central_value_fit_correlated=True,
        bootstrap_fit=True,
        bootstrap_fit_correlated=True,
    )
    fit(
        abscissa=abscissa,
        ordinate_est=ordinate_est,
        ordinate_var=ordinate_var,
        bootstrap_ordinate_est=bootstrap_ordinate_est,
        bootstrap_ordinate_cov=bootstrap_ordinate_cov,
        central_value_fit_correlated=True,
        bootstrap_fit=True,
        bootstrap_fit_correlated=True,
    )
    # 2. wrong arguments
    with pytest.raises(ValueError) as E:
        fit(
            abscissa=abscissa,
            ordinate_est=ordinate_est,
            ordinate_cov=ordinate_cov,
            bootstrap_ordinate_est=bootstrap_ordinate_est,
            bootstrap_ordinate_var=bootstrap_ordinate_var,
            central_value_fit_correlated=True,
            bootstrap_fit=True,
            bootstrap_fit_correlated=True,
        )  # -> req. bootstrap_ordinate_cov
    assert E.type is ValueError


if __name__ == "__main__":
    Nt = 10
    Nbst = 10
    Nconf = 100
    # abscissa: np.ndarray = np.zeros(Nt)
    # ordinate_est: np.ndarray[gv.GVar] = np.zeros(Nt,dtype=gv.GVar)
    # ordinate_var: np.ndarray[gv.GVar] = np.zeros(Nt,dtype=gv.GVar)
    # ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nt,Nt),dtype=gv.GVar)

    # bootstrap_ordinate_est: np.ndarray[gv.GVar] = np.zeros((Nbst,Nt),dtype=gv.GVar)
    # bootstrap_ordinate_var: np.ndarray[gv.GVar] = np.zeros((Nbst,Nt),dtype=gv.GVar)
    # bootstrap_ordinate_cov: np.ndarray[gv.GVar] = np.zeros((Nbst,Nt,Nt),dtype=gv.GVar)

    # fit(
    #     abscissa = abscissa,
    #     ordinate_est = ordinate_est,
    #     bootstrap_ordinate_est=bootstrap_ordinate_est,
    #     bootstrap_ordinate_cov=bootstrap_ordinate_cov
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
    data_bst = np.zeros((Nbst, Nt))
    for nbst in range(Nbst):
        data_bst[nbst] = np.mean(
            data2[np.random.randint(0, Nconf, size=(Nconf,))], axis=0
        )

    # plt.errorbar(abscissa,gv.mean(data2[0]),gv.sdev(data2[0]),capsize=2)
    # plt.yscale("log")
    # plt.show()
    # print(data_bst.mean(axis=0))
    # print(data_bst.std(axis=0))

    # #bootstrap fit:
    # res =fit(
    #     abscissa = abscissa,
    #     bootstrap_ordinate_est=data_bst,
    #     bootstrap_ordinate_cov=np.cov(data_bst,rowvar=False),
    #     prior = {
    #         "E0": gv.gvar(0.5,100), #flat prior
    #         "A0": gv.gvar(0.5,100)
    #     },
    #     model= lambda t,p: p["A0"]*np.exp(-t*p["E0"]),
    #     bootstrap_fit=True,
    #     bootstrap_fit_resample_prior=False,
    #     central_value_fit=False,
    #     bootstrap_fit_correlated=True
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
        # ordinate_est=gv.mean(data),
        # odinate_var=gv.var(data),
        bootstrap_ordinate_est=data_bst,
        bootstrap_ordinate_cov=np.cov(data_bst, rowvar=False),
        # prior = {
        #     "E0": gv.gvar(0.5,100), #flat prior
        #     "A0": gv.gvar(0.5,100)
        # },
        p0={"E0": 0.5, "A0": 0.5},
        model=lambda t, p: p["A0"] * np.exp(-t * p["E0"]),
        bootstrap_fit=True,
        bootstrap_fit_resample_prior=False,
        bootstrap_fit_correlated=True,
        central_value_fit=False,
    )

    # # print dict res:
    # for key, value in res.items():
    #     print(f"{key}:{value}")
    # # plt.show()
