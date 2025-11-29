from collections.abc import Callable

import numpy as np

import multiprocess as mp

from .data import Data

from .prior import Prior

from .fitResult import FitResult

import iminuit

def construct_correlated_least_square(abscissa, y_data, cov_inv, model, param_names:list[str], priors: dict[str|Prior] | None = None):

    if priors is None:
        def cost(*args):
            predi = model(abscissa, dict(zip(param_names,args)))
            delta = y_data - predi
            return np.einsum("i,j,ij", delta, delta, cov_inv)
        
    else:
        def cost(*args):
            predi = model(abscissa, dict(zip(param_names,args)))
            delta = y_data - predi
            prior = sum( [ priors[key](args[param_id]) for param_id,key in enumerate(priors.keys()) ] )

            return np.einsum("i,j,ij", delta, delta, cov_inv) + prior
    
    cost.errordef = iminuit.Minuit.LEAST_SQUARES
    cost.ndata = len(abscissa)

    return cost

def construct_uncorrelated_least_square(abscissa, y_data, sdev_inv, model, param_names: list[str], priors: dict[str|Prior] | None = None):

    if priors is None:
        def cost(*args):
            predi = model(abscissa, dict(zip(param_names,args)))
            delta = y_data - predi
            return sum((delta*sdev_inv)**2)
        
    else:
        def cost(*args):
            predi = model(abscissa, dict(zip(param_names,args)))
            delta = y_data - predi
            # prior = sum( [ priors[key](args[key]) for key in priors.keys() ] )
            prior = sum( [ priors[key](args[param_id]) for param_id,key in enumerate(priors.keys()) ] )

            return sum((delta*sdev_inv)**2) + prior
    
    cost.errordef = iminuit.Minuit.LEAST_SQUARES
    cost.ndata = len(abscissa)

    return cost

def execute_fit(fit_args: list[dict] | dict, nres: list[int] | None) -> dict:
    out_dict = {
        "nres": nres,
        "minuit" : None if nres is None else [None] * len(nres),
        "error":None if nres is None else [None] * len(nres),
    }

    # central value fits 
    if nres is None:
        try:
            minuit = iminuit.Minuit(
                fit_args["least_square"],
                **fit_args["p0"],
                name=list(fit_args["p0"].keys()),
            ).migrad()
            out_dict["minuit"] = minuit

        except Exception as e:
            out_dict["error"] = e 
    # resample fits fot a set of resamples provided in nres

    elif isinstance(nres,list):
        for res_id, _ in enumerate(nres):
            try:
                minuit = iminuit.Minuit(
                    fit_args[res_id]["least_square"],
                    **fit_args[res_id]["p0"],
                    name=list(fit_args[res_id]["p0"].keys()),
                ).migrad()
                out_dict["minuit"][res_id] = minuit

            except Exception as e:
                out_dict["error"][res_id] = e
    else: 
        raise ValueError(f"nres must but list of ints but is {type(nres)}: {nres}")
    
    return out_dict

def fit_iminuit(
    *,
    abscissa: Data | np.ndarray,
    ordinate: Data,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    resample_fit_resample_prior: bool = True,
    # args for lsqfit:
    model: Callable | None = None,
    prior: dict | None = None,
    p0: dict | None = None,
    svdcut: float | None = None,
    maxiter: int = 10_000,
    # optional parallelization:
    Nproc: int | None = None
) -> FitResult:
    r"""!
        @param abscissa: datapoints for the x-axis (i.e. an array containing Nt times (shape (Nt,))
        @param ordinate_est: datapoints for the y-axis (i.e. an array containing the datapoints measured at Nt times (shape (Nt,))
        @param ordinate_std: standard deviation of the given y datapoints (i.e. an array containing the error of the measured datapoints (shape (Nt,))
        @param ordinate_cov: covariance matrix of the y datapoints to specify the correlation betweem them
        @param resample_ordinate_est: datapoints for the y-axis for each resample (array of shape (Nres,Nt))
        @param resample_ordinate_std: standard deviation of the datapoints for each resample (array of shape (Nres,Nt) or (Nt,), the latter uses the given variance for all resamples)
        @param resample_ordinate_cov: covariance matrix of the datapoints for each resample (array of shape (Nres,Nt,Nt) or (Nt,Nt), the latter uses the given covariance matrix for all resamples)

        @param central_value_fit: option whether a central value fit should be performed (default: True)
        @param central_value_fit_correlated: option whether correlated fit should be performed (default: False)
        @param resample_fit: option whether a resample fit should be performed (delfault: False)
        @param resample_ft_corrrelated: option whether a correlated resample fit should be performed (default: False)
        @param resample_fit_resample_prior: option whether the meanvalue of the prior should be resampled for each resample, without resampling (option 'False') the same meanvalue for the prior is used for all resamples (default: True)
        @param resample_type: a string representing a resample type (None, 'bst' bootstrap, 'jkn' jackknife). This is passed to the FitResult to determine error calculation.

        @param model: the function to be fit to the datapoints, arguments should be the abscissa and the fit parameters
        @param prior: a priori estimates for the fit parameters (default: None)
        @param p0: start value for the fit parameters (default:None)
        @param maxiter: the maximum of iterations to perform the fit (default: 10_000)
        @param Nproc: Number of processes (cpus) to parallelize the resample fits. Seriell if None (default = None) 

        This function peforms a fit using lsqfit by Peter Lapage. Default is an uncorrelated central value fit.
        Optional are a correlated central value fit and a correlated (uncorrelated) resample fit. A FitResult is then returned
    """

    # Ensure that we got at least one fitting strategy (both are possible and will be handled accordingly)
    if not (central_value_fit or resample_fit):
        raise ValueError(f"At least one fit strategy needs to be defined: central_value_fit or resample_fit")


    # check if the given arguments have the correct dimensions:
    
    # The first axis defines the number of points, we may allow abscissas with more than one dimension
    # where the user needs to ensure that the respective model function fcn(abscissa, p) -> np.array
    # reduces to the output shape of the ordinate. 
    # e.g. in Lattice QCD we may analyse 3-point correlators with a source-sink separation t and an 
    # insertion time tau: C3pt(t,tau). 
    # Now one wants to fit multiple t,tau data points thus organizes the data such that
    # abscissa = [(t1,0), (t1,1), ..., (t1,t1+1), (t2,0), ... (t2,t2+1), ... ]
    # and respectively 
    # ordinate_est = [ C3pt(t1,0), C3pt(t1,1), ..., C3pt(t1,t1+1), ... ]
    # Then abscissa is two dimensional and the size of the first axis equals the number of data points
    # to fit against. 
    Nres = ordinate.Nresample

    if isinstance(abscissa, Data):
        if abscissa.Nresample != ordinate.Nresample:
            raise RuntimeError(f"abscissa {(abscissa)} doesn't match ordinate ({ordinate}) in number of resamples")

    # The organization of resample_ordinate_est.shape = Nres, Nt, ...
    # i.e. the second axis must match the first axis of abscissa. 
    # Further, dimensions are ignored and must be handled by the fit model
    if ordinate.shape != abscissa.shape:
        raise ValueError(f"Expecting ordinate shape ({ordinate.shape}) to match abscissa shape ({abscissa.shape})")
    
    # Check the existence of the model function
    if model is None:
        raise ValueError(f"A model for the fit is required, the function should have abscissa and the parameters as an argument")

    # Determine if we work with priors or simple start parameters
    if prior is None and p0 is None:
        raise ValueError(f"At least one of prior or p0 needs to be defined")


    if central_value_fit_correlated or resample_fit_correlated:
        cov_inv = np.linalg.inv(ordinate.cov)
        # todo dvd cut
    
    # ##############################################################################################
    # ##############################################################################################
    # Now all relevant parameters are there and have the expected shapes. 
    # We can now fill a dictionary args that is providing relevant information to the underlying fitter
    # provided by lsqfit  
    # ##############################################################################################
    # ##############################################################################################

    # prepare the arguments for lsqfit
    args = {}

    # populate the prior/start parameter
    if prior is not None:
        args["p0"] = {
            key: prior[key].mean for key in prior.keys()
        }
    else:
        args["p0"] = p0

    # define a FitResult that can be returned
    if resample_fit:
        # prepare for saving the resamples and possible central value fit results
        fit_result = FitResult(
            # abscissa used in the fit
            abscissa = abscissa,
            # number of resamples
            Nresample = ordinate.Nresample, 
            # resample type
            resample_type = ordinate.resample_type
        )  
    else: 
        # prepare for central value fit results only
        fit_result = FitResult(
            abscissa=abscissa,
        ) 

    # prepare data for the central value fit:
    if central_value_fit:
        if central_value_fit_correlated:
            least_square = construct_correlated_least_square(
                abscissa=abscissa.mean if isinstance(abscissa, Data) else abscissa, 
                y_data=ordinate.mean,
                cov_inv=cov_inv,
                model=model,
                param_names=list(args["p0"].keys()),
                priors=prior
            )
        else:
            least_square = construct_uncorrelated_least_square(
                abscissa=abscissa.mean if isinstance(abscissa, Data) else abscissa,
                y_data=ordinate.mean,
                sdev_inv=1/ordinate.serr,
                model=model,
                param_names=list(args["p0"].keys()),
                priors=prior
            )
    
        args["least_square"] = least_square

        # ##############################################################################################
        # Now all required fields in args are populated to attempt a fit 
        # ##############################################################################################
        res_dict = execute_fit(fit_args=args, nres=None)

        if res_dict["error"] is not None:
            raise res_dict["error"]

        fit_result.import_from_iminuit(minuit=res_dict["minuit"],model=model,prior=prior,Ndata=np.prod(ordinate.shape))
    # end if central value fit
    
    # If no resampled fits are supposed to be done we can return here
    if not resample_fit:
        return fit_result

    # collect all the data for resample fits
    args = np.empty(Nres, dtype=object)
    for nres in range(Nres):
        # prepare the arguments for lsqfit
        args[nres] = {}

        if prior is not None:
            args[nres]["p0"] = {
                key: prior[key].mean for key in prior.keys()
            }
        else:
            args[nres]["p0"] = p0

        if resample_fit_correlated:
            least_square = construct_correlated_least_square(
                abscissa=abscissa.rspl[nres] if isinstance(abscissa, Data) else abscissa, 
                y_data=ordinate.rspl[nres],
                cov_inv=cov_inv,
                model=model,
                param_names=list(args[nres]["p0"].keys()),
                priors=prior
            )
        else:
            least_square = construct_uncorrelated_least_square(
                abscissa=abscissa.rspl[nres] if isinstance(abscissa, Data) else abscissa,
                y_data=ordinate.rspl[nres],
                sdev_inv=1/ordinate.serr,
                model=model,
                param_names=list(args[nres]["p0"].keys()),
                priors=prior
            )
    
        args[nres]["least_square"] = least_square

        # ToDo: Something is wrong here...
        # varying the prior mean value for each resample sample, to avoid bias
        # if prior is not None and resample_fit_resample_prior:
        #     prior_res = gv.BufferDict()

        #     #we resample the prior in the standard way if the bootstrap is used
        #     if ordinate.resample_type == 'bst':
        #         for key in args[nres]["prior"].keys():
        #             prior_res[key] = gv.gvar(gv.sample(prior[key].gvar(), 1), prior[key].sdev)

        #     #if instead the jackknife resampling is being used, the resampple of the prior should be done with a std smaller by a factor of sqrt(Nres-1)
        #     elif ordinate.resample_type == 'jkn':
        #         for key in prior.keys():
        #             prior_res[key] = gv.gvar(gv.sample( gv.gvar(prior[key].mean, prior[key].sdev/np.sqrt(Nres-1)), 1), prior[key].sdev)
            
        #     args[nres]["prior"] = prior_res

        # # ##############################################################################################
        # # Now all required fields in args are populated to attempt a fit 
        # # ##############################################################################################

    if Nproc is None:
        out_dict = execute_fit(args, nres=list(range(Nres)))
        for nres in range(Nres):
            if out_dict["error"][nres] is not None:
                raise out_dict["error"][nres]
            fit_result.import_from_iminuit(out_dict["minuit"][nres],model=model,prior=prior,Ndata=np.prod(ordinate.shape),nres=nres)

    else:
        # compute the number of resamples that have to be done by any process
        Nblock = Nproc
        blockSize = Nres // Nblock
        Nrest  = Nres - (blockSize * Nblock)

        inputs = []
        for nblock in range(Nblock):
            res_slice = np.s_[ nblock*blockSize: (nblock+1)*blockSize ]
            inputs.append((
                args[res_slice], # list of dicts: fit_args 
                np.arange(Nres)[res_slice].tolist(), # list of resample ids: nres
                )
            )

        if Nrest > 0:
            res_slice = np.s_[ Nblock*blockSize: ]
            inputs.append(
                args[res_slice], # list of dicts: fit_args 
                np.arange(Nres)[res_slice].tolist(), # list of resample ids: nres
            )

        with mp.Pool(processes=Nproc) as pool:
            results = pool.starmap(execute_fit, inputs)

        # Now collect results and import them into FitResult
        errors = []
        for result in results:
            for res_id,nres in enumerate(result["nres"]):
                
                if result["error"][res_id] is not None:
                    errors.append((nres, result["error"]))
                    continue

                try:
                    fit_result.import_from_iminuit(out_dict["minuit"][nres],Ndata=np.prod(ordinate.shape),prior=prior,model=model,nres=nres)
                except Exception as e:
                    errors.append((nres, f"import_from_iminuit failed for nres={nres}: {e}"))

        if errors:
            for nres, err in errors:
                print(f"[parallel_run] nres={nres} error: {err}")
            raise RuntimeError("Found errors during execution of bootstrap fits")

    return fit_result

