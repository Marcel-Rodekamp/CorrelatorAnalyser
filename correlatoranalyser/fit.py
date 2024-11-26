import numpy as np

import gvar as gv

import lsqfit

import warnings

from collections.abc import Callable

from .fitResult import FitResult

def fit(
    *,
    abscissa: np.ndarray,
    ordinate_est: np.ndarray[gv.GVar] | None = None,
    ordinate_std: np.ndarray[gv.GVar] | None = None,
    ordinate_cov: np.ndarray[gv.GVar] | None = None,
    resample_ordinate_est: np.ndarray[gv.GVar] | None = None,
    resample_ordinate_std: np.ndarray[gv.GVar] | None = None,
    resample_ordinate_cov: np.ndarray[gv.GVar] | None = None,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    resample_fit_resample_prior: bool = True,
    resample_type: str | None = None,
    # args for lsqfit:
    model: Callable | None = None,
    prior: dict | None = None,
    p0: dict | None = None,
    svdcut: float | None = None,
    maxiter: int = 10_000,
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
        This function peforms a fit using lsqfit by Peter Lapage. Default is an uncorrelated central value fit.
        Optional are a correlated central value fit and a correlated (uncorrelated) resample fit. The the best parameters for the fit are then returned
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
    N = abscissa.shape[0]


    # In case a central value fit is supposed to be performed check the accessibility of relevant parameters
    # Central Value fit requires
    #   - ordinate_est 
    #   - optional: ordinate_std
    #   - optional: covariance 
    if central_value_fit:

        # We need ordinate data to fit against. 
        # We can either use provided ordinate_est or fit against the mean over resample_ordinate_est
        # Check that at least one is provided
        if ordinate_est is None and resample_ordinate_est is None:
            raise ValueError(f"Central value fit requires ordinate_est (or resample_ordinate_est)")

        # In case a central value fit is desired check that covariance matrix is provided
        if central_value_fit_correlated and ordinate_cov is None:
            raise ValueError(f"Central value fit (correlated) requires ordinate_cov")


        # ##############################################################################################
        # Now all relevant parameters are there. We now deduce the fitting strategy and check the arrays
        # for the correct size.

        # The organization of ordinate_est.shape = Nt, ...
        # i.e. the first axis must match the first axis of abscissa. 
        # Further, dimensions are ignored and must be handled by the fit model
        if ordinate_est.shape[0] != N:
            raise ValueError(f"Expecting ordinate_est ({ordinate_est.shape}) with same first axis size as abscissa ({abscissa.shape})")

        # ordinate_std is optional
        if ordinate_std is not None:
            if ordinate_std.shape[0] != N:
                raise ValueError(f"Expecting ordinate_std ({ordinate_std.shape}) with same first axis size as abscissa ({abscissa.shape})")

        # ordinate_cov is optional
        if ordinate_cov is not None:
            if ordinate_cov.shape != (N, N):
                raise ValueError(f"Expecting ordinate_cov of shape ({N},{N}), but has {ordinate_cov.shape}")

    # In case resampled fits are supposed to be performed check the accessibility of relevant parameters
    # Resampled fits requires
    #   - ordinate 
    #   - covariance (if correlated)
    #   - Resample types ('bst' by default)
    if resample_fit:

        if resample_ordinate_est is None:
            raise ValueError(f"Resampled fit requires resample_ordinate_est")
        
        if resample_fit_correlated and resample_ordinate_cov is None:
            raise ValueError(f"Resampled fit (correlated) requires resample_ordinate_cov")

        # Check that resample_type is provided and if not set default as 'bootstrap'
        # We raise a warning in case it is not provided
        if resample_type is None:
            resample_type = 'bst'

            warnings.warn(f"Expecting resample_type but is not provided... Choosing bootstrap ('bst') by default")

        # ##############################################################################################
        # Now all relevant parameters are there. We now deduce the fitting strategy and check the arrays
        # for the correct size.

        # Extract number of resamples 
        Nres = resample_ordinate_est.shape[0]

        # The organization of resample_ordinate_est.shape = Nres, Nt, ...
        # i.e. the second axis must match the first axis of abscissa. 
        # Further, dimensions are ignored and must be handled by the fit model
        if resample_ordinate_est.shape[1] != N:
            raise ValueError(f"Expecting resample_ordinate_est ({resample_ordinate_est.shape}) with same second axis size as abscissa ({abscissa.shape}) first axis size")
        
        # resample_ordinate_std is optional
        if resample_ordinate_std is not None:
            # We expect resample_ordinate_std by dimensions
            # 1. (Nbst, N, ...), i.e. one uncertainty per resample
            # 2. (N, ...), i.e. one uncertainty for all resamples (frozen)

            # case 1:
            if resample_ordinate_std.shape[0] == Nres:
                if resample_ordinate_std.shape[1] != N:
                    raise ValueError(f"Expecting resample_ordinate_std of shape (Nres,N, ...) or (N, ...) but has {resample_ordinate_std.shape}")
            # case 2:
            else :
                if resample_ordinate_std.shape[0] != N:
                    raise ValueError(f"Expecting resample_ordinate_std of shape (Nres,N, ...) or (N, ...) but has {resample_ordinate_std.shape}")

        # resample_ordinate_cov is optional
        if resample_ordinate_std is not None:
            # We expect resample_ordinate_cov by dimensions
            # 1. (Nbst, N, N), i.e. one uncertainty per resample
            # 2. (N, N), i.e. one uncertainty for all resamples (frozen)

            # case 1:
            if resample_ordinate_cov.shape[0] == Nres:
                if resample_ordinate_cov.shape[1] != N and resample_ordinate_cov.shape[2] != N and resample_ordinate_cov.ndim == 3:
                    raise ValueError(f"Expecting resample_ordinate_cov of shape (Nres,N,N) or (N,N) but has {resample_ordinate_cov.shape}")
            # case 2:
            else :
                if resample_ordinate_cov.shape[0] != N and resample_ordinate_cov.shape[1] != N and resample_ordinate_cov.ndim == 2:
                    raise ValueError(f"Expecting resample_ordinate_cov of shape (Nres,N,N) or (N,N) but has {resample_ordinate_cov.shape}")
    
    # Check the existence of the model function
    if model is None:
        raise ValueError(f"A model for the fit is required, the function should have abscissa and the parameters as an argument")

    # Determine if we work with priors or simple start parameters
    if prior is None and p0 is None:
        raise ValueError(f"At least one of prior or p0 needs to be defined")
    
    # ##############################################################################################
    # ##############################################################################################
    # Now all relevant parameters are there and have the expected shapes. 
    # We can now fill a dictionary args that is providing relevant information to the underlying fitter
    # provided by lsqfit  
    # ##############################################################################################
    # ##############################################################################################

    # prepare the arguments for lsqfit
    args = {}

    # populate the fit function
    args["fcn"] = model

    # populate a maximal iteration for the minimizer
    args["maxit"] = maxiter

    # populate the prior/start parameter
    # on resamples, the prior mean may be resampled depending on
    # use of prior and the resample_fit_resample_prior
    args["prior" if prior is not None else "p0"] = prior if prior is not None else p0

    # svdcut is optional
    if svdcut is not None:
        args["svdcut"] = svdcut

    # define a FitResult that can be returned
    if resample_fit:
        # prepare for saving the resamples and possible central value fit results
        fit_result = FitResult(
            # start point of the fit interval
            ts=abscissa[0], 
            # end point of the fit interval
            te=abscissa[-1], 
            # number of resamples
            Nres=Nres, 
            # resample type
            resample_type = resample_type
        )  
    else: 
        # prepare for central value fit results only
        fit_result = FitResult(
            # start point of the fit interval
            ts=abscissa[0], 
            # end point of the fit interval
            te=abscissa[-1]
        ) 

    # prepare data for the central value fit:
    if central_value_fit:
        # for the uncorrelated fit check in which way the standard deviation is given and save in temp
        if not central_value_fit_correlated:
            # simply provided standard deviation (preferred pass)
            if ordinate_std is not None:
                temp: np.ndarray = ordinate_std
            
            # provided covariance in case standard deviation is not given (preferred pass)
            elif ordinate_cov is not None:
                temp: np.ndarray = np.diag(ordinate_cov)

            # provided resample standard deviation only, we can attempt to reuse it (optional pass for reusability)
            # if the dimension matches the expected standard deviation (frozen case for resampled fits)
            elif resample_ordinate_std is not None:
                if resample_ordinate_std.shape[0] == N:
                    temp: np.ndarray = resample_ordinate_std
                else:
                    raise ValueError(f"No standard deviation specified for central value fit, only resample_ordinate_std with shape {resample_ordinate_var.shape}")

            # Same as above but with covariance (optional pass for reusability)
            elif resample_ordinate_cov is not None:
                if resample_ordinate_cov.shape == (N, N):
                    temp: np.ndarray = np.diag(resample_ordinate_cov)
                else:
                    raise ValueError(f"No standard devation given and could not be extracted from resample_ordinate_cov with shape {resample_ordinate_cov.shape}")
            
            # standard deviation not provided, doing an unweighted fit 
            else:
                temp: np.ndarray = np.ones_like(ordinate_est)


        # for the correlated fit check in which way the covariance  is given and save in temp
        else:
            # simply provided covariance (preferred pass)
            if ordinate_cov is not None:
                temp: np.ndarray = ordinate_cov
            # provided resample covariance only, we can attempt to reuse it (optional pass for reusability)
            # if the dimension matches the expected covariance (frozen case for resampled fits)
            elif resample_ordinate_cov is not None:
                if resample_ordinate_cov.shape == (N, N):
                    temp: np.ndarray = resample_ordinate_cov
                else:
                    raise ValueError(f"No covariance specified for central value fit, only resample_ordinate_cov with shape {resample_ordinate_cov.shape}")
            else:
                raise ValueError(f"No covariance given and could not be extracted")

        ordinate_gvar = gv.gvar(
            # at least one is provided as checked above
            ordinate_est if ordinate_est is not None else np.mean(resample_ordinate_est, axis=0),
            # put the standard deviation/covariance as extraced into temp
            temp,
        )

        # Ensure un-/correlated fits are preformed by providing the correct data form to lsqfit
        # data :   correlated fit
        # udata: uncorrelated fit
        args["data" if central_value_fit_correlated else "udata"] = (abscissa, ordinate_gvar)


        # ##############################################################################################
        # Now all required fields in args are populated to attempt a fit 
        # ##############################################################################################

        try:
            # Save the fit results in fit_result (object of type FitResult)
            fit_result.import_from_lsqfit(nlf=lsqfit.nonlinear_fit(**args))  
        
        # In case something goes wrong we collect additional information and extend the exception message 
        except Exception as e:
            msg = f"Fit Failed: :\n"
            for key, val in args.items():
                msg += f"- {key}: {val}\n"
            raise RuntimeError(f"{msg}\n{e}")


    # end if central value fit
    
    # If no resampled fits are supposed to be done we can return here
    if not resample_fit:
        # return fit results
        return fit_result

    # prepare data for a resample fit
    for nres in range(Nres):
        # for uncorrelated resampled fits check in which way the standard is given and save in temp
        if not resample_fit_correlated:
            # simply provided standard deviation 
            if resample_ordinate_std is not None:
                # frozen error: one for all
                if resample_ordinate_std.shape[0] == N:
                    temp: np.ndarray = resample_ordinate_std
                # one std for each resample
                elif resample_ordinate_std.shape[0] == Nres:
                    temp: np.ndarray = resample_ordinate_std[nres]
                else:
                    raise ValueError(f"Couldn't identify resample standard deviation from provided resample_ordinate_std of shape {resample_ordinate_std.shape}")
            # provided covariance in case standard deviation is not given
            elif resample_ordinate_cov is not None:
                # frozen error: one for all
                if resample_ordinate_cov.shape[0] == N and resample_ordinate_cov.shape[1] == N:
                    temp: np.ndarray = np.diag(resample_ordinate_cov)
                # one std for each resample
                elif resample_ordinate_cov.shape[0] == Nres and resample_ordinate_cov.shape[1] == N and resample_ordinate_cov.shape[2] == N:
                    temp: np.ndarray = np.diag(resample_ordinate_cov[nres,:,:])
                else:
                    raise ValueError(f"Couldn't identify resample standard deviation from provided resample_ordinate_cov of shape {resample_ordinate_cov.shape}")

            # standard deviation not provided, doing an unweighted fit 
            else:
                temp: np.ndarray = np.ones_like(resample_ordinate_est[nres])

        # for a correlated resample fit check in which way the covariance is given and save in temp
        else:
            # simply provided covariance
            # frozen covariance: one for all
            if resample_ordinate_cov.shape[0] == N and resample_ordinate_cov.shape[1] == N:
                temp = resample_ordinate_cov
            # one covariance per resample
            elif resample_ordinate_cov.shape[0] == Nres and resample_ordinate_cov.shape[1] == N and resample_ordinate_cov.shape[2] == N:
                temp = resample_ordinate_cov[nres]
            else:
                raise ValueError(f"Couldn't identify resample covariance from provided resample_ordinate_cov of shape {resample_ordinate_cov.shape}")


        ordinate_gvar = gv.gvar(
            # each resample has its own ordinate data    
            resample_ordinate_est[nres, :], 
            # and deduced uncertainty
            temp
        )

        # Ensure un-/correlated fits are preformed by providing the correct data form to lsqfit
        # data :   correlated fit
        # udata: uncorrelated fit
        args["data" if resample_fit_correlated else "udata"] = (abscissa, ordinate_gvar)

        # varying the prior mean value for each resample sample, to avoid bias
        if prior is not None and resample_fit_resample_prior:
            prior_res = gv.BufferDict()

            for key in prior.keys():
                prior_res[key] = gv.gvar(gv.sample(prior[key], 1), prior[key].sdev)
            
            args["prior"] = prior_res

        # ##############################################################################################
        # Now all required fields in args are populated to attempt a fit 
        # ##############################################################################################
            
        try:
            # Save the fit results in fit_result (object of type FitResult)
            fit_result.import_from_lsqfit(nlf = lsqfit.nonlinear_fit(**args), nres = nres)

        # In case something goes wrong we collect additional information and extend the exception message 
        except Exception as e:
            msg = f"Fit Failed: :\n"
            for key, val in args.items():
                msg += f"- {key}: {val}\n"
            msg += f"- nres: {nres}\n"
            raise RuntimeError(f"{msg}\n{e}")


    return fit_result


