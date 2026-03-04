from collections.abc import Callable

import numpy as np

import multiprocess as mp

from .data import Data

from .prior import Prior

from .fitResult import FitResult

import iminuit

def construct_uncorrelated_variable_projection_least_square( abscissa, y_data, sdev_inv, model: Callable, param_names: list[str], linear_params: list[str], priors: dict | None = None):
    """
        Construct an uncorrelated least-squares cost function using
        Variable Projection (Golub–Pereyra method).
  
        Linear parameters are eliminated analytically at every function
        evaluation by solving the weighted linear least squares problem.

        Parameters
        ----------
        abscissa : array-like
        y_data : array-like
        sdev_inv : array-like
            Inverse standard deviations (1/sigma_i)
        model : callable
            Model function model(x, **params)
        param_names : list[str]
            All parameter names (order used by iminuit)
        linear_params : list[str]
            Subset of param_names that enter linearly
        priors : dict[str, Prior] or None
            Optional Gaussian priors

        Returns
        -------
        cost_function : callable
            Function suitable for iminuit
    """

    x = np.asarray(abscissa)
    y = np.asarray(y_data)
    w = np.asarray(sdev_inv)

    if priors is None:
        priors = {}

    # Split parameters
    linear_params = list(linear_params)
    nonlinear_params = [p for p in param_names if p not in linear_params]

    if len(linear_params) == 0:
        raise ValueError("Variable projection requires at least one linear parameter.")

    # ------------------------------------------------------------------
    # Helper: build design matrix for linear parameters
    # ------------------------------------------------------------------
    # This works only if no factorisation of the linear params is allowed
    # However, this is not always the case. ToDo: formulate 
    def build_design_matrix(nl_dict):
        """
            Construct weighted design matrix A and weighted RHS b.

            A_ij = w_i * d f(x_i) / d linear_param_j
            b_i  = w_i * y_i

            for a function if the form

                f(x_i) = \sum_{n} (linear_param_n) \Phi_n(x_i)

            such that 

                d f(x_i) / d linear_param_j = (linear_param_j) \Phi_j(x_i)

            which is equivalent to 
                d f(x_i) / d linear_param_j = f(x_i; linear_param_n = 1, if j==n else 0, non_linlear_params)
        """
        n_data = len(x)
        n_lin = len(linear_params)

        X = np.zeros((n_data, n_lin))

        # Compute basis functions by toggling linear parameters
        # This computes d f(x_i) / d linear_param_j
        for j, lp in enumerate(linear_params):
            test_params = {**nl_dict}

            for p in linear_params:
                test_params[p] = 0.0
            test_params[lp] = 1.0

            X[:, j] = model(x, test_params)

        # Apply weights
        X *= w[:, None]
        b = w * y
        
        return X, b

    # ------------------------------------------------------------------
    # Helper: solve linear system (with optional priors TBD)
    # ------------------------------------------------------------------
    def solve_linear(X, b):
        """
        Solve (X^T X) c = X^T b
        """
        XTX = X.T @ X
        XTb = X.T @ b

        try:
            coeffs = np.linalg.solve(XTX, XTb)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            coeffs, *_ = np.linalg.lstsq(XTX, XTb, rcond=None)
        except Exception as e:
            raise e

        return coeffs

    # ------------------------------------------------------------------
    # Cost function seen by iminuit (nonlinear params only)
    # ------------------------------------------------------------------
    def cost_function(*nl_values):
        # sort non_linear params
        nl_dict = dict(zip(nonlinear_params, nl_values))
        
        # linear solve for linear params
        X, b = build_design_matrix(nl_dict)
        coeffs = solve_linear(X, b)

        # Collect all params
        params = dict(zip(linear_params, coeffs))
        params.update(nl_dict)

        # evaluate model
        y_fit = model(x, params)

        # compute weighted residuals
        r = w * (y - y_fit)

        # sum over data points
        chi2 = np.sum(r**2)

        # add priors
        chi2+=sum( [ priors[key](nl_dict[key]) for key in enumerate(nl_dict.keys()) if key in priors ] )

        return chi2

    # # ---- Make iminuit see correct signature ----
    cost_function.func_code = type(
        "", (), {
            "co_varnames": tuple(nonlinear_params),
            "co_argcount": len(nonlinear_params),
        }
    )()

    cost_function._build_design_matrix = build_design_matrix
    cost_function._solve_linear = solve_linear
    cost_function._linear_params = linear_params
    cost_function._nonlinear_params = nonlinear_params

    return cost_function

def execute_fit(fit_args: list[dict] | dict, nres: list[int] | None) -> dict:
    out_dict = {
        "nres": nres,
        "minuit" : None if nres is None else [None] * len(nres),
        "varproj" : None if nres is None else [None] * len(nres),
        "error":None if nres is None else [None] * len(nres),
    }

    # central value fits 
    if nres is None:
        try:
            cost = fit_args["least_square"]

            minuit = iminuit.Minuit(
                fit_args["least_square"],
                **{
                    key:fit_args["p0"][key] for key in fit_args["p0"] if key not in cost._linear_params
                },
                name=[key for key in fit_args["p0"] if key not in cost._linear_params] #list(fit_args["p0"].keys()),
            )
            # Possible stability parameters
            # minuit.tol = 1e-16
            # minuit.precision = 1e-16
            # minuit.strategy = 2
            minuit.migrad()

            # Resolve linear params for best fit in order to have all parameters available

            # nonlinear best-fit values
            best_nl = {p: minuit.values[p] for p in cost._nonlinear_params}

            X, b = cost._build_design_matrix(best_nl)
            coeffs = cost._solve_linear(X, b)

            # inject linear parameters into minuit object
            varproj = dict(zip(cost._linear_params, coeffs))

            # Store the results in the output dict
            out_dict["minuit"] = minuit
            out_dict["varproj"] = varproj

        except Exception as e:
            out_dict["error"] = e 
    # resample fits fot a set of resamples provided in nres

    elif isinstance(nres,list):
        for res_id, _ in enumerate(nres):
            try:
                cost = fit_args[res_id]["least_square"]

                minuit = iminuit.Minuit(
                    fit_args[res_id]["least_square"],
                    **{
                        key:fit_args[res_id]["p0"][key] for key in fit_args[res_id]["p0"] if key not in cost._linear_params
                    },
                    name=[key for key in fit_args[res_id]["p0"] if key not in cost._linear_params] 
                )

                # Possible stability parameters
                # minuit.tol = 1e-16
                # minuit.precision = 1e-16
                # minuit.strategy = 2
                minuit.migrad()

                # Resolve linear params for best fit in order to have all parameters available
                
                # nonlinear best-fit values
                best_nl = {p: minuit.values[p] for p in cost._nonlinear_params}

                X, b = cost._build_design_matrix(best_nl)
                coeffs = cost._solve_linear(X, b)

                # inject linear parameters into minuit object
                varproj = dict(zip(cost._linear_params, coeffs))

                out_dict["minuit"][res_id] = minuit
                out_dict["varproj"][res_id] = varproj

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
    # variable projection
    linear_params: list[str],
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
    if ordinate.shape[0] != abscissa.shape[0]:
        raise ValueError(f"Expecting ordinate shape ({ordinate.shape}) to match abscissa shape ({abscissa.shape[0]})")
    
    # Check the existence of the model function
    if model is None:
        raise ValueError(f"A model for the fit is required, the function should have abscissa and the parameters as an argument")

    # Determine if we work with priors or simple start parameters
    if prior is None and p0 is None:
        raise ValueError(f"At least one of prior or p0 needs to be defined")

    if prior is not None:
        for key in linear_params:
            if key in prior.keys():
                prior.pop(key)
                print(f"Found prior for linear parameter ({key}). Will be ignored")


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
            key: prior[key].mean for key in prior.keys() if key not in linear_params
        }
    else:
        args["p0"] = {
            key: p0[key] for key in p0.keys() if key not in linear_params
        }

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
            raise NotImplementedError(
                "Variable projection is currently only implemented for uncorrelated fits"
            )
        else:
            # print(f"Performing variable projections for: {linear_params}")

            least_square = construct_uncorrelated_variable_projection_least_square(
                abscissa=abscissa.mean if isinstance(abscissa, Data) else abscissa,
                y_data=ordinate.mean,
                sdev_inv=1/ordinate.serr,
                model=model,
                param_names=list(args["p0"].keys()),
                linear_params=list(linear_params),
                priors=prior
            )

        args["least_square"] = least_square

        # ##############################################################################################
        # Now all required fields in args are populated to attempt a fit 
        # ##############################################################################################
        res_dict = execute_fit(fit_args=args, nres=None)

        if res_dict["error"] is not None:
            raise res_dict["error"]

        fit_result.import_from_iminuit(minuit=res_dict["minuit"],model=model,variable_projection=res_dict["varproj"],prior=prior,Ndata=np.prod(ordinate.shape))
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
            raise NotImplementedError(
                "Variable projection is currently only implemented for uncorrelated fits"
            )
        else:
            least_square = construct_uncorrelated_variable_projection_least_square(
                abscissa=abscissa.rspl[nres] if isinstance(abscissa, Data) else abscissa,
                y_data=ordinate.rspl[nres],
                sdev_inv=1/ordinate.serr,
                model=model,
                param_names=list(args[nres]["p0"].keys()),
                linear_params=list(linear_params),
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

            fit_result.import_from_iminuit(out_dict["minuit"][nres],model=model,variable_projection=out_dict["varproj"][nres],prior=prior,Ndata=np.prod(ordinate.shape),nres=nres)

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
            inputs.append((
                args[res_slice], # list of dicts: fit_args 
                np.arange(Nres)[res_slice].tolist(), # list of resample ids: nres
                )
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
                    fit_result.import_from_iminuit(result["minuit"][res_id],Ndata=np.prod(ordinate.shape),prior=prior,model=model,variable_projection=result["varproj"][res_id],nres=nres)
                except Exception as e:
                    errors.append((nres, f"import_from_iminuit failed for nres={nres}: {e}"))

        if errors:
            for nres, err in errors:
                print(f"[parallel_run] nres={nres} error: {err}")
            raise RuntimeError("Found errors during execution of bootstrap fits")

    return fit_result

