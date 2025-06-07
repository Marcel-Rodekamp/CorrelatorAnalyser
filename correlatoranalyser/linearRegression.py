import numpy as np

import gvar as gv

import warnings

import h5py

from dataclasses import dataclass, field, fields

from pathlib import Path

from typing import Self, List, Dict

from .fitResult import FitResult

# # TODO: Project Generalize

def linear_regression(     
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
    resample_type: str | None = None,
    has_intercept: bool = True,
    parameter_names: tuple | None = None 
):
    # Ensure that we got at least one fitting strategy (both are possible and will be handled accordingly)
    if not (central_value_fit or resample_fit):
        raise ValueError(f"At least one fit strategy needs to be defined: central_value_fit or resample_fit")

    if parameter_names is None:
        # slope, (intercept)
        if has_intercept:
            parameter_names = ["m", "b"]
        else:
            parameter_names = ["m"]

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
        if resample_ordinate_cov is not None:
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
    
    # ##############################################################################################
    # ##############################################################################################
    # Now all relevant parameters are there and have the expected shapes. 
    # We can now fill a dictionary args that is providing relevant information to the underlying fitter
    # provided by lsqfit  
    # ##############################################################################################
    # ##############################################################################################

    def lin_reg(y, X = None, W = None, S = None):
        # if the solution matrix is not given we need X and W to construct it
        # here we check if X,W are given
        if (X is None or W is None) and S is None:
            raise RuntimeError(f"Solution matrix not provided and not possible to construct")
        # Now we can construct the solution matrix

        if S is None:
            S = np.linalg.inv(X.T @ W @ X) @ X.T @ W
        
        # This returns a np.array of size 1 (2) being slope (intercept)
        # and the constructed solution matrix, for later use
        return S @ y, S 
    # end if lin_reg

    # define a FitResult that can be returned
    if resample_fit:
        # prepare for saving the resamples and possible central value fit results
        fit_result = FitResult(
            # start point of the fit interval
            ts=abscissa[0], 
            # end point of the fit interval
            te=abscissa[-1], 
            # Number of data points
            Ndata = len(abscissa),
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
            te=abscissa[-1],
            # Number of data points
            Ndata = len(abscissa),
        ) 

    if central_value_fit:
        # for the uncorrelated fit check in which way the standard deviation is given and compute the weight matrix 
        if not central_value_fit_correlated:
            # simply provided standard deviation (preferred pass)
            if ordinate_std is not None:
                weight_matrix: np.ndarray = np.diag(1./ordinate_std**2)
            
            # provided covariance in case standard deviation is not given (preferred pass)
            elif ordinate_cov is not None:
                weight_matrix: np.ndarray = np.diag(1/np.diag(ordinate_cov))

            # provided resample standard deviation only, we can attempt to reuse it (optional pass for reusability)
            # if the dimension matches the expected standard deviation (frozen case for resampled fits)
            elif resample_ordinate_std is not None:
                if resample_ordinate_std.shape[0] == N:
                    weight_matrix: np.ndarray = np.diag(1/resample_ordinate_std**2)
                else:
                    raise ValueError(f"No standard deviation specified for central value fit, only resample_ordinate_std with shape {resample_ordinate_std.shape}")

            # Same as above but with covariance (optional pass for reusability)
            elif resample_ordinate_cov is not None:
                if resample_ordinate_cov.shape == (N, N):
                    weight_matrix: np.ndarray = np.diag(1/np.diag(resample_ordinate_cov))
                else:
                    raise ValueError(f"No standard devation given and could not be extracted from resample_ordinate_cov with shape {resample_ordinate_cov.shape}")
            
            # standard deviation not provided, doing an unweighted fit 
            else:
                weight_matrix: np.ndarray = np.eye(N)


        # for the correlated fit check in which way the covariance  is given and save in temp
        else:
            # simply provided covariance (preferred pass)
            if ordinate_cov is not None:
                weight_matrix: np.ndarray = np.linalg.inv(ordinate_cov)
            # provided resample covariance only, we can attempt to reuse it (optional pass for reusability)
            # if the dimension matches the expected covariance (frozen case for resampled fits)
            elif resample_ordinate_cov is not None:
                if resample_ordinate_cov.shape == (N, N):
                    weight_matrix: np.ndarray = np.linalg.inv(resample_ordinate_cov)
                else:
                    raise ValueError(f"No covariance specified for central value fit, only resample_ordinate_cov with shape {resample_ordinate_cov.shape}")
            else:
                raise ValueError(f"No covariance given and could not be extracted")

        design_matrix: np.ndarray = np.column_stack((abscissa, np.ones_like(abscissa))) if has_intercept else abscissa.reshape(-1, 1)

        result_params, solution_matrix = lin_reg(
            y = ordinate_est if ordinate_est is not None else np.mean(resample_ordinate_est, axis=0), 
            X = design_matrix, 
            W = weight_matrix, 
            S = None
        )

        fit_result.import_from_linear_regression(
            target_data = ordinate_est if ordinate_est is not None else np.mean(resample_ordinate_est, axis=0),
            result_params = result_params,
            design_matrix = design_matrix,
            weight_matrix = weight_matrix,
            parameter_names = parameter_names,
            nres = None
        )

    # end if central_value_fit

    if not resample_fit:
        return fit_result

    # cache the solution matrix if desired!
    solution_matrix = None 

    for nres in range(Nres):
        # for uncorrelated resampled fits check in which way the standard is given and save in temp
        if not resample_fit_correlated:
            # simply provided standard deviation 
            if resample_ordinate_std is not None:
                # frozen error: one for all
                if resample_ordinate_std.shape[0] == N:
                    frozen_weight_matrix:bool = True

                    weight_matrix: np.ndarray = np.diag(1/resample_ordinate_std**2)
                # one std for each resample
                elif resample_ordinate_std.shape[0] == Nres:
                    frozen_weight_matrix:bool = False

                    weight_matrix: np.ndarray = np.diag(1/resample_ordinate_std[nres]**2)
                else:
                    raise ValueError(f"Couldn't identify resample standard deviation from provided resample_ordinate_std of shape {resample_ordinate_std.shape}")
            # provided covariance in case standard deviation is not given
            elif resample_ordinate_cov is not None:
                # frozen error: one for all
                if resample_ordinate_cov.shape[0] == N and resample_ordinate_cov.shape[1] == N:
                    frozen_weight_matrix:bool = True
                    weight_matrix: np.ndarray = np.diag(1/np.diag(resample_ordinate_cov))

                # one std for each resample
                elif resample_ordinate_cov.shape[0] == Nres and resample_ordinate_cov.shape[1] == N and resample_ordinate_cov.shape[2] == N:
                    frozen_weight_matrix:bool = False
                    weight_matrix: np.ndarray = np.diag(1/np.diag(resample_ordinate_cov[nres,:,:]))
                else:
                    raise ValueError(f"Couldn't identify resample standard deviation from provided resample_ordinate_cov of shape {resample_ordinate_cov.shape}")

            # standard deviation not provided, doing an unweighted fit 
            else:
                frozen_weight_matrix:bool = True
                weight_matrix: np.ndarray = np.eye(N)

        # for a correlated resample fit check in which way the covariance is given and save in temp
        else:
            # simply provided covariance
            # frozen covariance: one for all
            if resample_ordinate_cov.shape[0] == N and resample_ordinate_cov.shape[1] == N:
                frozen_weight_matrix:bool = True
                weight_matrix:np.ndarray = np.linalg.inv(resample_ordinate_cov)
            # one covariance per resample
            elif resample_ordinate_cov.shape[0] == Nres and resample_ordinate_cov.shape[1] == N and resample_ordinate_cov.shape[2] == N:
                frozen_weight_matrix:bool = False
                weight_matrix:np.ndarray = np.linalg.inv(resample_ordinate_cov[nres])
            else:
                raise ValueError(f"Couldn't identify resample covariance from provided resample_ordinate_cov of shape {resample_ordinate_cov.shape}")

        design_matrix: np.ndarray = np.column_stack((abscissa, np.ones_like(abscissa))) if has_intercept else abscissa.reshape(-1, 1)

        result_params, solution_matrix = lin_reg(
            y = resample_ordinate_est[nres], 
            X = design_matrix, 
            W = weight_matrix, 
            S = solution_matrix if frozen_weight_matrix else None
        )

        fit_result.import_from_linear_regression(
            target_data = resample_ordinate_est[nres],
            result_params = result_params,
            design_matrix = design_matrix,
            weight_matrix = weight_matrix,
            parameter_names = parameter_names,
            nres = nres
        )

    return fit_result

