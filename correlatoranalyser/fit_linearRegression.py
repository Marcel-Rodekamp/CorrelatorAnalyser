import numpy as np

import gvar as gv

import warnings

from .data import Data

from .fitResult import FitResult


def linear_regression(     
    *,
    abscissa: Data | np.ndarray,
    ordinate: Data,
    # fit strategy, default: only uncorrelated central value fit:
    central_value_fit: bool = True,
    central_value_fit_correlated: bool = False,
    resample_fit: bool = False,
    resample_fit_correlated: bool = False,
    has_intercept: bool = True,
    parameter_names: tuple | None = None 
):
    # Ensure that we got at least one fitting strategy (both are possible and will be handled accordingly)
    if not (central_value_fit or resample_fit):
        raise ValueError(f"At least one fit strategy needs to be defined: central_value_fit or resample_fit")

    if parameter_names is None:
        # slope, (intercept)
        if has_intercept:
            parameter_names = ("m", "b")
        else:
            parameter_names = ("m",)

    Nres = ordinate.Nresample

    if isinstance(abscissa, Data):
        if abscissa.Nresample != ordinate.Nresample:
            raise RuntimeError(f"abscissa {(abscissa)} doesn't match ordinate ({ordinate}) in number of resamples")

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

    if central_value_fit:
        if central_value_fit_correlated:
            weight_matrix = np.linalg.inv(ordinate.cov)
        else:
            weight_matrix = np.diag(ordinate.serr)

        if isinstance(abscissa,Data):
            design_matrix: np.ndarray = np.column_stack((abscissa.mean, np.ones_like(abscissa.mean))) if has_intercept else abscissa.mean.reshape(-1, 1)
        else:
            design_matrix: np.ndarray = np.column_stack((abscissa, np.ones_like(abscissa))) if has_intercept else abscissa.reshape(-1, 1)

        result_params, solution_matrix = lin_reg(
            y = ordinate.mean, 
            X = design_matrix, 
            W = weight_matrix, 
            S = None
        )

        fit_result.import_from_linear_regression(
            target_data = ordinate.mean,
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
        if resample_fit_correlated:
            weight_matrix = np.linalg.inv(ordinate.cov)
        else:
            weight_matrix = np.diag(ordinate.serr)

        if isinstance(abscissa,Data):
            design_matrix: np.ndarray = np.column_stack((abscissa.rspl[nres], np.ones_like(abscissa.rspl[nres]))) if has_intercept else abscissa.rspl[nres].reshape(-1, 1)
        else:
            design_matrix: np.ndarray = np.column_stack((abscissa, np.ones_like(abscissa))) if has_intercept else abscissa.reshape(-1, 1)

        result_params, solution_matrix = lin_reg(
            y = ordinate.rspl[nres], 
            X = design_matrix, 
            W = weight_matrix, 
            S = solution_matrix
        )

        fit_result.import_from_linear_regression(
            target_data = ordinate.rspl[nres],
            result_params = result_params,
            design_matrix = design_matrix,
            weight_matrix = weight_matrix,
            parameter_names = parameter_names,
            nres = nres
        )

    return fit_result

