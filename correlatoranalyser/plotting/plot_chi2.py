import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import gvar as gv

import lsqfit

import itertools

from collections.abc import Callable

from matplotlib.ticker import MaxNLocator

from .mplStyle import *

def plot_chi2(
    *,
    param_ranges: dict,
    abscissa: np.ndarray,
    ordinate_est: np.ndarray[gv.GVar],
    ordinate_std: np.ndarray[gv.GVar] | None = None,
    ordinate_cov: np.ndarray[gv.GVar] | None = None,
    correlated_chi2: bool = False,
    model: Callable | None = None,
    prior: dict | None = None,
    log_chi2: bool = False,
    fixed_params: dict | None = None
):
    r"""!
        @param abscissa: datapoints for the x-axis (i.e. an array containing Nt times (shape (Nt,))
        @param ordinate_est: datapoints for the y-axis (i.e. an array containing the datapoints measured at Nt times (shape (Nt,))
        @param ordinate_std: standard deviation of the given y datapoints (i.e. an array containing the error of the measured datapoints (shape (Nt,))
        @param ordinate_cov: covariance matrix of the y datapoints to specify the correlation betweem them
        @param correlated_chi2: Set to True if the covariance should be used instead of the standard deviation in the definition of the chi^2
        @param parameter_range: A dictionary with TWO parameter keys as used in model. 
                                Each entry is expected to be a numpy array with allowed values. 
                                The arrays need to be of equal length (they are zipped to get a full set of parameters) 
        
        @param model: the function to be fit to the datapoints, arguments should be the abscissa and the fit parameters
        @param prior: a priori estimates for the fit parameters, it extends the chi^2 definition (default: None),
        @param fixed_params, further parameters for the model that are fixed to the provided value
    """

    # check if the given arguments have the correct dimensions etc:
    for key, p in param_ranges.items():
        if not isinstance(p,np.ndarray):
            raise ValueError(f"Parameter range {key} is not a valid array: {p}")

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

    # We need ordinate data to fit against. 
    # We can either use provided ordinate_est or fit against the mean over resample_ordinate_est
    # Check that at least one is provided
    if ordinate_est is None:
        raise ValueError(f"Requires ordinate_est (or resample_ordinate_est)")

    # In case a central value fit is desired check that covariance matrix is provided
    if correlated_chi2 and ordinate_cov is None:
        raise ValueError(f"Correlated_chi2 requires ordinate_cov")


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

    # Check the existence of the model function
    if model is None:
        raise ValueError(f"A model for the fit is required, the function should have abscissa and the parameters as an argument")
    
    # ##############################################################################################
    # ##############################################################################################
    # Now all relevant parameters are there and have the expected shapes. 
    # ##############################################################################################
    # ##############################################################################################

    # for the uncorrelated fit check in which way the standard deviation is given and save in temp
    if not correlated_chi2:
        # simply provided standard deviation (preferred pass)
        if ordinate_std is not None:
            temp: np.ndarray = ordinate_std
        
        # provided covariance in case standard deviation is not given (preferred pass)
        elif ordinate_cov is not None:
            temp: np.ndarray = np.diag(ordinate_cov)

        # standard deviation not provided, doing an unweighted fit 
        else:
            temp: np.ndarray = np.ones_like(ordinate_est)

    else: # correlated fit
            # simply provided covariance (preferred pass)
            if ordinate_cov is not None:
                temp: np.ndarray = ordinate_cov
            else:
                raise ValueError(f"No covariance given")

    ordinate_gvar = gv.gvar(
        # at least one is provided as checked above
        ordinate_est,
        # put the standard deviation/covariance as extraced into temp
        temp,
    )
    if prior is not None:
        ordinate_gvar = np.concatenate( (ordinate_gvar, *prior.values()) )

    param_combinations = list(itertools.product(*param_ranges.values()))
    param_keys         = list(param_ranges.keys())

    chi2 = np.zeros( len(param_combinations), dtype = float )
    
    # dof = Num Points - Num params = Num Points - Num vaied params - Num fixed params
    dof = len(abscissa) - len(param_keys) - len(fixed_params.keys())

    for param_combinationID, param_combination in enumerate(param_combinations): 
        params = { key: param_combination[i_key] for i_key, key in enumerate(param_ranges.keys()) }
        for key,val in fixed_params.items():
            params[key] = val
        chi2[param_combinationID] = gv.chi2( ordinate_gvar, model( abscissa, params ) ) / dof

    chi2 = chi2.reshape(len(param_ranges[param_keys[0]]), len(param_ranges[param_keys[1]]))

    fig = plt.figure(figsize = (34,16))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    contour_axs = fig.add_subplot(gs[0, :])
    param_1_axs = fig.add_subplot(gs[1, 0])
    param_2_axs = fig.add_subplot(gs[1, 1])
  
    contour_axs.contour(
        param_ranges[param_keys[0]],
        param_ranges[param_keys[1]],
        chi2,
        levels = 50,
        linewidths=0.5, 
        colors='k',
    )
    cntr1 = contour_axs.contourf(
        param_ranges[param_keys[0]], 
        param_ranges[param_keys[1]], 
        chi2, 
        levels=50, 
        norm = colors.LogNorm( 
            vmin=np.min(chi2), 
            vmax=np.max(chi2), 
        ) if log_chi2 else None,
    )

    cbar = fig.colorbar(cntr1, ax=contour_axs, orientation='horizontal', pad=0.15)
    cbar.set_label(rf"$\chi^2 / \mathrm{{dof}}~[{dof}]$")
    
    contour_axs.set_xlabel(param_keys[0])
    contour_axs.set_ylabel(param_keys[1])
    
    param_1_axs.set_xlabel(param_keys[0])
    param_1_axs.set_ylabel(param_keys[1])

    for i in range(0, len(param_ranges[param_keys[1]]), len(param_ranges[param_keys[1]])//6 ):
        param_1_axs.plot( param_ranges[param_keys[0]], chi2[:,i], label = f"{param_keys[1]} = {param_ranges[param_keys[1]][i]:g}" )

    param_1_axs.set_xlabel(param_keys[0])
    param_1_axs.set_ylabel(r"$\chi^2 / \mathrm{dof}$")
    param_1_axs.legend(bbox_to_anchor=(-0.15, 1))
    if log_chi2:
        param_1_axs.set_yscale('log')

    for i in range(0, len(param_ranges[param_keys[0]]), len(param_ranges[param_keys[0]])//6 ):
        param_2_axs.plot( param_ranges[param_keys[1]], chi2[i,:], label = f"{param_keys[0]} = {param_ranges[param_keys[0]][i]:g}" )

    param_2_axs.set_xlabel(param_keys[1])
    param_2_axs.set_ylabel(r"$\chi^2 / \mathrm{dof}$")
    param_2_axs.legend(bbox_to_anchor=(1, 1))
    if log_chi2:
        param_2_axs.set_yscale('log')
    
    fig.tight_layout()

    return fig,[contour_axs, param_1_axs,param_2_axs], chi2

