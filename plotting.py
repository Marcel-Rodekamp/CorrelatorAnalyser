# We offload all numerical work to numpy
import numpy as np 

# We plot using matplotlib
import matplotlib.pyplot as plt

# we typically expect arrays of gvar
import gvar as gv 

# sometimes we have to deal with fit results 
import lsqfit

# globally load a style defined though mplStyle.py
import mplStyle as style

from fitter import Fitter

def plotCorrelator(
        C: np.ndarray[ gv.gvar ], abscissa: np.ndarray = None, figAxTuple:  (plt.Figure, plt.Axes) = None, title:str = '', ylabel: str = r"$C(\tau)$", xlabel: str = r"$\tau$", label: str = None, connectDots: bool = False, color = None) -> (plt.Figure, plt.Axes):
    r"""
        Plot Correlator Data 

        Arguments:
            - Cest: np.array( (Nt,), type = float ), central value points with Nt time slices
            - Cerr: np.array( (Nt,), type = float ), Standard Deviation (Plotted as errorbars)
    """

    # retrieve the plt.Figure,plt.Axes
    if figAxTuple is None:
        fig, ax = plt.subplots(1,1)
    else :
        fig = figAxTuple[0]
        ax  = figAxTuple[1] 

    # set the absissa if not yet set
    if abscissa is None:
        abscissa = np.arange( C.shape[0] )

    # define the marker and check if the dots should be connected
    fmt = '.'
    if connectDots:
        fmt+=':'

    # set the line color 
    if color is None:
        # use the default color cycle whichs colors are defined in mplStyle.py
        color = None 
    
    # actually plot the data the color uses 
    ax.errorbar(
        x = abscissa,
        y = gv.mean(C),
        yerr = gv.sdev(C),
        fmt = fmt,
        capsize = 2,
        color = color,
        label = label
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale('log')
    ax.legend()

    return fig, ax

def plotFit(fit: Fitter, figAxTuple:  (plt.Figure, plt.Axes) = None, plotData: bool = True, title:str = '', Nplot = 100, ylabel: str = r"$C(\tau)$", xlabel: str = r"$\tau$", label: str = None, color = None, putFitReport: bool = True, zorder=0) -> (plt.Figure, plt.Axes):
    r"""
        Plot fit restults.

        Arguments:

    """

    # retrieve the plt.Figure,plt.Axes
    if figAxTuple is None:
        fig, ax = plt.subplots(1,1)
    else :
        fig = figAxTuple[0]
        ax  = figAxTuple[1] 

    # potentially plot the data we fitted against
    if plotData:
        fig,ax=plotCorrelator(
            C = fit.fitResult.y,
            abscissa = fit.fitResult.x,
            figAxTuple=(fig,ax),
            title=title,
            ylabel=ylabel,
            xlabel=xlabel,
            connectDots = False,
            color = style.MAIN_COLORS["primary"]
        )


    # evaluate the model (we could provide an abscissa or let it auto compute here)
    # abscissa is a np.array
    # ordinate is a np.array of gv.gvars
    abscissa, ordinate = fit.evalModel()

    # plot the central value of the fit
    line, = ax.plot(
        abscissa, gv.mean(ordinate), '-',
        color = color,
        label = label,
        zorder = zorder
    )

    # plot the 1-sigma confidence interval
    ax.fill_between(
        abscissa, 
        gv.mean(ordinate) + gv.sdev(ordinate),
        gv.mean(ordinate) - gv.sdev(ordinate),
        color = line.get_color(),
        alpha = 0.4,
        zorder = zorder
    )

    # plot the 2-sigma confidence interval
    ax.fill_between(
        abscissa, 
        gv.mean(ordinate) + 2*gv.sdev(ordinate),
        gv.mean(ordinate) - 2*gv.sdev(ordinate),
        color = line.get_color(),
        alpha = 0.2,
        zorder = zorder
    )

    # potentially ad a text-box below the plot which summarizes the fit result
    if putFitReport:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=0.7)
        text = fit.report()

        ax.text(
            0.5, -0.1, 
            text,
            ha='center',
            va='top',
            fontsize = 10,
            bbox=bbox_props,
            transform=ax.transAxes,
        )
    return fig, ax

def plotParameterPerAIC( paramKey: str, fits: list[Fitter], title:str = '', ylabel:str = None):
    r"""

    """
    fig, axs = plt.subplots(2,1, sharex = True, gridspec_kw={'height_ratios': [0.75,0.25]})


    # ##############################################
    # 1. Collect Data
    # ##############################################

    # retrieve data to plot
    params_est = np.zeros(len(fits))
    params_err = np.zeros(len(fits))
    AICs   = np.zeros(len(fits))

    # We assume all fits have bootstrap samples if the first fit has them
    hasBootstraps = fits[0].bootstrapFlag

    if hasBootstraps:
        Nbst = fits[0].Nbst
        params_bst = np.zeros( (Nbst,len(fits)) )
        AICs_bst   = np.zeros( (Nbst,len(fits)) )

    for fitID, fit in enumerate(fits):
        params_est[fitID] = fit.bestParameter()['est'][paramKey]
        params_err[fitID] = fit.bestParameter()['err'][paramKey]
        AICs[fitID] = fit.AIC()

        if hasBootstraps:
            params_bst[:,fitID] = fit.bestParameter()['bst'][paramKey]
            AICs_bst[:,fitID] = fit.AIC(getBst=True)

    # ##############################################
    # 2. Evaluate Histograms 
    # ##############################################

    # compute histogram on central values
    numBins = 10
    params_hist, params_bins = np.histogram(params_est, bins = numBins, density = True)
    AICs_hist, AICs_bins = np.histogram(params_est, bins = numBins, density = True)
    
    # compute uncertainty of the histogram if bootstraps are available
    if hasBootstraps:
        params_hist_bst = np.zeros( (Nbst, len(params_hist) )  )
        AICs_hist_bst = np.zeros( (Nbst, len(AICs_hist) )  )
        for nbst in range(Nbst):
            params_hist_bst[nbst], _ = np.histogram(params_bst, bins = params_bins, density = True)
            AICs_hist_bst[nbst], _   = np.histogram(AICs_bst, bins = AICs_bins, density = True)
        params_hist_err = np.std( params_hist_bst, axis = 0)
        AICs_hist_err   = np.std( AICs_hist_bst, axis = 0)

    # ##############################################
    # 3. Compute Model Average
    # ##############################################

    # compute model average
    weights =np.exp(-0.5 * (AICs - np.min(AICs)))
    modelAvg_est = np.average(params_est, weights = weights)

    if hasBootstraps:
        modelAvg_err = np.std( 
            np.average(params_bst, weights = weights,axis=1), 
            axis = 0
        )
    else:
        modelAvg_err = np.average(params_err, weights = weights)

    # ##############################################
    # 4. Sort the AIC and Parameters
    # ##############################################

    # sort the data AIC
    sorter = np.argsort(AICs)
    AICs  = AICs[sorter]
    params_est= params_est[sorter]
    params_err= params_err[sorter]
    
    # ##############################################
    # 5. Plot Parameters vs AIC
    # ##############################################

    #axs[0].errorbar( 
    #    AICs, params_est, yerr=params_err, fmt='.', color = style.MAIN_COLORS['primary'], capsize=2,
    #    label = 'Fit Result'
    #)
    axs[0].plot( AICs, params_est, '.', color = style.MAIN_COLORS['primary'])
    axs[1].plot( AICs, AICs, '.',  color = style.MAIN_COLORS['complementary'] )

    # ##############################################
    # 6. Plot Model Average
    # ##############################################

    axs[0].axhline( y=modelAvg_est, xmin=0,xmax=1, color = style.MAIN_COLORS['highlight'],label = "Model Average"  )
    axs[0].axhspan( 
        modelAvg_est+modelAvg_err, 
        modelAvg_est-modelAvg_err, 
        color = style.MAIN_COLORS['highlight'],
        alpha = 0.2
    )

    # ##############################################
    # 7. Plot Histograms
    # ##############################################

    ax_param = axs[0].inset_axes([1.0, 0, 0.25, 1], sharey=axs[0])
    ax_param.barh(
        y = params_bins[:-1], 
        width = params_hist,
        height = 0.99*(params_bins[1] - params_bins[0]),
        align = 'center',
        color=style.MAIN_COLORS['primary'],
        label = f'Unweighted Histogram({len(fits)} Fits)',
    )
    if hasBootstraps:
        ax_param.errorbar( params_hist,  params_bins[:-1], xerr=params_hist_err, fmt='none', capsize = 2, color = 'k', line=None )


    ax_param.set_xticks([])
    ax_param.tick_params(axis='y', labelleft=False)
    ax_param.tick_params(axis='x', labelbottom=False)

    ax_param.legend()

    ax_AIC = axs[1].inset_axes([1.0, 0, 0.25, 1], sharey=axs[1])
    ax_AIC.barh(
        y = AICs_bins[:-1], 
        width = AICs_hist,
        height = 0.99*(AICs_bins[1] - AICs_bins[0]),
        align = 'center',
        color=style.MAIN_COLORS['complementary'],
    )
    if hasBootstraps:
        ax_AIC.errorbar( AICs_hist,  AICs_bins[:-1], xerr=AICs_hist_err, fmt='none', capsize = 2, color = 'k' )

    ax_AIC.set_xticks([])
    ax_AIC.tick_params(axis='y', labelleft=False)
    ax_AIC.tick_params(axis='x', labelbottom=False)


    if ylabel is None:
        axs[0].set_ylabel(paramKey)
    else:
        axs[0].set_ylabel(ylabel)

    axs[1].set_xlabel(r"AIC")
    axs[1].set_ylabel(r"AIC")

    axs[0].set_title(title)

    return fig,axs

