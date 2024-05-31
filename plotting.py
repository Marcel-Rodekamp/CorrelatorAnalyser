# We offload all numerical work to numpy
import numpy as np 

# We plot using matplotlib
import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

# we typically expect arrays of gvar
import gvar as gv 

# sometimes we have to deal with fit results 
import lsqfit

# globally load a style defined though mplStyle.py
import mplStyle as style

from fitter import Fitter

def plotCorrelator(
        C: np.ndarray, abscissa: np.ndarray = None, figAxTuple:  (plt.Figure, plt.Axes) = None, title:str = '', ylabel: str = r"$C(\tau)$", xlabel: str = r"$^{\tau}/_{\delta}$", label: str = None, connectDots: bool = False, color = None) -> (plt.Figure, plt.Axes):
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
    if label is not None:
        ax.legend()

    return fig, ax

def plotFit(fit: Fitter, figAxTuple:  (plt.Figure, plt.Axes) = None, plotData: bool = True, title:str = '', Nplot = 100, ylabel: str = r"$C(\tau)$", xlabel: str = r"$^{\tau}/_{\delta}$", label: str = None, color = None, putFitReport: bool = True, zorder=0) -> (plt.Figure, plt.Axes):
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
        lw = 1,
        zorder = zorder,
    )

    # plot the 1-sigma confidence interval
    ax.fill_between(
        abscissa, 
        gv.mean(ordinate) + gv.sdev(ordinate),
        gv.mean(ordinate) - gv.sdev(ordinate),
        color = line.get_color(),
        alpha = 0.2,
        zorder = zorder
    )

    # plot the 2-sigma confidence interval
    ax.fill_between(
        abscissa, 
        gv.mean(ordinate) + 2*gv.sdev(ordinate),
        gv.mean(ordinate) - 2*gv.sdev(ordinate),
        color = line.get_color(),
        alpha = 0.1,
        zorder = zorder
    )

    # potentially ad a text-box below the plot which summarizes the fit result
    if putFitReport:
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=2, alpha=0.7)
        text = fit.report()#.replace('_', ' ').replace('@', '\@' )

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

def plotParameterPerAIC( 
    paramKey: str, fits: list[Fitter], title:str = '', ylabel:str = None, colorModelAvg = None
):
    r"""

    """
    fig, axs = plt.subplots(1,1)


    # ##############################################
    # 1. Collect Data
    # ##############################################
    if len(fits) == 0: return fig,axs

    # retrieve data to plot
    params_est = np.zeros(len(fits))
    params_err = np.zeros(len(fits))
    nstates    = np.ones((len(fits),2),dtype = int)
    AICs       = np.zeros(len(fits))

    removedIDs = []
    # We assume all fits have bootstrap samples if the first fit has them
    hasBootstraps = fits[0].bootstrapFlag

    if hasBootstraps:
        Nbst = fits[0].Nbst
        params_bst = np.zeros( (Nbst,len(fits)) )
        AICs_bst   = np.zeros( (Nbst,len(fits)) )

    for fitID, fit in enumerate(fits):
        AIC = fit.AIC()

        if AIC > 1000:
            removedIDs.append(fitID)
            continue

        bestParams = fit.bestParameter()
        if paramKey not in bestParams['est'].keys(): 
            continue
        params_est[fitID] = bestParams['est'][paramKey]
        params_err[fitID] = bestParams['err'][paramKey]
        # if the models have different nstates, we would like to add different markers
        try:
            nstates[fitID] = (
                fit.model.Nstates_D,
                fit.model.Nstates_I
            )
            
        except:
            pass
        AICs[fitID] = AIC

        if hasBootstraps:
            params_bst[:,fitID] = bestParams['bst'][paramKey]
            AICs_bst[:,fitID] = fit.AIC(getBst=True)
    hasDifferentStates = np.any(nstates!=1)

    params_est = np.delete(params_est,removedIDs)
    params_err = np.delete(params_err,removedIDs)
    nstates = np.delete(nstates,removedIDs, axis = 0)
    AICs = np.delete(AICs,removedIDs)

    if hasBootstraps:
        params_bst = np.delete(params_bst,removedIDs,axis=1)
        AICs_bst = np.delete(AICs_bst,removedIDs,axis=1)

    AICs -= np.min(AICs)
    if hasBootstraps:
        AICs_bst -= np.min(AICs_bst,axis=1)[:,None]

    # ##############################################
    # 2. Compute Model Average
    # ##############################################

    # compute model average
    weights =np.exp(-0.5 * AICs)

    modelAvg_est = np.average(params_est, weights = weights)

    if hasBootstraps:
        weights_bst = np.exp(-0.5 * AICs_bst)

        modelAvg_err = np.std( 
            np.average(params_bst, weights = weights_bst,axis=1), 
            axis = 0
        )
    else:
        modelAvg_err = np.average(params_err, weights = weights)

    # ##############################################
    # 3. Evaluate Histograms 
    # ##############################################

    # compute histogram on central values
    numBins = 10
    params_hist, params_bins = np.histogram(params_est, bins = numBins)

    bins = 10**np.linspace(
        np.min(np.log10(weights)), 0, 20 + 1
    )
    print(bins)
    weights_hist, weights_bins = np.histogram(weights, bins = bins)

    # compute uncertainty of the histogram if bootstraps are available
    if hasBootstraps:
        params_hist_bst = np.zeros( (Nbst, len(params_hist) )  )
        weights_hist_bst = np.zeros( (Nbst, len(weights_hist) )  )
        for nbst in range(Nbst):
            params_hist_bst[nbst], _ = np.histogram(params_bst, bins = params_bins)
            weights_hist_bst[nbst], _   = np.histogram(weights_bst, bins = weights_bins)
        params_hist_err = np.std( params_hist_bst, axis = 0)
        weights_hist_err   = np.std( weights_hist_bst, axis = 0)

    # ##############################################
    # 4. Sort the AIC and Parameters
    # ##############################################

    # sort the data AIC
    sorter    = np.argsort(weights)
    weights   = weights[sorter]
    nstates   = nstates[sorter,:]
    params_est= params_est[sorter]
    params_err= params_err[sorter]
    
    # ##############################################
    # 5. Plot Parameters vs AIC
    # ##############################################

    #axs[0].errorbar( 
    #    AICs, params_est, yerr=params_err, fmt='.', color = style.MAIN_COLORS['primary'], capsize=2,
    #    label = 'Fit Result'
    #)
    markers = [".","+","^","s","p","*","h","v","x","<",">","D","o"]
    covered = []
    for fitID in range(len(weights)):
        if np.max(nstates[fitID]) in covered:
            axs.plot( 
                weights[fitID], 
                params_est[fitID], 
                markers[np.max(nstates[fitID])-1], 
                color = style.MAIN_COLORS['primary']
            )
        else:
            axs.plot( 
                weights[fitID], 
                params_est[fitID], 
                markers[np.max(nstates[fitID])-1], 
                color = style.MAIN_COLORS['primary'],
                label = rf"$({nstates[fitID][0]},{nstates[fitID][1]})$ States"
            )

            covered.append(np.max(nstates[fitID]))

    # ##############################################
    # 6. Plot Model Average
    # ##############################################

    axs.axhline( y=modelAvg_est, xmin=0,xmax=1, 
            color = style.MAIN_COLORS['highlight'] if colorModelAvg is None else colorModelAvg,
            label = f"Model Average: {gv.gvar(modelAvg_est,modelAvg_err)}" )
    axs.axhspan( 
        modelAvg_est+modelAvg_err, 
        modelAvg_est-modelAvg_err, 
        color = style.MAIN_COLORS['highlight'] if colorModelAvg is None else colorModelAvg,
        alpha = 0.2
    )
    axs.legend()

    # ##############################################
    # 7. Plot Histograms
    # ##############################################

    ax_param = axs.inset_axes([1.02, 0, 0.25, 1], sharey=axs)
    ax_param.barh(
        y = params_bins[:-1], 
        width = params_hist,
        height = 0.99*(params_bins[1] - params_bins[0]),
        align = 'edge',
        color=style.MAIN_COLORS['primary'],
        #label = f'{len(fits)} Fits',
    )
    if hasBootstraps:
        ax_param.errorbar( 
            params_hist,  
            params_bins[:-1] + 0.5*(params_bins[1] - params_bins[0]) , xerr=params_hist_err, fmt='none', capsize = 3, elinewidth=4, color = style.MAIN_COLORS['highlight'], line=None )


    #ax_param.set_xticks([])
    ax_param.tick_params(axis='y', labelleft=False)
    ax_param.tick_params(axis='x', labelbottom=False)
    #ax_param.xaxis.set_label_position('top') 
    ax_param.xaxis.set_ticks_position('top') 
    ax_param.xaxis.set_major_locator(MaxNLocator(integer=True, nbins = 'auto'))

    #ax_param.legend()

    ax_weights = axs.inset_axes([0, -0.275, 1, 0.25], sharex = axs)
    ax_weights.bar(
        x = weights_bins[:-1], 
        height = weights_hist,
        width = 0.99*np.diff(weights_bins),
        align = 'edge',#'center',
        color=style.MAIN_COLORS['complementary'],
    )
    if hasBootstraps:
        ax_weights.errorbar( 
            np.exp(
                (np.log(weights_bins[:-1]) + np.log(weights_bins[1:])) / 2 
            ),
            weights_hist, yerr=weights_hist_err, fmt='none', capsize = 3, elinewidth=4, color = style.MAIN_COLORS['highlight'])

    #ax_weights.set_yticks([])
    #ax_weights.tick_params(axis='y', labelleft=False)
    ax_weights.tick_params(axis='x', labelbottom=False)
    ax_weights.yaxis.set_major_locator(MaxNLocator(integer=True, nbins = 'auto'))

    ax_weights.invert_yaxis()

    axs.text(
        x = 1.125,
        y = -0.125,
        s = f'{len(fits)} Fits',
        ha='center',
        va='center',
        fontsize = 28,
        bbox=dict(
            boxstyle="Round,pad=0.3", 
            fc="white", 
            ec="grey", 
            #lw=2, 
            alpha=0.3,
        ),
        transform=axs.transAxes,
    )


    if ylabel is None:
        axs.set_ylabel(paramKey)
    else:
        axs.set_ylabel(ylabel)

    axs.set_xlabel(r"$p(m\vert D)$", labelpad=20)

    axs.set_xscale('log')
    axs.xaxis.set_label_position('top') 
    axs.xaxis.set_ticks_position('top') 

    axs.set_title(title)

    return fig,axs

