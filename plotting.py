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

