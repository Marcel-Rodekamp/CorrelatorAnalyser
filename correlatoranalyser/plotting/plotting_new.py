import matplotlib.pyplot as plt
import numpy as np
from ..fitState import FitState
# from dataclasses import dataclass, fields
# from typing import Self, List, Dict
import gvar as gv

#=========================================================================================================
#
def plot_best_fits(
        *,
        fit_state: FitState,
        C: np.ndarray[gv.GVar],
        C_dim: int = None, #option to specify which dimension of the corelator should be plotted (in case of multidim. correlators)
        num_fits: int = 5,
        title: str = "Best fits"
        )-> tuple[plt.Figure,plt.Axes]:
    r"""!
        Function to plot the raw correlator data and the best fit results. For multidimensional correlator data
        the dimesion is specified with C_dim. Also the number of fits to plot can be changed (default 5) and the title.
        The function uses the matplotlib libary and return the Figure and Axes of the plot and the figure is saved in the
        Report folder.
        @param fit_state: FitState object which contains the fit results
        @param C : (multidim.) arrray containing the raw data of the correlator for the fit
        @param C_dim: specify which dimension should be plotted for multidimensional correlator data
        @param num_fits: to choose the number of fits default 5)
        @param title: choose a title for the plot 
    """

    #check if Correlator data is multidimensional
    if C.ndim >1:
        if C_dim is None:
            raise ValueError(f"The Correlator is multidimensional, but no C_dim is specified.")

    dim = np.shape(C)[0]
    #ToDo: warning if no C_dim set, but C multidim
    fig,axs = plt.subplots(1,1,figsize=(25,8))

    if C_dim is None:
        abscissa = np.arange(len(C))
    else:
        abscissa = np.arange(len(C[C_dim]))
    #plot the correlator and its error
    axs.errorbar(
        x = abscissa,
        y = gv.mean(C),
        yerr = gv.sdev(C),
        capsize = 4
    )
    # top_fits = [[[], []] for _ in range(num_fits)]
    for i,fit in enumerate(fit_state.fit_results):
        if i< num_fits:
            abscissa = np.arange(fit.ts,fit.te)
            ordinate = fit.eval(abscissa)
            nstates = fit.fcn.Nstates
            line, = axs.plot(
                abscissa,
                ordinate["est"],
                "-",
                 label = rf"$N_\text{{states}}={nstates}, ({abscissa[0]},{abscissa[-1]}) " \
                         rf"\chi^2/_\mathrm{{dof}}~[\mathrm{{dof}}] = {fit.chi2/fit.dof:g}~[{fit.dof:g}], " \
                         rf"\text{{AIC}} = {fit.AIC:g} $"
            )
            axs.fill_between(
            abscissa,
            ordinate["est"] + ordinate["err"], 
            ordinate["est"] - ordinate["err"], 
            color = line.get_color(),
            alpha = 0.4
        )
    axs.set_ylabel(r"$C(\tau)$",fontsize=18)
    axs.set_xlabel(r"$\tau/a$",fontsize=18)
    axs.tick_params(axis="x", labelsize=12)
    axs.tick_params(axis="y", labelsize=12)
    axs.set_title(title)
    axs.legend(fontsize=18)
    axs.set_yscale('log')
    axs.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8)
    axs.minorticks_on()  # Aktiviert die Minor-Ticks
    axs.grid(True, which='minor', color='lightgray', linestyle=':', linewidth=0.5)



    filename = f'./Report/{num_fits}_best_fits.png'
    fig.savefig(filename)

    return fig, axs

            

#========================================================================================================
#            
def plot_fit_overview(
    fit_state:FitState    
    )-> tuple[plt.Figure,plt.Axes]:
    
    fig,axs = plt.subplots(1,1,figsize=(25,8))
    number_of_fits = len(fit_state.fit_results)

    params_est = np.zeros(number_of_fits)
    params_err = np.zeros(number_of_fits)
    nstates = np.zeros(number_of_fits, dtype = int)
    AICs = np.zeros(number_of_fits)
    
    resample_flag = fit_state.fit_results[0].has_resamples()

    if resample_flag:
        Nbst = fit_state.fit_results[0].Nres
        params_bst = np.zeros( (Nbst,number_of_fits))
        AICs_bst = np.zeros( (Nbst, number_of_fits))
    
    # Remove Fits with AIC>1000, as they don't have an impact on the result
    # but create bad pictures
    removesIDs = []

    for fitID, fit in enumerate(fit_state.fit_results):
        AIC = fit.AIC

        if AIC > 1000:
            removesIDs.append(fitID)
            continue
        # if param_key not in fit.result_params("est").keys:
        #     removesIDs

    return fig,axs

    



# @dataclass
# class DataPlotter:



#     data_complete: FitState


    # def __init__(self, input_data: FitState ):
    #     """
    #     Initialize the DataPlotter with a dataset.

    #     Parameters:
    #     @param data: FitState containing the fit results and averaged parameters
    #     """
    #     if not isinstance(input_data, FitState):
    #         raise TypeError(f"Expected FitState, got {type(input_data).__name__}")
    #     # if type(input_data) != FitState:
    #     #     print("Wrong input type!")
    #     # print(input_data.fit_results[0].te)
    #     self.data_complete = input_data
        
    
    #plot the top 5 fit results with the smallest AIC
# def plot_best_fits(self,
#     C: np.ndarray[gv.GVar], #corelator data
#     #abscissa: np.ndarray = None,
#     figAxTuple: tuple[plt.Figure, plt.Axes] = None,
#     title: str = "",
#     ylabel: str = r"$C(\tau)$",
#     xlabel: str = r"$^{\tau}/_{\delta}$",
#     no_fits: int = 5 #no. of fits to plot
#     # label: str = None,
#     # connectDots: bool = False,
#     # color=None,
#     ) -> tuple[plt.Figure, plt.Axes]:
#     r"""
#     Plot Correlator Data

#     Arguments:
        
#         optional:
#         @param title: choose own title
#         @param ylabel: choose label for y-axis
#         @param xlabel: choose label for x-axis
#     """

#     # retrieve the plt.Figure,plt.Axes
#     if figAxTuple is None:
#         fig, ax = plt.subplots(1, 1)
#     else:
#         fig = figAxTuple[0]
#         ax = figAxTuple[1]

#     # define the marker and check if the dots should be connected
#     fmt = "."
#     # if connectDots:
#     #     fmt += ":"

#     # # set the line color
#     # if self.color is None:
#     #     # use the default color cycle whichs colors are defined in mplStyle.py
#     #     self.color = None

#     # actually plot the data the color uses
#     ax.errorbar(
#         x=np.arange(0,len(C)),
#         y=gv.mean(C),
#         yerr=gv.sdev(C),
#         fmt=fmt,
#         capsize=2,
#         # color=color,
#         # label=label,
#     )


#     #read in the best Fits from data_complete
    # top_fits = [[[], []] for _ in range(no_fits)]
    # for i,fit in enumerate(top_fits):
    #     # print("type now",self.data_complete)
    #     nstates= self.data_complete.fit_results[i].fcn.Nstates
    #     print(nstates) 
    #     print(self.data_complete.fit_results[i].best_fit_param)
    #     fit[0] = np.arange(self.data_complete.fit_results[i].ts,self.data_complete.fit_results[i].te) #x-data
    #     fit[1] = self.data_complete.fit_results[i].eval(fit[0])["est"] #y-data
    #     #y-error?
    #     # ax.set_label( f'AIC {self.data_complete.fit_results[i].AIC}')
    #     # label = rf'$ N_\text{{states}}{nstates} [{fit[0][0]},{fit[0][-1]}]:\'
    #     #         + rf'\text{AIC} {np.round(self.data_complete.fit_results[i].AIC,2)}$'
    #     label = rf"$N_\text{{states}}={nstates}, ({fit[0][0]},{fit[0][-1]}) " \
    #     rf"\chi^2/_\mathrm{{dof}}~[\mathrm{{dof}}] = {self.data_complete.fit_results[i].chi2/self.data_complete.fit_results[i].dof:g}~[{self.data_complete.fit_results[i].dof:g}], " \
    #     rf"\text{{AIC}} = {self.data_complete.fit_results[i].AIC:g} $"
    #     ax.plot(fit[0],fit[1],label= label)
        # $N_\text{{states}}=

#     # set the absissa if not yet set
#     # if abscissa is None:
#     #     abscissa = np.arange(C.shape[0])



#     ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8)
#     ax.minorticks_on()  # Aktiviert die Minor-Ticks
#     ax.grid(True, which='minor', color='lightgray', linestyle=':', linewidth=0.5)

#     ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel(xlabel)
#     ax.set_yscale("log")
#     ax.legend()
#     # if self.label is not None:
#     #     ax.legend()

#     return fig, ax



