import matplotlib.pyplot as plt
import numpy as np
from ..fitState import FitState
from dataclasses import dataclass, fields
from typing import Self, List, Dict


@dataclass
class DataPlotter:

    data_complete: FitState


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
    def plotTopFits(self,
        # C: np.ndarray,
        # abscissa: np.ndarray = None,
        figAxTuple: tuple[plt.Figure, plt.Axes] = None,
        title: str = "",
        ylabel: str = r"$C(\tau)$",
        xlabel: str = r"$^{\tau}/_{\delta}$",
        no_fits: int = 5 #no. of fits to plot
        # label: str = None,
        # connectDots: bool = False,
        # color=None,
    ) -> tuple[plt.Figure, plt.Axes]:
        r"""
        Plot Correlator Data

        Arguments:
            
            optional:
            @param title: choose own title
            @param ylabel: choose label for y-axis
            @param xlabel: choose label for x-axis
        """

        # retrieve the plt.Figure,plt.Axes
        if figAxTuple is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = figAxTuple[0]
            ax = figAxTuple[1]

        # define the marker and check if the dots should be connected
        fmt = "."
        # if connectDots:
        #     fmt += ":"

        # # set the line color
        # if self.color is None:
        #     # use the default color cycle whichs colors are defined in mplStyle.py
        #     self.color = None


        #read in the Correlator and data of the top 5 fits from data_complete
        # from where read in correlator? ord_est not saved in FitState/FitResult--> set abscissa
        top_fits = [[[], []] for _ in range(no_fits)]
        for i,fit in enumerate(top_fits):
            # print("type now",self.data_complete)
            fit[0] = np.arange(self.data_complete.fit_results[i].ts,self.data_complete.fit_results[i].te) #x-data
            fit[1] = self.data_complete.fit_results[i].eval(fit[0])["est"] #y-data
            #y-error?
            # ax.set_label( f'AIC {self.data_complete.fit_results[i].AIC}')
            ax.plot(fit[0],fit[1],label= f'[{fit[0][0]},{fit[0][-1]}]:AIC {np.round(self.data_complete.fit_results[i].AIC,2)}')
            

        # set the absissa if not yet set
        # if abscissa is None:
        #     abscissa = np.arange(C.shape[0])

        
        # actually plot the data the color uses
        # ax.errorbar(
        #     x=abscissa,
        #     y=gv.mean(C),
        #     yerr=gv.sdev(C),
        #     fmt=fmt,
        #     capsize=2,
        #     color=color,
        #     label=label,
        # )
        ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.8)
        ax.minorticks_on()  # Aktiviert die Minor-Ticks
        ax.grid(True, which='minor', color='lightgray', linestyle=':', linewidth=0.5)

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_yscale("log")
        ax.legend()
        # if self.label is not None:
        #     ax.legend()

        return fig, ax



