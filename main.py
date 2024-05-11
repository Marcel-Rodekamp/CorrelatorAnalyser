# We offload all numerical work to numpy
import numpy as np

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

# Logging is done by the standard module 
import logging 

# CLI arguments are handled/defined using 
import argparse

# Files are handled using pathlib
from pathlib import Path

# We primarily work with hd5f data
import h5py as h5

# we still import plt to access interactive show windows etc
import matplotlib.pyplot as plt

# we track progress using progressbars 
from tqdm import tqdm


# Some plotting functions are defined in
import plotting

# Fitting is done using the fitter interface defined here
import fitter 

import fitModels


def testData(
    T: float, Nt: int, Nconf: int, As_f: np.ndarray, Es_f: np.ndarray, As_b: np.ndarray, Es_b: np.ndarray, hasStN: bool = False) -> np.ndarray:
    r"""
        This generates a set of artificial data points mimicking a correlation function
        \f[
            C(t) = \sum_{n=0}^{Nstates-1} A_n exp(-t E_n) 
                 + \sum_{n=0}^{Nstates-1} A_n exp(-t E_n^b)
        \f]
        where t takes Nt separate values. 

        The generated data uses 
        \f[
            E^f_n = 0.5 \cdot ( 1 + n ) \\
            E^b_n = ( 1 + 0.5 n ) \\
            A^f_n = 1/n \\
            A^b_n = exp(-0.5*Nt) / n  
        \f]

        Arguments:
            - Nt: int, number of time slices
            - Nconf: int, number of data points
            - Nstates: int, number of states (exponential terms) resolved in the data. Default = 3
            - backpropagating: bool, A flag whether backpropagating states are included or not . Default = True
            
        Returns:
            Fictitious correlator data
            - np.array(shape=(Nconf,Nt), dtype = float)
    """

    # we assume a stochastic scaling of the standard deviation as it is expected from a MCMC algorithm
    std = 1. / np.sqrt( Nconf )
    
    # Check that we have the same number of states in A_b and E_b
    if len(As_f) != len(Es_f): 
        raise RuntimeError(f"Length of As_f ({len(As_f)}) and Es_f ({len(Es_f)}) must match!")

    # Check that we have the same number of states in A_b and E_b
    if len(As_f) != len(Es_f): 
        raise RuntimeError(f"Length of As_f ({len(As_f)}) and Es_f ({len(Es_f)}) must match!")

    # define an artifical lattice spacing 
    delta = T/Nt

    # define abscissa with lattice spacing = 1
    abscissa = np.arange( Nt )

    # compute the central values (underlying truth) of the data
    corrUnderlying = np.zeros( Nt, dtype = float )

    # forward propagating states
    for A,E in zip(As_f,Es_f):
        corrUnderlying += A * np.exp( - E * abscissa * delta)

    # backward propagating states
    for A,E in zip(As_b,Es_b):
        corrUnderlying += A * np.exp(   E * abscissa * delta)

    # Add a signal to noise porblem ~ exp(- t)
    if hasStN:
        std = std * np.exp(np.max(np.concatenate((Es_f,Es_b))) * (corrUnderlying[0] - corrUnderlying) )
    else:
        std = np.full_like(corrUnderlying, std)

    # define the actual data
    corrData = np.random.normal(
        loc = corrUnderlying,
        scale = std,
        size = (Nconf,Nt)
    )

    return corrData, corrUnderlying, abscissa, delta 
# end def testData 

def main(CLIargs: argparse.Namespace) -> None:
    r"""
        Main entry to this code. 
        Arguments: 
            - CLIargs: argparse.Namespace, command-line arguments as parsed by argparse. Required constituents:
                1. cacheFolder: str, a folder to store analysis results (as hdf5 format).
                2. reportFolder: str, a folder to store plots and text files summarizing the fit results.

        Requires a logger `logger` to be defined on the global level.
    """
    global logger

    logger.info(CLIargs)

    # define the Cache folder in which we cache the fit results
    CacheFolder = Path(CLIargs.cacheFolder)

    # make sure that the folder exists
    CacheFolder.mkdir(parents=True, exist_ok=True)

    # define the Cache folder in which we cache the fit results
    ReportFolder = Path(CLIargs.reportFolder)

    # make sure that the folder exists
    ReportFolder.mkdir(parents=True, exist_ok=True)
    
    # Create fake data
    # - corrData: np.array(Nconf,Nt), sampled data, normally distributed and uncorrelated
    # - corrUnderlying: np.array(Nt), underlying true values from which we sampled
    # - abscissa: np.arrange(Nt), the array which we can use for plotting
    # - delta: float, lattice spacing = T / Nt
    corrData, corrUnderlying, abscissa, delta  = testData(
        T = CLIargs.T, Nt = CLIargs.Nt, Nconf = CLIargs.Nconf, 
        As_f = [0.6,0.4], Es_f = [1,1.5], 
        As_b = [0.006*np.exp(-CLIargs.T*2), 0.004*np.exp(-CLIargs.T*4)], Es_b = [2,4], 
        hasStN = CLIargs.StN,
    )

    # estimate data
    if CLIargs.Nbst is None:
        # if no bootstrap is desired we can simplify using gvar which automatically
        Cest = corrData.mean(axis=0)
        Cerr = corrData.std(axis=0)/np.sqrt(CLIargs.Nconf)
        Cgvar = gv.gvar(Cest,Cerr)
    else:
        bootstrapIDs = np.random.randint(
            0, CLIargs.Nconf, size = (CLIargs.Nbst, CLIargs.Nconf)
        )

        Cbst = np.zeros( (CLIargs.Nbst, CLIargs.Nt ) )
        for nbst in tqdm(range(CLIargs.Nbst), desc = 'Bootstrapping Correlator'):
            sample = corrData[ bootstrapIDs[nbst] ]
            Cbst[nbst] = np.mean( sample,axis=0 )
        Cest = np.mean(Cbst,axis=0)
        Cerr = np.std(Cbst,axis=0)
        Cgvar = gv.gvar(Cest,Cerr)
    # end else Nbst
    
    # plot the data
    fig, ax = plotting.plotCorrelator(
        C = Cgvar,
        abscissa = abscissa,
        color = plotting.style.MAIN_COLORS['primary'],
        connectDots = True,
        label = "Sampled Data"
    )
    ax.plot(
        abscissa,
        corrUnderlying,
        '-k',
        label = "Underlying"
    )
    ax.legend()

    fig.savefig( ReportFolder/"correlator.pdf" )

    fit = fitter.Fitter("ExponentialFitter") \
            .setFitModel(fitModels.SimpleSumOfExponentialsModel(Nstates=2)) \
            .setPrior(fitModels.SimpleSumOfExponentialsFlatPrior(Nstates=2))

    fit_1 = fit( abscissa[:], Cgvar[:], createCopy = True )
    fit_2 = fit( abscissa[5:22], Cgvar[5:22], createCopy = True )

    logger.info(f"fit_1={fit_1.report()}")
    logger.info(f"fit_2={fit_2.report()}")

    fig, ax = plotting.plotCorrelator(
        C = Cgvar,
        abscissa = abscissa,
        color = plotting.style.MAIN_COLORS['primary'],
        connectDots = False,
        label = "Sampled Data"
    )

    fig, ax = plotting.plotFit(
        fit = fit_1,
        figAxTuple = (fig,ax),
        plotData = False,
        label = rf"{fit_1.short()}",
        color = plotting.style.MAIN_COLORS['complementary'],
        putFitReport = fit_1.AIC() < fit_2.AIC()
    )

    fig, ax = plotting.plotFit(
        fit = fit_2,
        figAxTuple = (fig,ax),
        plotData = False,
        label = rf"{fit_2.short()}",
        color = plotting.style.MAIN_COLORS['highlight'],
        putFitReport = fit_1.AIC() > fit_2.AIC()
    )

    ax.legend()

    fig.savefig( ReportFolder/"fits.pdf",bbox_inches="tight" )

# end def main

if __name__ == "__main__":
    # define a logger
    logger = logging.getLogger(__name__)

    # and set the default logging interface 
    logging.basicConfig(format='Analysis %(asctime)s~> %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
    
    # Define the command-line arguments
    CLIParser = argparse.ArgumentParser(
        prog='Data Analyser'
    )

    CLIParser.add_argument('-T', type = float, required = True )
    CLIParser.add_argument('-Nt', type = int, required = True )
    CLIParser.add_argument('-Nconf', type = int, required = True )
    CLIParser.add_argument('--StN', action='store_true')
    CLIParser.add_argument('-Nbst', type = int, required = False, default = None )
    CLIParser.add_argument('-cacheFolder', type = str, required = False, default = "Cache")
    CLIParser.add_argument('-reportFolder', type = str, required = False, default = "Report")

    # Parse the CLI arguments
    CLIargs = CLIParser.parse_args()

    main(CLIargs)

