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


def testData(T: float, Nt: int, Nconf: int, As_f: np.ndarray, Es_f: np.ndarray, As_b: np.ndarray, Es_b: np.ndarray, hasStN: bool = False) -> np.ndarray:
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

    # ##########################################################
    # Setting up the interface
    # ##########################################################

    # print the provided CLI arguments
    logger.info(CLIargs)

    # define the Cache folder in which we cache the fit results
    CacheFolder = Path(CLIargs.cacheFolder)

    # make sure that the folder exists
    CacheFolder.mkdir(parents=True, exist_ok=True)

    # define the Cache folder in which we cache the fit results
    ReportFolder = Path(CLIargs.reportFolder)

    # make sure that the folder exists
    ReportFolder.mkdir(parents=True, exist_ok=True)

    # ##########################################################
    # Data generation
    # ##########################################################
    
    # Create fake data, this could be reading in etc.
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

    # ##########################################################
    # Data Estimation
    # ##########################################################

    # Either bootstrap or compute via standard deviation
    if CLIargs.Nbst is None:
        # if no bootstrap is desired we can simplify using gvar which automatically
        Cest = corrData.mean(axis=0)
        Cerr = corrData.std(axis=0)/np.sqrt(CLIargs.Nconf)
    else:
        bootstrapIDs = np.random.randint(
            0, CLIargs.Nconf, size = (CLIargs.Nbst, CLIargs.Nconf)
        )

        Cbst = np.zeros( (CLIargs.Nbst, CLIargs.Nt ) )
        Cbrr = np.zeros( (CLIargs.Nbst, CLIargs.Nt ) )
        for nbst in tqdm(range(CLIargs.Nbst), desc = 'Bootstrapping Correlator'):
            sample = corrData[ bootstrapIDs[nbst] ]
            Cbst[nbst] = np.mean( sample,axis=0 )
            Cbrr[nbst] = np.std( sample,axis=0 )
        Cest = np.mean(Cbst,axis=0)
        Cerr = np.std(Cbst,axis=0)
    # end else Nbst
    
    # ##########################################################
    # Perform fits
    # ##########################################################
    
    # Here we set up various fits and execute them according to some rules that we define 
    # for the specific data. 
    # We explicitly make use of the prior and fit-models from fitModels.py.
    # If none of those fits your problem/desire you eventually have to create them yourself
    # simply by inheriting from the provided base-class

    # Collect the fit results in
    fitResultList = []

    # maximally allowed number of states
    maxStates = 2
    
    # define a fit endpoint, this is usually defined by a signal to noise point from the data. 
    te = 22

    # iterate over possible states
    for ns in range(1,maxStates+1):
        # Create the fitter for an ns-state fit:
        
        # 1. Define a raw fitter
        fit = fitter.Fitter("ExponentialFitter")

        # 2. Add a ns-state fit model. We want the energies in lattice units thus we add
        #    the lattice spacing delta (default = 1 if not given) into this. 
        #    Alternatively, the abscissa could be multiplied by delta in the first place.
        fit = fit.setFitModel(fitModels.SimpleSumOfExponentialsModel(Nstates=ns,delta=delta)) \

        # 3.1 Add a prior to the fit model
        fit = fit.setPrior(fitModels.SimpleSumOfExponentialsFlatPrior(Nstates=ns))
        
        # 3.2 or add start parameter to the fit model
        #fit = fit.setP0(fitModels.SimpleSumOfExponentialsP0(Nstates=ns))

        # 4. Don't do a correlated fit (there are no correlations in the testData) 
        #    default = True -> This requires the covariance in ordinate_err
        fit = fit.setCorrelated(False)

        # 5. Turn on bootstrapped fits. This requires the bootstrap samples in ordinate
        #    default = False
        fit = fit.setBootstrapped(CLIargs.Nbst)

        # iterate over starting point such that at least a point per parameter + 2 exists 
        # excluding the t=0 term
        for ts in np.arange( 1, te - 2*ns - 2 ):
            # perform the fit
            # We could also provide Ccov instead of Cerr to utilize a chi^2 with correlations
            #fitRes = fit( abscissa[ts:te], Cest[ts:te], Cerr[ts:te], createCopy = True )
            # alternatively, we just provide the bst samples and the covariance/uncertainty
            # is automatically computed
            fitRes = fit( abscissa[ts:te], Cbst[:,ts:te], Cerr[ts:te], createCopy = True )
            # We can even provide a (co)variance per bootstrap sample
            #fitRes = fit( abscissa[ts:te], Cbst[:,ts:te], Cbrr[:,ts:te], createCopy = True )

            
            # append the fit result
            fitResultList.append(fitRes)

            # report to terminal
            logger.info(f"f{fitRes.report()}")

    # sort the list, such that the best fit comes first
    fitResultList.sort(key=lambda x: x.AIC())

    
    # ##########################################################
    # Plotting and Reporting
    # ##########################################################

    # dictionary of fit results with
    # "bst": { "A0":array,"A1":array,..., "E0":array,"E1":array,...} # if bootstrapped fit
    # "est": { "A0":float,"A1":float,..., "E0":float,"E1":float,...} 
    # "err": { "A0":float,"A1":float,..., "E0":float,"E1":float,...} 
    bestParams = fitResultList[0].bestParameter()

    msg = "Best Fit Result:\n"
    for key in bestParams["est"].keys():
        msg+=f"\t\t\t* {key}: {gv.gvar(bestParams['est'][key], bestParams['err'][key])}\n"
    logger.info(msg)
    
    # We provide a set of basic plotting routines that also return the plt.Figure, and axis
    # this way we can modify them after the basic plots have been done. 

    # #####################
    # 1. Plot the raw correlator data
    # #####################
    fig, ax = plotting.plotCorrelator(
        C = gv.gvar(Cest,Cerr),
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
    
    # simply save the figure into the report folder
    fig.savefig( ReportFolder/"correlator.pdf" )

    # #####################
    # 2. Plot the fit result
    # #####################
    
    # First plot the raw data once again to compare the best fit
    fig, ax = plotting.plotCorrelator(
        C = gv.gvar(Cest,Cerr),
        abscissa = abscissa,
        color = plotting.style.MAIN_COLORS['primary'],
        connectDots = False,
        label = "Sampled Data"
    )
    
    # Now finally plot the best 5 fits. 
    # For a comprehensive report you might want to fit all, but not in the same file. 
    # The legend can become quite cluttered for many fits
    for fitID,fit in enumerate(fitResultList[:5]):
        # We can add a textbox below the plot for the best fit
        # The best fit is the first element of the list as by our sorting above
        putFitReport = fitID == 0

        fig, ax = plotting.plotFit(
            fit = fit,
            figAxTuple = (fig,ax),
            plotData = False,
            label = rf"{fit.short()}",
            putFitReport = putFitReport,
            # the best fit should be plotted over all others
            zorder = 5 - fitID
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

