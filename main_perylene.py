# We offload all numerical work to numpy
import numpy as np

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

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

import itertools 

# Some plotting functions are defined in
import plotting

# Fitting is done using the fitter interface defined here
import fitter 

import fitModels

import lib_preprocessData as prep 

import lib_perylene as perylene

import multiprocessing


def fit(CLIargs,Cbst,Cest,Cerr,Ccov,abscissa,delta,irrep,irrepID,CacheFolder,ReportFolder,ensemblePath):

    tmin = np.argmin( np.abs(Cest) )

    decayingCorr = False
    if tmin > CLIargs.Nt // 2:
        decayingCorr = True

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
    maxStates = CLIargs.maxNumStateFits
    
    # iterate over possible states
    for ns in range(1,maxStates+1):
        # Create the fitter for an ns-state fit:
        # 1. Define a raw fitter
        fit = fitter.Fitter("ExponentialFitter")

        # 2. Add a ns-state fit model. We want the energies in lattice units thus we add
        #    the lattice spacing delta (default = 1 if not given) into this. 
        #    Alternatively, the abscissa could be multiplied by delta in the first place.
        fit = fit.setFitModel(
            perylene.DoubleExponentialsModel(
                Nstates_D=ns if decayingCorr else 1,
                Nstates_I=1 if decayingCorr else ns,
                Nt=CLIargs.Nt,
                beta=CLIargs.beta
            )
        ) 

        # 3.1 Add a prior to the fit model
        #fit = fit.setPrior(
        #    fitModels.DoubleExponentialsFlatPrior(Nstates=ns)
        #)
        
        fit = fit.setPrior(
            perylene.DoubleExponentialsDataBasedPrior(
                Nstates_D=ns if decayingCorr else 1,
                Nstates_I=1 if decayingCorr else ns,
                irrepID   = irrepID,
                CLIargs   = CLIargs,
                signMu    = 1 if decayingCorr else -1,
                modelAverage = fitter.modelAverage(fitResultList) if len(fitResultList) > 0 else None
            )
        )
        
        # 3.2 or add start parameter to the fit model
        #fit = fit.setP0(fitModels.SimpleSumOfExponentialsP0(Nstates=ns))

        # 4. Don't do a correlated fit (there are no correlations in the testData) 
        #    default = True -> This requires the covariance in ordinate_err
        if CLIargs.doCorrelatedFits:
            fit = fit.setCorrelated(True)

        # 5. Turn on bootstrapped fits. This requires the bootstrap samples in ordinate
        #    default = False
        if CLIargs.doBootstrapFits:
            fit = fit.setBootstrapped(CLIargs.Nbst)

        # taking all possible intervals to the left given at least enough points for the parameters
        Ls = int( 0.75*( tmin - 1 ) )
        Le = int( 0.75*( CLIargs.Nt - tmin ) )

        start_times = np.arange(1, tmin - Ls)
        end_times = np.arange( tmin+Le, CLIargs.Nt)

        # possibly reduce number of start times
        if CLIargs.maxNumIntervals is not None:
            length_s = start_times[-1] - start_times[0]
            if length_s > CLIargs.maxNumIntervals:
                skip = length_s // CLIargs.maxNumIntervals
                start_times = start_times[::skip]

            length_e = end_times[-1] - end_times[0]
            if length_e > CLIargs.maxNumIntervals:
                skip = length_e // CLIargs.maxNumIntervals
                end_times = end_times[::skip]

        # ensure that we don't loop over empty slices
        if len(start_times) == 0: start_times = [1]
        if len(end_times) == 0: end_times = [CLIargs.Nt-1]

        for ts,te in itertools.product(start_times,end_times):
            # perform the fit
            # We could also provide Covariance instead of standard deviation 
            # to utilize a chi^2 with correlations
            # Notice, don't provide bootstrap samples if no bootstrap fits are being done
            # otherwise a runtime error will be thrown due to no matching axes. 
            if not fit.bootstrapFlag:
                if fit.correlatedFit:
                    # Using Covariance
                    fitRes = fit( abscissa[ts:te], Cest[ts:te], Ccov[ts:te,ts:te], createCopy = True )
                else:
                    # Using standard deviation
                    fitRes = fit( abscissa[ts:te], Cest[ts:te], Cerr[ts:te], createCopy = True )

            # If bootstrapped fits are desired we need to provide a set of bootstrap samples
            # for the central values.
            # If a single standard devation (or covariance) is provided it will be used
            # for all bootstrap samples (key: frozen covariance)
            # this is typically a good approximation especially helps to stabilize 
            # correlated fits
            else:
                if fit.correlatedFit:
                    # Using Covariance
                    fitRes = fit( abscissa[ts:te], Cbst[:,ts:te], Ccov[ts:te,ts:te], createCopy = True )
                else:
                    # Using standard deviation
                    fitRes = fit( abscissa[ts:te], Cbst[:,ts:te], Cerr[ts:te], createCopy = True )

            # If we provide a standard deviation (or covariance) for each bootstrap sample
            # it will be used as provided.
            # Notice, a bootstrapped fit will also fit to the central values:
            # - fitRes.fitResult
            # - fitRes.fitResult_bst
            # will be filled. Typically, you want to use those as central values and determine
            # confidence from the bootstrap.
                    #fitRes = fit( abscissa[ts:te], Cbst[:,ts:te], Cbrr[:,ts:te], createCopy = True )

            # We serialize the result to disc
            fitRes.serialize(CacheFolder/ensemblePath/f"fit_Nstate{ns:g}_ts{ts:g}_te{te:g}.h5", overwrite=True)

            # we can later read it back in with 
            #fitRes = fitter.Fitter.deserialize(CacheFolder/f"fit_Nstate{ns:g}_ts{ts:g}_te{te:g}.h5")
            
            # append the fit result
            fitResultList.append(fitRes)

            # report to terminal
            print(f"{irrep[1:-1]}({irrepID}) = {fitRes.report()}")
        # end for ts,te
    # end for ns
    # sort the list, such that the best fit comes first
    fitResultList.sort(key=lambda x: x.AIC())

    # ##########################################################
    # Plotting and Reporting
    # ##########################################################

    # dictionary of fit results with
    # "bst": { "A0":array,"A1":array,..., "E0":array,"E1":array,...} # if bootstrapped fit
    # "est": { "A0":float,"A1":float,..., "E0":float,"E1":float,...} 
    # "err": { "A0":float,"A1":float,..., "E0":float,"E1":float,...} 
    bestFit = fitResultList[0]
    bestParams = bestFit.bestParameter()

    msg = f"\nBest Fit Result {irrep[1:-1]}({irrepID}): AIC={fitResultList[0].AIC():g}\n"
    for key in bestParams["est"].keys():
        # log(E0) -> E0 and so on
        if 'log' in key: key = key[4:-1]
        msg+=f"\t\t\t{key}: {gv.gvar(bestParams['est'][key], bestParams['err'][key])}\n"
    print(msg)

    # We provide a set of basic plotting routines that also return the plt.Figure, and axis
    # this way we can modify them after the basic plots have been done. 

    colors = list(plotting.style.COLORS.values())
    colorIndex = lambda k: (k * 7) % len(colors)

    # #####################
    # 1. Plot the raw correlator data
    # #####################
    fig, ax = plotting.plotCorrelator(
        C = gv.gvar(Cest,Cerr),
        abscissa = abscissa,
        color = colors[colorIndex(irrepID)],#plotting.style.MAIN_COLORS['primary'],
        connectDots = True
    )
    
    # simply save the figure into the report folder
    fig.savefig( ReportFolder/ensemblePath/f"correlator.pdf" )
    plt.close(fig)

    # #####################
    # 2. Plot the fit result
    # #####################
    
    # First plot the raw data once again to compare the best fit
    fig, ax = plotting.plotCorrelator(
        C = gv.gvar(Cest,Cerr),
        abscissa = abscissa,
        color = colors[colorIndex(irrepID)],#plotting.style.MAIN_COLORS['primary'],
        connectDots = False
    )

    # Now finally plot the best 5 fits. 
    # For a comprehensive report you might want to fit all, but not in the same file. 
    # The legend can become quite cluttered for many fits
    numPlottedFits = np.minimum( len(fitResultList), 5)
    for fitID,fit in enumerate(fitResultList[:numPlottedFits]):
        fig, ax = plotting.plotFit(
            fit = fit,
            figAxTuple = (fig,ax),
            plotData = False,
            label = rf"$({fit.model.Nstates_D},{fit.model.Nstates_I})$ States [{fit.fitResult.x[0]},{fit.fitResult.x[-1]}]: AIC={fit.AIC():g}",
            putFitReport = False,
            color = list(plotting.style.COLORS.values())[ (fitID * 4) ],
            # the best fit should be plotted over all others
            zorder = 5 - fitID
        )

    leg = ax.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)

    fig.savefig( ReportFolder/ensemblePath/f"fits_cut.pdf",bbox_inches="tight" )
    plt.close(fig)

    # First plot the raw data once again to compare the best fit
    fig, ax = plotting.plotCorrelator(
        C = gv.gvar(Cest,Cerr),
        abscissa = abscissa,
        color = colors[colorIndex(irrepID)],#plotting.style.MAIN_COLORS['primary'],
        connectDots = False
    )
    
    # Now finally plot the best 5 fits. 
    # For a comprehensive report you might want to fit all, but not in the same file. 
    # The legend can become quite cluttered for many fits
    numPlottedFits = np.minimum( len(fitResultList), 5)
    for fitID,fit in enumerate(fitResultList[:numPlottedFits]):
        # We can add a textbox below the plot for the best fit
        # The best fit is the first element of the list as by our sorting above
        putFitReport = (fitID == 0)

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

    fig.savefig( ReportFolder/ensemblePath/f"fits.pdf",bbox_inches="tight" )
    plt.close(fig)

    # #####################
    # 3. Plot a Param per AIC
    # #####################
    E_nonInt = np.abs(perylene.quantum_numbers[irrepID,2] + CLIargs.mu)

    fig, ax = plotting.plotParameterPerAIC('E0', fitResultList, colorModelAvg=colors[colorIndex(irrepID)])
    ax.set_ylabel(r"$E_0^L$")
    
    if decayingCorr:
        
        ax.axhline(
            y = E_nonInt,
            xmin = 0, xmax = 1,
            label = f'Non-Interacting: {E_nonInt: .2f}',
            color = plotting.style.MAIN_COLORS['neutral'],
            ls = '--'
        )
        ax.legend()

    
    fig.savefig( ReportFolder/ensemblePath/f"E0PerAIC.pdf",bbox_inches="tight" )
    plt.close(fig)

    fig, ax = plotting.plotParameterPerAIC('F0', fitResultList, colorModelAvg=colors[colorIndex(irrepID)])
    ax.set_ylabel(r"$E_0^R$")
    
    if not decayingCorr:
        ax.axhline(
            y = E_nonInt,
            xmin = 0, xmax = 1,
            label = f'Non-Interacting: {E_nonInt: .2f}',
            color = plotting.style.MAIN_COLORS['neutral'],
            ls = '--'
        )
        ax.legend()

    fig.savefig( ReportFolder/ensemblePath/f"F0PerAIC.pdf", bbox_inches="tight" )
    plt.close(fig)

    fig, ax = plotting.plotParameterPerAIC('A0', fitResultList, colorModelAvg=colors[colorIndex(irrepID)])
    ax.set_ylabel(r"$A_0$")
    fig.savefig( ReportFolder/ensemblePath/f"A0PerAIC.pdf",bbox_inches="tight" )
    plt.close(fig)

    fig, ax = plotting.plotParameterPerAIC('B0', fitResultList, colorModelAvg=colors[colorIndex(irrepID)])
    ax.set_ylabel(r"$B_0$")
    fig.savefig( ReportFolder/ensemblePath/f"B0PerAIC.pdf",bbox_inches="tight" )
    plt.close(fig)


    for n in range(1,maxStates):

        # Remove the 1 state fits for the following plots as they don't have the parameters
        fig, ax = plotting.plotParameterPerAIC(
            f'ΔE{n}', 
            [fit for fit in fitResultList if fit.model.Nstates_D > n], 
            colorModelAvg=colors[colorIndex(irrepID)]
        )
        ax.set_ylabel(rf"$ΔE_{{{n}}}^L$")
        fig.savefig( ReportFolder/ensemblePath/f"ΔE{n}PerAIC.pdf",bbox_inches="tight" )
        plt.close(fig)

        fig, ax = plotting.plotParameterPerAIC(
            f'ΔF{n}', 
            [fit for fit in fitResultList if fit.model.Nstates_I > n], 
            colorModelAvg=colors[colorIndex(irrepID)]
        )
        ax.set_ylabel(rf"$ΔE_{{{n}}}^R$")
        fig.savefig( ReportFolder/ensemblePath/f"ΔF{n}PerAIC.pdf",bbox_inches="tight" )
        plt.close(fig)

        fig, ax = plotting.plotParameterPerAIC(
            f'A{n}', 
            [fit for fit in fitResultList if fit.model.Nstates_D > n], 
            colorModelAvg=colors[colorIndex(irrepID)]
        )
        ax.set_ylabel(rf"$A_{{{n}}}$")
        fig.savefig( ReportFolder/ensemblePath/f"A{n}PerAIC.pdf",bbox_inches="tight" )
        plt.close(fig)

        fig, ax = plotting.plotParameterPerAIC(
            f'B{n}', 
            [fit for fit in fitResultList if fit.model.Nstates_I > n], 
            colorModelAvg=colors[colorIndex(irrepID)]
        )
        ax.set_ylabel(rf"$B_{{{n}}}$")
        fig.savefig( ReportFolder/ensemblePath/f"B{n}PerAIC.pdf",bbox_inches="tight" )
        plt.close(fig)

    del fitResultList

    return 0
# end def fit

def main(CLIargs: argparse.Namespace) -> None:
    r"""
        Main entry to this code. 
        Arguments: 
            - CLIargs: argparse.Namespace, command-line arguments as parsed by argparse. Required constituents:
                1. cacheFolder: str, a folder to store analysis results (as hdf5 format).
                2. reportFolder: str, a folder to store plots and text files summarizing the fit results.

    """

    # ##########################################################
    # Setting up the interface
    # ##########################################################

    # print the provided CLI arguments
    print(CLIargs)

    # make an ensemble string to put results per ensamble
    ensemblePath = Path(f"Nt{CLIargs.Nt:g}/beta{CLIargs.beta:g}/U{CLIargs.U:g}/mu{CLIargs.mu:g}")

    # define the Cache folder in which we cache the fit results
    CacheFolder = Path(CLIargs.cacheFolder)

    # define the Cache folder in which we cache the fit results
    ReportFolder = Path(CLIargs.reportFolder)

    # Read in the data and preprocess it
    DataFolder = Path( CLIargs.dataFolder )


    # ##########################################################
    # Data Read and Preprocess
    # ##########################################################

    # we have a mixture of isle and NSL data
    dataFN = DataFolder/f"HMC/HMC_Nt{CLIargs.Nt:g}_beta{CLIargs.beta:g}_U{CLIargs.U:g}_mu{CLIargs.mu:g}.h5"
    if dataFN.exists():
        isIsleData = True
    else: 
        dataFN = DataFolder/f"HMC_Nt{CLIargs.Nt:g}_beta{CLIargs.beta:g}_U{CLIargs.U:g}_mu{CLIargs.mu:g}.h5"
        if dataFN.exists():
            isIsleData = False
        else:
            raise RuntimeError("Data not found. Exiting...")

    # We can use the preprocess data class to get a handle on the correlators from NSL
    # Access via
    # corrData.C_sp_bst, *_est, *_err: correlator data
    corrData = prep.Data(
        h5path = dataFN, 
        Nt = CLIargs.Nt, beta = CLIargs.beta, U = CLIargs.U, mu = CLIargs.mu, 
        Nconf = CLIargs.Nconf, Nbst = CLIargs.Nbst, 
        isIsleData = isIsleData
    )
    
    delta = corrData.delta
    abscissa = np.arange(CLIargs.Nt)

    # generate list of irreps to fit
    if len(CLIargs.includeIrrepIDs) > 0:
        irreps = [ (irrepID, perylene.D2irreps[irrepID]) for irrepID in CLIargs.includeIrrepIDs ]
        total = len(irreps)
    else:
        irreps = list(enumerate(perylene.D2irreps))
        total = len(perylene.D2irreps)

    # make sure that the folders exists
    for irrepID,irrep in irreps: 
        (CacheFolder/ensemblePath/f"irrepID{irrepID}").mkdir(parents=True, exist_ok=True)
        (ReportFolder/ensemblePath/f"irrepID{irrepID}").mkdir(parents=True, exist_ok=True)

    # loop over irreps 
    if CLIargs.numProc is None:
        for irrepID,irrep in tqdm(irreps, total = total):
            fit(
                CLIargs,
                corrData.C_sp_bst[:,:,irrepID],
                corrData.C_sp_est[:,irrepID],
                corrData.C_sp_err[:,irrepID],
                corrData.C_sp_cov[:,:,irrepID],
                abscissa,delta,
                irrep,irrepID,
                CacheFolder,ReportFolder,ensemblePath/f"irrepID{irrepID}"
            )

        return
    # if numProc is given parallel process!
    with multiprocessing.get_context('spawn').Pool(processes=CLIargs.numProc) as pool:
        res = pool.starmap_async(
            fit,
            [
                (
                    CLIargs,
                    corrData.C_sp_bst[:,:,irrepID],
                    corrData.C_sp_est[:,irrepID],
                    corrData.C_sp_err[:,irrepID],
                    corrData.C_sp_cov[:,:,irrepID],
                    abscissa,delta,
                    irrep,irrepID,
                    CacheFolder,ReportFolder,ensemblePath/f"irrepID{irrepID}" 
                )
                for irrepID,irrep in irreps
            ]
        )
        
        # wait until all are done
        res.wait()
        
# end def main

if __name__ == "__main__":

    # fix the random seed for reproducibility
    np.random.seed(21)

    # Define the command-line arguments
    CLIParser = argparse.ArgumentParser(
        prog='Data Analyser'
    )

    # Ensemble parameters
    CLIParser.add_argument('-Nt', type = int, required = True )
    CLIParser.add_argument('-beta', type = float, required = True )
    CLIParser.add_argument('-U', type = float, required = False, default = 2 )
    CLIParser.add_argument('-mu', type = float, required = True )
    # for convenience
    CLIParser.add_argument('-Nx', type = int, required = False, default=20 )

    # Markov Chain parameters
    CLIParser.add_argument('-Nconf', type = int, required = True )

    # Analysis specification
    CLIParser.add_argument('-Nbst', type = int, required = True)
    CLIParser.add_argument('-maxNumStateFits', type = int, required = True)
    CLIParser.add_argument('-maxNumIntervals', type = int, required = False, default = None)
    CLIParser.add_argument('--doBootstrapFits', action='store_true')
    CLIParser.add_argument('--doCorrelatedFits', action='store_true')
    CLIParser.add_argument('-includeIrrepIDs', nargs='+', type = int, required = False, default = [])

    # Serialization 
    CLIParser.add_argument('-cacheFolder', type = str, required = False, default = "Cache")
    CLIParser.add_argument('-reportFolder', type = str, required = False, default = "Report")
    CLIParser.add_argument('-dataFolder', type = str, required = True)
    CLIParser.add_argument('-numProc', type = int, required = False, default = None)

    # Parse the CLI arguments
    CLIargs = CLIParser.parse_args()
    
    main(CLIargs)

    print("Done.")
# end __name__ == __main__
