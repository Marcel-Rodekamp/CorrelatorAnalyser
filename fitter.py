# We offload all numerical work to numpy
import numpy as np

# we might want to allow deepcopies of fit models
from copy import deepcopy

# Fitting is done using lsqfit, as it provides many features that we utilize. 
# A lot of this code implements some wrapper around this
import lsqfit

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

# File names are handled using pathlib
from pathlib import Path

# import h5py and pickle to (de)serialize a Fitter
import pickle

import h5py as h5

# for type hints import Self 
from typing import Self

# Import the base classes for type annotation
from fitModels import FitModelBase, PriorBase

# Bootstrapping can sometimes be slow we use multiprocessing to parralelize the fits over
# the different bst samples
# ToDo We need to figure out how to pickle the gvars with multiprocessing
#from multiprocessing import Pool, cpu_count


def AIC(fit: lsqfit.nonlinear_fit) -> float:
    r"""
        Compute the Akaike information criterion for a fit result
        based on the chi^2 obtained from lsqfit.
        The form can be found in 
            https://arxiv.org/abs/2305.19417
            https://arxiv.org/abs/2208.14983
            https://arxiv.org/abs/2008.01069
        equation 3 in the first:
            AIC^{perf} = -2ln L^* + 2k - 2d_K
        Here we compare
            1. -2*ln(L^*) = chi^2 
            2. k = number of parameters
            3. d_K = number of points

        If priored fit, this includes the prior. For a prior less version see AICp
    """

    return fit.chi2 + 2*len(fit.p) - 2*len(fit.x)

def AICp(fit: lsqfit.nonlinear_fit) -> float:
    chip = np.sum([
        (fit.prior[key].mean-fit.p[key].mean)**2/fit.prior[key].sdev**2 
            for key in fit.prior.keys()
    ])

    return AIC(fit) - chip

class Fitter:
    r"""
        ToDo
    """
    # Fit model: callable(tau,p) where the parameters p can be fitted
    model: FitModelBase = None
    # Prior: callable(nbst = None) a prior to the parameters
    prior: PriorBase = None
    p0:    callable = None

    correlatedFit: bool = False

    bootstrapFlag: bool = False
    Nbst: int = None
    fitResult_bst: np.ndarray = None

    # Results
    fitResult: lsqfit.nonlinear_fit = None

    def __init__(self, repr:str = "Fit", useMultiprocessing: bool = True, Nplot = 100, maxiter:int = 1000) -> Self:
        r"""
            Setup the basic fitter
        """
        self.maxiter = maxiter
        self.repr = repr
        self.Nplot = Nplot
        self.useMultiprocessing = useMultiprocessing

    def setFitModel(self, model: FitModelBase ) -> Self:
        self.model = model
        return self

    def setPrior(self, prior: PriorBase ) -> Self:
        if self.p0 is not None:
            raise RuntimeError( f"Attempting to set prior where start parameter is already set: {self.p0()}" )

        self.prior = prior
        return self

    def setP0(self, p0: PriorBase ) -> Self:
        if self.prior is not None:
            raise RuntimeError( f"Attempting to set start parameter where prior is already set: {self.prior()}" )
        self.p0 = p0
        return self

    def setCorrelated(self, flag: bool) -> Self:
        r"""
            Turns correlated fits on. 
            When this method is called each fitted uses a correlated $\chi^2$. 
            The provided Cerr in the fits call is then interpreted as a covariance matrix.
            If one diagonal (standard deviations) are provided it will be understood as a
            diagonal covariance matrix (the square is automatically computed).

            Caution, this tends to be very unstable when combined with bootstrap!
            When the covariance is not invertible consider adding an svd cut. 

            default = False
        """
        self.correlatedFit = flag
        return self

    def setBootstrapped(self, Nbst:int, flag: bool = True) -> Self:
        r"""
            Turn on fitting over bootstraps. 
            When this method is called a fit on bootstrap samples is performed which must 
            be represented in the data provided to the fit call.

            Notice, doing correlated fits under the bootstrap tends to be unstable. It should
            be save to do uncorrelated fits here as the bootstrap maintains necessary correlations

            It is encourage to develop priors/start parameters without fitting over bootstraps
            as the required compute time becomes larger! If it is turned off uncertainties
            are propagated using Gaussian error-propagation taking correlations into account
            where possible.

            default = False
        """
        self.bootstrapFlag = flag
        self.Nbst = Nbst
        self.fitResult_bst = np.empty( self.Nbst, dtype=object )

        return self
    
    def AIC(self, getBst = False):
        if getBst:
            aic = np.zeros(self.Nbst)
            for nbst in range(self.Nbst):
                aic[nbst] = AIC(self.fitResult_bst[nbst])
            return aic
        else:
            return AIC(self.fitResult)

    def AICp(self, getBst = False):
        if getBst:
            aicp = np.zeros(self.Nbst)
            for nbst in range(self.Nbst):
                aicp[nbst] = AICp(self.fitResult_bst[nbst])
            return aicp
        else:
            return AICp(self.fitResult)

    def Q(self,getBst = False):
        if getBst:
            Qs = np.zeros(self.Nbst)
            for nbst in range(self.Nbst):
                Qs[nbst] = self.fitResult_bst[nbst].Q
            return Qs
        else:
            return self.fitResult.Q

    def logGBF(self,getBst = False):
        if getBst:
            logGBF = np.zeros(self.Nbst)
            for nbst in range(self.Nbst):
                logGBF[nbst] = self.fitResult_bst[nbst].logGBF
            return logGBF
        else:
            return self.fitResult.logGBF

    def chi2(self, getBst = False):
        if getBst:
            chi2s = np.zeros(self.Nbst)
            for nbst in range(self.Nbst):
                chi2s[nbst] = self.fitResult_bst[nbst].chi2/self.fitResult_bst[nbst].dof
            return logGBF
        else:
            return self.fitResult.chi2/self.fitResult.dof 

    def bestParameter(self):
        out = None

        if self.bootstrapFlag:
            # collect the bst fit results
            p_bst = {}
            p_est = {}
            p_err = {}

            keys = list(self.fitResult.p.keys())

            for key in keys:
                if "log" in key:
                    keyRed = key[4:-1]
                    p_est[keyRed] = gv.mean(gv.exp(self.fitResult.p[key]))
                    p_bst[keyRed] = np.zeros(self.Nbst)
                    for nbst in range(self.Nbst):
                        p_bst[keyRed][nbst] = gv.mean(gv.exp(self.fitResult_bst[nbst].p[key]))
                    p_err[keyRed] = np.std(p_bst[keyRed], axis=0)
                elif key[4:-1] not in p_est.keys():
                    p_est[key] = gv.mean(self.fitResult.p[key])
                    p_bst[key] = np.zeros(self.Nbst)
                    for nbst in range(self.Nbst):
                        p_bst[key][nbst] = gv.mean(self.fitResult_bst[nbst].p[key])
                    p_err[key] = np.std(p_bst[key], axis=0)
                else:
                    # we don't want to read the param which has been computed from the log 
                    pass

            out = { 
                "bst": p_bst,
                "est": p_est,
                "err": p_err
            }
        else:
            # lsqfit does weird things with the std when using log priors. We translate
            # them ourself here 
            # this is not required for the bootstrap as the mean value is not affected,
            # and the std is computed over the bst samples.
            p = self.fitResult.p
            p_res = {}

            keys = list(p.keys())
            for key in keys:
                if "log" in key: 
                    p_res[key[4:-1]] = gv.exp(p[key])
                elif key[4:-1] not in p_res.keys():
                    p_res[key] = p[key]
                else:
                    # we don't want to read the param which has been computed from the log 
                    pass


            out = { 
                "est": gv.mean(p_res),
                "err": gv.sdev(p_res)
            }

        return out

    def evalModel(self, abscissa = None):
        if abscissa is None:
            abscissa = np.linspace( self.fitResult.x[0], self.fitResult.x[-1], self.Nplot )
            
        if self.bootstrapFlag:
            ordinate_bst = np.zeros((self.Nbst,len(abscissa)))
            
            for nbst in range(self.Nbst):
                ordinate_bst[nbst] = gv.mean( self.fitResult_bst[nbst].fcn(abscissa, self.fitResult_bst[nbst].p) )

            ordinate = gv.gvar(
                np.mean(ordinate_bst,axis=0),
                np.std (ordinate_bst,axis=0)
            )

        else:
            ordinate = self.fitResult.fcn(abscissa, self.fitResult.p)

        return abscissa,ordinate

    def report(self):
        return f"{self.short()}\n{self.fitResult}"

    def short(self):
        return f"{self.repr}[{{{self.model}}}] @ [{self.fitResult.x[0]},{self.fitResult.x[-1]}]: AIC={self.AIC():g}"

    def __repr__(self):
        if self.fitResult is None:
            return f"{self.repr}{{{self.model}}} [Not Executed]"
        else:
            return f"{self.repr}{{{self.model}}} [χ²/dof={self.chi2():g}, Q={self.Q():g}, AIC={self.AIC():g}, logGBF={self.logGBF():g}, converged={bool(self.fitResult.stopping_criterion)}]"

    def __call__(self, abscissa: np.ndarray, ordinate: np.ndarray, ordinate_err: np.ndarray, createCopy: bool = False) -> Self:
        r"""
            Perform the fit

            Arguments:
                - abscissa: np.array(N), mesh to evaluate the function; matching the data.
                - ordinate: np.array(([Nbst],N)), data to fit against.
                - ordinate_err: np.array(([Nbst],N)) or np.array(([Nbst],N,N)), standard deviation for each datapoint or covariance between the data points
                - createCopy: bool, flag weather the returned Fitter is a copy or reference to self.
        """
        
        # assemble the arguments according to the construction of the class
        args = {}

        # Provide fit model
        args["fcn"] = self.model

        # Provide a maximum of iteration to prevent infinitely running fits
        args["maxit"] = self.maxiter 
        
        # Some arguments will differ whether a bootstrapped fit is performed or not
        if self.bootstrapFlag:
            # check that the data is in the right shape
            if ordinate.shape[0] != self.Nbst:
                raise RuntimeError( 
                    f"Ordinates first axis does not match number of bootstrap samples: Nbst={self.Nbst}, {ordinate.shape=}"
                )
            if ordinate.shape[-1] != abscissa.shape[0]:
                raise RuntimeError( 
                    f"Ordinates last axis does not match abscissa: {ordinate.shape=}, {abscissa.shape=}"
                )

            if ordinate_err.shape[-1] != abscissa.shape[0]:
                raise RuntimeError( 
                    f"Ordinate_errs last axis does not match abscissa: {ordinate_err.shape=}, {abscissa.shape=}"
                )

            if ordinate_err.ndim >= 2:
                if ordinate_err.shape[0] == self.Nbst:
                    ordinateErrHasBst = True
                else:
                    ordinateErrHasBst = False
            else:
                ordinateErrHasBst = False

            # go through the bootstrap samples and perform the fit
            for nbst in range(self.Nbst):
                
                # A (co)varaince per bootstrap sample if ordinateErrHasBst is True
                # A (co)varaince for all bootstrap samples if ordinateErrHasBst is False 
                ordinate_gvar = gv.gvar(
                    ordinate[nbst],
                    ordinate_err[nbst] if ordinateErrHasBst else ordinate_err
                )

                # Provide the data for the fit
                args["data" if self.correlatedFit else "udata"] = (abscissa, ordinate_gvar)

                # Provide a prior or a starting value
                if self.prior is not None:
                    args["prior"] = self.prior(nbst)
                elif self.p0 is not None:
                    args["p0"] = self.p0()

                # Perform fit, the try-except block provides some more information about 
                # the fit before raising the exception.
                try:
                    self.fitResult_bst[nbst] = lsqfit.nonlinear_fit(**args)
                except Exception as e:
                    msg =  f"Fit Failed: {self}:\n"
                    for key,val in args.items():
                        msg+= f"- {key}: {val}\n"

                    msg += f"nbst = {nbst}/{self.Nbst}\n"

                    raise RuntimeError(f"{msg}\n{e}")
            # end for nbst 

            # Prepare for central value fit
            ordinate_gvar = gv.gvar(
                np.mean(ordinate,axis = 0),
                np.mean(ordinate_err,axis=0) if ordinateErrHasBst else ordinate_err
            )

            # Provide the data for the fit
            args["data" if self.correlatedFit else "udata"] = (abscissa, ordinate_gvar)

            # Provide a prior or a starting value
            if self.prior is not None:
                args["prior"] = self.prior()
            elif self.p0 is not None:
                args["p0"] = self.p0()

        # #######################################################################################################
        else: # if not bootstrapped 
             # check that the data is in the right shape
            if ordinate.shape[-1] != abscissa.shape[0]:
                raise RuntimeError( 
                    f"Ordinates last axis does not match abscissa: {ordinate.shape=}, {abscissa.shape=}"
                )

            if ordinate_err.shape[-1] != abscissa.shape[0]:
                raise RuntimeError( 
                    f"Ordinate_errs last axis does not match abscissa: {ordinate_err.shape=}, {abscissa.shape=}"
                )

            ordinate_gvar = gv.gvar(ordinate, ordinate_err)

            # Provide the data for the fit
            args["data" if self.correlatedFit else "udata"] = (abscissa, ordinate_gvar)

            # Provide a prior or a starting value
            if self.prior is not None:
                args["prior"] = self.prior()
            elif self.p0 is not None:
                args["p0"] = self.p0()
        # end else (not bootstrapped) 

        # Perform fit, the try-except block provides some more information about 
        # the fit before raising the exception.
        try:
            self.fitResult = lsqfit.nonlinear_fit(**args)
        except Exception as e:
            msg =  f"Fit Failed: {self}:\n"
            for key,val in args.items():
                msg+= f"- {key}: {val}\n"

            raise RuntimeError(f"{msg}\n{e}")

        # The work is done, we can now return
        if createCopy:
            return deepcopy(self)
        else:
            return self
    
    def serialize(self, file: Path, overwrite: bool = True) -> None:
        r"""
            Dump a Fitter to file
        """
        
        handle = 'w' if overwrite else 'r+' 

        with h5.File(file, handle) as h5f:
            for key,value in self.__dict__.items():
                if "fitResult" == key or "fitResult_bst" == key :
                    value = np.void(gv.dumps(value))
                elif "model" == key or "prior" == key or "p0" == key:
                    value = np.void(pickle.dumps(value))

                h5f.create_dataset(
                    key, data = value 
                )

    @staticmethod
    def deserialize(file: Path) -> Self:
        r"""
            Read a Fitter from file
        """
        fitter = Fitter()

        with h5.File(file, 'r') as h5f:

            for key in h5f.keys():
                if "fitResult" == key or "fitResult_bst" == key :
                    value = gv.loads(h5f[key][()])
                elif "model" == key or "prior" == key or "p0" == key:
                    value = pickle.loads(h5f[key][()])
                elif "repr" == key:
                    value = h5f[key][()].decode('utf-8')
                else: 
                    value = h5f[key][()]

                fitter.__dict__[key] = value

        return fitter
