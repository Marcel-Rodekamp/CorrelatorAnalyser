# for type hints import Self 
from typing import Self

# we might want to allow deepcopies of fit models
from copy import deepcopy

# We offload all numerical work to numpy
import numpy as np

# Fitting is done using lsqfit, as it provides many features that we utilize. 
# A lot of this code implements some wrapper around this
import lsqfit

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

#
from fitModels import FitModelBase, PriorBase


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

        @1. we use the augmented chi^2 from lsqfit including the priors
        @2. we have a prior for each parameter thus each counts twice
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
    model: callable = None
    prior: callable = None

    # Results
    fitResult: lsqfit.nonlinear_fit = None

    def __init__(self, repr:str = "Fit", Nplot = 100, maxiter:int = 1000) -> Self:
        r"""
            Setup the basic fitter
        """
        self.maxiter = maxiter
        self.repr = repr
        self.Nplot = Nplot

    def setFitModel(self, model: FitModelBase ) -> Self:
        self.model = model
        return self

    def setPrior(self, prior: PriorBase ) -> Self:
        self.prior = prior
        return self
    
    def AIC(self):
        return AIC(self.fitResult)

    def AICp(self):
        return AICp(self.fitResult)

    def Q(self):
        return self.fitResult.Q

    def logGBF(self):
        return self.fitResult.logGBF

    def chi2(self):
        return self.fitResult.chi2/self.fitResult.dof 

    def __call__(self, abscissa: np.ndarray, ordinate: np.ndarray[gv.gvar], createCopy: bool = False) -> Self:
        r"""
            Perform the fit

            Arguments:
                - data: np.ndarray, data to fit against. Must match the setup provided bevore
                        this call is done.
        """
        
        self.fitResult = lsqfit.nonlinear_fit(
            data = (abscissa, ordinate),
            prior = self.prior(),
            fcn = self.model,
            maxit = self.maxiter,
        )

        if createCopy:
            return deepcopy(self)
        else:
            return self

    def bestParameter(self, asGvar = False):
        out = None

        if asGvar:
            out = { "est": self.fitResult.p }
        else:
            out = { 
                "est": gv.mean(self.fitResult.p),
                "err": gv.sdev(self.fitResult.p)
            }

        return out

    def evalModel(self, abscissa = None):
        
        if abscissa is None:
            abscissa = np.linspace( self.fitResult.x[0], self.fitResult.x[-1], self.Nplot )
            
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

