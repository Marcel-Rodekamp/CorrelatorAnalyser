# We offload all numerical work to numpy
import numpy as np

# for type hints import Self 
from typing import Self

# for the base class definition we import abstract utils
from abc import abstractmethod

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

from helper import meff

#########################################################################################
#########################################################################################
# Each section defines a model and a prior associated to it. Some of these might be reusable
# for various purpuses but in general I think most of the times one has to redefine the 
# model and especially the corresponding prior
#########################################################################################
#########################################################################################


#########################################################################################
# The Base Class
#---------------
# To ensure the layout is available for the Fitter please follow the construction of the 
# fit model classes from these base classes.
#########################################################################################

class FitModelBase(object):
    @abstractmethod
    def __call__(self, tau: np.ndarray, p:dict):
        raise NotImplementedError("Trying to call abstractmethod __call__ of FitModelBase")

    @abstractmethod
    def __repr__(self):
        return "FitModelBase"

class PriorBase(object):
    @abstractmethod
    def __call__(self, nbst = None):
        raise NotImplementedError("Trying to call abstractmethod __call__ of PriorBase")

    @abstractmethod
    def __repr__(self):
        return "PriorBase"

#########################################################################################
# The Simple Exponential Model
#-----------------------------
# A simple sum of exponentials with factors.
#
# sum_n A_n e^{- t E_n}
#
#########################################################################################

class SimpleSumOfExponentialsModel(FitModelBase): 
    def __init__(self, Nstates, delta = 1):
        self.Nstates = Nstates
        self.delta = delta

    def __call__(self,tau:np.ndarray, p:dict) -> np.ndarray:
        r"""
            Arguments:
                tau: np.array(Nt), abscissa in lattice units, e.g. `np.arange(Nt)` or `np.linspace(0,T,Nt)/delta`
                p: dict, dictionary of parameters. Keys are assumed to be 
                    - A0, A1, ... (overlap factors)
                    - E0, E1, ... (energies in lattice units)
            
            Evaluating the function

            \f[
                C(t) = \sum_{n=0}^{Nstates-1} A_n e^{- E_n * t}
            \f]
        """

        out = np.zeros_like(tau, dtype = object)

        for n in range(self.Nstates):
            out += p[f"A{n}"] * np.exp( - tau * p[f"E{n}"] * self.delta )

        return out

    def __repr__(self):
        return f"ExpModel[Nstates={self.Nstates}]"

class SimpleSumOfExponentialsFlatPrior(PriorBase): 
    def __init__(self, Nstates, repr = "Prior"):
        self.Nstates = Nstates
        self.repr = repr

    def __call__(self, nbst = None) -> gv.BufferDict:
        prior = gv.BufferDict()
    
        for n in range(self.Nstates):
            prior[f"E{n}"] = gv.gvar(0, 100)
            prior[f"A{n}"] = gv.gvar(0, 100)

        return prior

    def __repr__(self):
        return self.repr

class SimpleSumOfExponentialsP0(PriorBase): 
    def __init__(self, Nstates, repr = "P0"):
        self.Nstates = Nstates
        self.repr = repr

    def __call__(self) -> gv.BufferDict:
        prior = gv.BufferDict()
    
        for n in range(self.Nstates):
            prior[f"E{n}"] = 0
            prior[f"A{n}"] = 1
    
        return prior

    def __repr__(self):
        return self.repr

#########################################################################################
# The Sum of forward and backward running exponentials with ordered size
#-----------------------------
# A simple sum of exponentials with factors.
#
#   sum_n A_n e^{- t E_n} + B_n e^{ t F_n}
# = A_0 e^{- t E_0} + B_0 e^{ t F_0} + sum_n>0 A_n e^{- t ΔE_n} + B_n e^{ t ΔF_n}
#
#########################################################################################

class DoubleExponentialsModel(FitModelBase): 
    def __init__(self, Nstates:int, delta:float = 1):
        self.Nstates = Nstates
        self.delta = delta

    def __call__(self,tau:np.ndarray, p:dict) -> np.ndarray:
        r"""
            Arguments:
                tau: np.array(Nt), abscissa in lattice units, e.g. `np.arange(Nt)` or `np.linspace(0,T,Nt)/delta`
                p: dict, dictionary of parameters. Keys are assumed to be 
                    - A0, A1, ... (overlap factors)
                    - E0, E1, ... (energies in lattice units)
            
            Evaluating the function 

            \f[
                C(t) &= \sum_n A_n e^{- t E_n} + B_n e^{ t F_n} \\
                     &= A_0 e^{- t E_0} + B_0 e^{ t F_0} 
                      + \sum_{n>0} A_n e^{- t ΔE_n} + B_n e^{ t ΔF_n}
            \f]
        """
        out = np.zeros_like(tau, dtype = object)
        
        E = p["E0"]
        F = p["F0"]

        out += p["A0"] * np.exp( - tau * E * self.delta )
        out += p["B0"] * np.exp(   tau * F * self.delta )

        for n in range(1,self.Nstates):
            E += p[f"ΔE{n}"]
            F += p[f"ΔF{n}"]

            out += p[f"A{n}"] * np.exp( - tau * E * self.delta )
            out += p[f"B{n}"] * np.exp(   tau * F * self.delta )

        return out

    def __repr__(self):
        return f"DoubleExpModel[Nstates={self.Nstates}]"

class DoubleExponentialsFlatPrior(PriorBase): 
    def __init__(self, Nstates, repr = "Prior"):
        self.Nstates = Nstates
        self.repr = repr

    def __call__(self, nbst = None) -> gv.BufferDict:
        prior = gv.BufferDict()
    
        prior[f"log(E{0})"] = gv.log(gv.gvar(1, 10))
        prior[f"log(F{0})"] = gv.log(gv.gvar(1, 10))

        for n in range(1,self.Nstates):
            prior[f"log(ΔE{n})"] = gv.log(gv.gvar(1, 10))
            prior[f"log(ΔF{n})"] = gv.log(gv.gvar(1, 10))

        for n in range(self.Nstates):
            prior[f"A{n}"] = gv.gvar(0, 1)
            prior[f"B{n}"] = gv.gvar(0, 1)

        return prior

    def __repr__(self):
        return self.repr

class DoubleExponentialsDataBasedPrior(PriorBase): 
    def __init__(self, Nstates:int, Corr_est: np.ndarray = None, Corr_err: np.ndarray = None, Corr_bst: np.ndarray = None, delta:float = 1, percent:float = None, widthFactor:float = None, repr = "Prior"):
        self.Nstates = Nstates
        self.repr = repr
        self.delta = delta

        if Corr_est is None:
            if Corr_bst is None: raise RuntimeError(f"Either provide Corr_est and Corr_err or Corr_bst")
            Corr_est = np.mean(Corr_bst, axis=0)
        
        if Corr_err is None:
            if Corr_bst is None: raise RuntimeError(f"Either provide Corr_est and Corr_err or Corr_bst")
            Corr_err = np.std(Corr_bst, axis=0)

        # put the correlator into a local gvar for easy use
        if Corr_bst is None:
            Corr_gvar = gv.gvar(Corr_est,Corr_err)
        else:
            Corr_gvar = gv.dataset.avg_data(Corr_bst)

        # find the axis length
        N, = Corr_est.shape

        # compute the minimal point of the correlator:
        tmin = np.argmin(Corr_est)

        # find the midpoint of the left and right interval to compute the effective mass
        # as a prior to E0,F0
        tmidL = (tmin - 1) // 2

        tmidR = (N-1 - tmin) // 2

        # compute effective masses 
        _, meff_est,meff_err = meff( Corr_est, Corr_err, Corr_bst, delta=self.delta )

        # and place them as a prior
        if percent is None and widthFactor is not None:
            self.Eeff_prior = gv.gvar(np.abs(meff_est[tmidL]), widthFactor*meff_err[tmidL])
            self.Feff_prior = gv.gvar(np.abs(meff_est[tmidR]), widthFactor*meff_err[tmidR])
        elif widthFactor is None and percent is not None:
            self.Eeff_prior = gv.gvar(np.abs(meff_est[tmidL]), percent*meff_est[tmidL])
            self.Feff_prior = gv.gvar(np.abs(meff_est[tmidR]), percent*meff_est[tmidR])
        else: 
            raise RuntimeError("Either provide percent ({percent}) or widthFactor ({widthFactor})")

        # With these we can approximate a prior for the ground state overlap assuming
        # C(t_mid) = A0 exp(-t Eeff)
        self.Aeff_prior = Corr_gvar[tmidL] * gv.exp( tmidL * self.delta * self.Eeff_prior)
        # C(t_mid) = B0 exp( t Feff)
        self.Beff_prior = Corr_gvar[tmidR] * gv.exp(-tmidR * self.delta * self.Feff_prior)

    def __call__(self, nbst = None) -> gv.BufferDict:
        prior = gv.BufferDict()
    
        prior[f"log(E{0})"] = gv.log( self.Eeff_prior )
        prior[f"log(F{0})"] = gv.log( self.Feff_prior )
        prior[f"A{0}"] = self.Aeff_prior
        prior[f"B{0}"] = self.Beff_prior

        for n in range(1,self.Nstates):
            prior[f"log(ΔE{n})"] = gv.log(gv.gvar(1, 10))
            prior[f"log(ΔF{n})"] = gv.log(gv.gvar(1, 10))

            prior[f"A{n}"] = gv.gvar(0, 1)
            prior[f"B{n}"] = gv.gvar(0, 1)

        return prior

    def __repr__(self):
        return self.repr

