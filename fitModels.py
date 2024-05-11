# We offload all numerical work to numpy
import numpy as np

# for type hints import Self 
from typing import Self

# for the base class definition we import abstract utils
from abc import abstractmethod

# gvar - Gaussian Random Variables - is a package that defines random variables and is heavily used
# in lsqfit. 
import gvar as gv

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
    def __call__(self):
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
    def __init__(self, Nstates):
        self.Nstates = Nstates

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
            out += p[f"A{n}"] * np.exp( - tau * p[f"E{n}"] )

        return out

    def __repr__(self):
        return f"ExpModel[Nstates={self.Nstates}]"

class SimpleSumOfExponentialsFlatPrior(PriorBase): 
    def __init__(self, Nstates, repr = "Prior"):
        self.Nstates = Nstates
        self.repr = repr

    def __call__(self) -> gv.BufferDict:
        prior = gv.BufferDict()
    
        for n in range(self.Nstates):
            prior[f"E{n}"] = gv.gvar(0, 100)
            prior[f"A{n}"] = gv.gvar(0, 100)
    
        return prior

    def __repr__(self):
        return self.repr
