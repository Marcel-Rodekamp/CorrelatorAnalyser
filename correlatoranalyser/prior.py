import numpy as np 
import h5py as h5
from typing import Self

class Prior:
    mean: float 
    sdev: float 

    dist: str 

    def __init__(self, mean:float, sdev:float, dist = 'normal'):
        self.mean:float = mean
        self.sdev:float = sdev

        if dist == 'normal':
            self._eval = self.normal
            self.dist = dist
        elif dist == 'log' or dist == 'log-normal':
            self._eval = self.log_normal
            self.dist = 'log-normal'
        else:
            raise RuntimeError(
                f"dist must be one of:\n"
               +f"    - 'normal': normal distribution\n"
               +f"    - 'log-normal', 'log': log-normal distribution\n"
            )

    def gvar(self):
        import gvar as gv 

        if self.dist == "normal":
            return gv.gvar( self.mean, self.sdev )
        else:
            return gv.log(gv.gvar( self.mean, self.sdev ))

        
    def normal(self, theta:float) -> float:
        return ((theta - self.mean)/self.sdev)**2

    def log_normal(self,theta:float) -> float:
        return ((np.log(theta) - self.mean)/self.sdev)**2

    def __call__(self, theta:float) -> float:
        return self._eval(theta) 

    @staticmethod
    def import_from_lsqfit(prior:dict) -> dict[str,Self]:
        import gvar as gv

        out:dict[str,Self] = {}
        for key, value in prior.items():
            if 'log' in key:
                key_ = key[3:-1]
                out[key_] = Prior( mean = gv.mean(value), sdev=gv.sdev(value), dist='log-normal' ) 
            else:
                out[key] = Prior( mean = gv.mean(value), sdev=gv.sdev(value), dist='normal' ) 

        return out             

    def serialize(self, h5f: h5.Group, node:str|None = None) -> None:
        if node is None:
            grp = h5f
        else:
            grp = h5f[node]

        grp.create_dataset("mean", data=self.mean)
        grp.create_dataset("sdev", data=self.sdev)
        grp.create_dataset("dist", data=self.dist)
    
    @staticmethod
    def deserialize(h5f: h5.Group, node:str|None = None) -> Self:
        if node is None:
            grp = h5f
        else:
            grp = h5f[node]

        new:Prior = Prior(
            mean=grp["mean"],
            sdev=grp["sdev"],
            dist=grp["dist"],
        )

        return new

    def __repr__(self):
        if self.dist == "normal":
            return f"N[μ={self.mean}, σ={self.sdev}]"
        else:
            return f"logN[μ={self.mean}, σ={self.sdev}]"


        