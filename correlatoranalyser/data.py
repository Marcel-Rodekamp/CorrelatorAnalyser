import numpy as np
from numpy import _NoValue # type: ignore

import h5py as h5

from typing import Union,Self,Any,Callable
Number = Union[int, float, complex, np.floating]
Real   = Union[int, float]

HANDLED_FUNCTIONS_DATA = {}
def implements(numpy_function):
    """Register an __array_function__ implementation for Data objects."""
    def decorator(func):
        HANDLED_FUNCTIONS_DATA[numpy_function] = func
        return func
    return decorator

def is_broadcastable(*arrays: np.ndarray) -> bool:
    try:
        np.nditer(arrays)
    except ValueError:
        return False
    return True

SEED:int = 88567
np.random.seed(SEED)
rng: Callable[[],np.random.Generator]  = lambda : np.random.default_rng(seed=SEED) 

class Data:
    # The resample type: 'bst': bootstraps (with replacement), 'jkn': leave-1-out jackknife
    resample_type:str
    
    # central values (arithmetic mean over resamples or provided central value estimate). May have been estimated from resamples 
    _mean: np.ndarray

    # resamples of the raw data
    _rspl:np.ndarray

    # number of resamples (axis=0 length of _rspl)
    Nresample:int

    # number of data points used to be resampled. May not be given  
    Ndata: int | None = None

    # determins the size of each bin if the data is blocked before resample
    blocksize: int | None = None

    # a string tag attached to the data set 
    tag: str | None = None

    # resamples of reweighting factors if provided
    _rwf_rspl: np.ndarray|None = None

    # for error of error analysis a inner bootstrap is executed using this number of samples
    Nbst_inner: int = 100

    # we may cache (standard) covariances / statistical error / correlation or signal to noise estimates
    # after respective methods are called first time 
    # the term standard refers to the usual construction of standard error (std(bst), sqrt(N-1)std(jkn)) 
    # and correlation / covariance similarly
    cache_field_names = ["_serr", "_cov", "_cor", "_StN"]

    def __init__(self, resample_type:str, data:np.ndarray, mean:np.ndarray|Number|None=None, rwf:np.ndarray|None=None, Nresample:int|None=None, blocksize:int|None = None, tag:str|None=None):
        r"""
            param: 
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - data: np.ndarray,                 numpy array of the raw data which becomes resampled
                - mean: np.ndarray|Number|None,     estimate of the central value. If None will be estimated from Data using arithmetic mean (default: None) 
                - rwf: np.ndarray|None,             reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
                - Nresample: int|None,              Number of resamples; required if resample_type=='bst'. (default: None)
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """

        # check and store the resample type 
        if (not isinstance(resample_type,str)) and (resample_type.lower() not in ['jkn','bst']):
            raise ValueError(f"Data type must be ['jkn', 'bst'] but is: {resample_type}")
        self.resample_type:str = resample_type.lower()

        # check that data is numpy array
        if not isinstance(data,np.ndarray):
            raise ValueError(f"data should be np.ndarray but is: {type(data)}")

        # ToDo allow binning

        # resample data. These methods set self._rspl, self._rwf_rspl, self.Nresample
        if self.resample_type == 'jkn':
            self.__jackknife(data=data,rwf=rwf,blocksize=blocksize)
        elif self.resample_type == 'bst':
            if Nresample is None:
                raise ValueError("Data with resample_type='bst' requires parameter Nbst:int")
            
            self.__bootstrap(data=data, rwf=rwf, Nresample=Nresample, blocksize=blocksize)
        else:
            raise RuntimeError(f"Something went wrong initializing Data with:\n - resample_type={resample_type}\n - data={data}")

        # check for provided mean value and if not given, estimate 
        if mean is None:
            self._mean = np.mean(self._rspl, axis = 0)
        else:
            self._mean = mean

        # set the Ndata
        self.Ndata = data.shape[0]

        # finally set tag
        self.tag = tag

    # =================================================================================================================
    # Factories
    # =================================================================================================================
    @staticmethod
    def import_resamples(resample_type:str, rspl:np.ndarray, mean: np.ndarray|Number|None = None, rwf_rspl:np.ndarray|None=None, Ndata:int|None = None, Nresample:int|None = None, tag:str | None = None) -> Self:
        r"""
            param: 
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - rspl: np.ndarray,                 numpy array of the already resampled data
                - mean: np.ndarray|Number|None,     estimateion of the central value. If None will be estimated from Data using arithmetic mean (default: None) 
                - rwf: np.ndarray|None,             reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
                - Ndata: int|None,                  Number of raw data points used for this resample. (default: None)
                - Nresample: int|None,              Number of resamples; may be deduced from rspl.shape[0] (default: None)
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """
        new:Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = resample_type

        # Import from resampled data
        new._rspl = rspl
        new._rwf_rspl = rwf_rspl

        # Import or compute means
        if mean is None:
            new._mean = np.mean(rspl,axis=0)
        else:
            new._mean = mean

        # Set the number of resamples
        if Nresample is None:
            new.Nresample = rspl.shape[0]
        else:
            new.Nresample = Nresample
            if Nresample != rspl.shape[0]:
                raise RuntimeError(f"Data assumes resample axis=0 but Nresample ({Nresample}) does not match rspl.shape ({rspl.shape})")

        # Set the number of data points used for the resample. May be None
        new.Ndata = Ndata
        
        return new

    @staticmethod
    def zeros(resample_type:str, shape: tuple[int] | None = None, Ndata:int|None = None, Nresample:int|None = None, tag:str | None = None) -> Self:
        r"""
            param: 
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - shape: tuple[int],                shape of the observable, resample axis will be ste internally (this class stores the rspl array of size (Nresample, *shape) ), if shape is None one-dimensional Data is used assumed (default: None)
                - Ndata: int|None,                  Number of raw data points used for this resample. (default: None)                
                - Nresample: int|None,              Number of resamples required if resample_type=='bst'. (default: None)  
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """

        # deduce the number of resamples if not provided
        if Nresample is not None:
            pass

        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            # if Nresample is not provided 
            # replace it with number of configs
            # but only if we are jackknifing
            Nresample = Ndata
        else:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")
        
        # ensure it is now set
        if Nresample is None:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        # for jackknife we can still deduce Ndata if not provided
        if Ndata is None:
            if resample_type == "jkn":
                Ndata = Nresample

        # create new instance of Data
        new:Data = Data.__new__(Data)

        # fill in the tag
        new.tag = tag

        # fill in the resample_type
        new.resample_type = resample_type

        # fill in the number of data points (may be None)
        new.Ndata = Ndata

        # fill in the number of resamples
        new.Nresample = Nresample

        # if shape not provided assume one dimensional data
        if shape is None:
            new._rspl = np.zeros((Nresample,))
            new._mean = 0
        else:
            new._rspl = np.zeros((Nresample, *shape))
            new._mean = np.zeros((*shape,))

        # new.cache_field_names.append("_mean")

        return new

    @staticmethod
    def ones(resample_type:str, shape: tuple[int] | None = None, Ndata:int|None = None, Nresample:int|None = None, tag:str | None = None) -> Self:
        r"""
            param: 
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - shape: tuple[int],                shape of the observable, resample axis will be ste internally (this class stores the rspl array of size (Nresample, *shape) ), if shape is None one-dimensional Data is used assumed (default: None)
                - Ndata: int|None,                  Number of raw data points used for this resample. (default: None)                
                - Nresample: int|None,              Number of resamples required if resample_type=='bst'. (default: None)  
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """

        # deduce the number of resamples if not provided
        if Nresample is not None:
            pass

        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            # if Nresample is not provided 
            # replace it with number of configs
            # but only if we are jackknifing
            Nresample = Ndata
        else:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")
        
        # ensure it is now set
        if Nresample is None:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        # for jackknife we can still deduce Ndata if not provided
        if Ndata is None:
            if resample_type == "jkn":
                Ndata = Nresample

        # create new instance of Data
        new:Data = Data.__new__(Data)

        # fill in the tag
        new.tag = tag

        # fill in the resample_type
        new.resample_type = resample_type

        # fill in the number of data points (may be None)
        new.Ndata = Ndata

        # fill in the number of resamples
        new.Nresample = Nresample

        # if shape not provided assume one dimensional data
        if shape is None:
            new._rspl = np.ones((Nresample,))
            new._mean = 0
        else:
            new._rspl = np.ones((Nresample, *shape))
            new._mean = np.ones((*shape,))

        new.cache_field_names.append("_mean")

        return new

    @staticmethod
    def empty(resample_type:str, shape: tuple[int] | None = None, Ndata:int|None = None, Nresample:int|None = None, tag:str | None = None, **kwargs) -> Self:
        r"""
            param: 
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - shape: tuple[int],                shape of the observable, resample axis will be ste internally (this class stores the rspl array of size (Nresample, *shape) ), if shape is None one-dimensional Data is used assumed (default: None)
                - Ndata: int|None,                  Number of raw data points used for this resample. (default: None)                
                - Nresample: int|None,              Number of resamples required if resample_type=='bst'. (default: None)  
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """

        # deduce the number of resamples if not provided
        if Nresample is not None:
            pass

        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            # if Nresample is not provided 
            # replace it with number of configs
            # but only if we are jackknifing
            Nresample = Ndata
        else:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")
        
        # ensure it is now set
        if Nresample is None:
            raise ValueError(f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        # for jackknife we can still deduce Ndata if not provided
        if Ndata is None:
            if resample_type == "jkn":
                Ndata = Nresample

        # create new instance of Data
        new:Data = Data.__new__(Data)

        # fill in the tag
        new.tag = tag

        # fill in the resample_type
        new.resample_type = resample_type

        # fill in the number of data points (may be None)
        new.Ndata = Ndata

        # fill in the number of resamples
        new.Nresample = Nresample

        # if shape not provided assume one dimensional data
        if shape is None:
            new._rspl = np.empty((Nresample,), **kwargs)
            new._mean = None
        else:
            new._rspl = np.empty((Nresample, *shape), **kwargs)
            new._mean = np.empty((*shape,), **kwargs)

        return new

    @staticmethod
    def zeros_like(other:Self, copy_rwf:bool = True) -> Self:
        r"""
            param 
                - other: Data or np.ndarray, Create a Data instance filled with zeros and all other properties deduced from other 
                - copy_rwf: bool,            specify if rwf should be copied or not. If no rwfs are present, this will be ignored (default: True)                
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.zeros_like requires other Data object but got {type(other)}")

        new:Data = Data.__new__(Data)

        # copy tag 
        new.tag = other.tag

        # copy resample type
        new.resample_type = other.resample_type

        # copy number of data points
        new.Ndata = other.Ndata

        # copy number of resamples 
        new.Nresample = other.Nresample

        # copy reweighting factors of provided 
        if other._rwf_rspl is None:
            new._rwf_rspl = None
        elif copy_rwf:
            new._rwf_rspl = other._rwf_rspl.copy()
        else:
            new._rwf_rspl = np.zeros_like(other._rwf_rspl)

        # create new resample array
        new._rspl = np.zeros_like(other._rspl)

        # create new mean array
        if other._rspl.ndim > 1:
            new._mean = np.zeros_like(other._mean)
        else:
            new._mean = 0

        return new

    @staticmethod
    def empty_like(other:Self, copy_rwf:bool = True) -> Self:
        r"""
            param 
                - other: Data or np.ndarray, Create a Data instance filled with zeros and all other properties deduced from other 
                - copy_rwf: bool,            specify if rwf should be copied or not. If no rwfs are present, this will be ignored (default: True)                
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.zeros_like requires other Data object but got {type(other)}")

        new:Data = Data.__new__(Data)

        # copy tag 
        new.tag = other.tag

        # copy resample type
        new.resample_type = other.resample_type

        # copy number of data points
        new.Ndata = other.Ndata

        # copy number of resamples 
        new.Nresample = other.Nresample

        # copy reweighting factors of provided 
        if other._rwf_rspl is None:
            new._rwf_rspl = None
        elif copy_rwf:
            new._rwf_rspl = other._rwf_rspl.copy()
        else:
            new._rwf_rspl = np.empty_like(other._rwf_rspl)

        # create new resample array
        new._rspl = np.empty_like(other._rspl)

        # create new mean array
        if other._rspl.ndim > 1:
            new._mean = np.empty_like(other._mean)
        else:
            new._mean = None

        return new

    @staticmethod
    def full_like(other:Self, value:Number, copy_rwf:bool = True) -> Self:
        r"""
            param 
                - other: Data or np.ndarray, Create a Data instance filled with zeros and all other properties deduced from other 
                - copy_rwf: bool,            specify if rwf should be copied or not. If no rwfs are present, this will be ignored (default: True)                
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.zeros_like requires other Data object but got {type(other)}")

        new:Data = Data.__new__(Data)

        # copy tag 
        new.tag = other.tag

        # copy resample type
        new.resample_type = other.resample_type

        # copy number of data points
        new.Ndata = other.Ndata

        # copy number of resamples 
        new.Nresample = other.Nresample

        # copy reweighting factors of provided 
        if other._rwf_rspl is None:
            new._rwf_rspl = None
        elif copy_rwf:
            new._rwf_rspl = other._rwf_rspl.copy()
        else:
            new._rwf_rspl = np.ones_like(other._rwf_rspl)

        # create new resample array
        new._rspl = np.full_like(other._rspl, value)

        # create new mean array
        if other._rspl.ndim > 1:
            new._mean = np.full_like(other._mean, value)
        else:
            new._mean = value

        return new

    def copy(self, deepcopy:bool = True) -> Self:
        r"""
            param 
                - deepcopy: bool,   specify if the copy should be deep (True) or shallow (False) (default = True) 
        """
        new:Data = Data.__new__(Data)

        # copy tag 
        new.tag = self.tag

        # copy resample type
        new.resample_type = self.resample_type

        # copy number of data points
        new.Ndata = self.Ndata

        # copy number of resamples 
        new.Nresample = self.Nresample

        if deepcopy:
            # copy the data
            new._rspl = self._rspl.copy()

            if isinstance(self._mean, np.ndarray): 
                new._mean = self._mean.copy()
            else: 
                new._mean = self._mean

            # copy reweighting factors of provided 
            if self._rwf_rspl is None:
                new._rwf_rspl = None
            else:
                new._rwf_rspl = self._rwf_rspl.copy()
        else:
            # one can just use new:Data = old 
            raise NotImplementedError
        
        return new

    # =================================================================================================================
    # Resample techniques
    # =================================================================================================================

    @staticmethod
    def blocking(data:np.ndarray, blocksize:int, rwf:np.ndarray|None = None) -> tuple[np.ndarray,np.ndarray|None]:
        # number of configurations
        Ncfg:int = data.shape[0]
        # number of blocks = Number of configs / size of each block 
        Nblock:int = Ncfg // blocksize
        # shape of the observable ignoring the configuration dimension
        shape = data.shape[1:]
        # output array
        blocked_data = np.zeros((Nblock, *shape))

        if rwf is None:
            for k in range(Nblock):
                if k < Nblock-1:
                    blocked_data[k] = np.mean(data[k*blocksize:(k+1)*blocksize,...],axis=0) 
                else:
                    blocked_data[k] = np.mean(data[k*blocksize:,...],axis=0) 

            return blocked_data, None

            raise RuntimeError(f"blocking method must be 'block', 'skip' or None (null) but got {self.params['blocking method']}")
            
        else: # rwf is not None
            data *= rwf[:, *(np.newaxis,)*len(shape)] 
            
            blocked_rwf  = np.zeros((Nblock))
            for k in range(Nblock):
                if k < Nblock-1:
                    blocked_data[k] = np.mean(data[k*blocksize:(k+1)*blocksize,...],axis=0) 
                    blocked_rwf[k]  = np.mean( rwf[k*blocksize:(k+1)*blocksize]    ,axis=0) 
                else:
                    blocked_data[k] = np.mean(data[k*blocksize:,...],axis=0) 
                    blocked_rwf[k]  = np.mean( rwf[k*blocksize:]     ,axis=0) 

            return blocked_data, blocked_rwf

    @ staticmethod
    def jackknife(data:np.ndarray, rwf:np.ndarray|None = None, blocksize: int|None = None) -> np.ndarray:
        r"""
            param:
                - data: np.ndarray,     numpy array of the raw data which is jackknifed 
                - rwf: np.ndarray|None  reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
            Perform a leave-one-out jackknife on the data. Data axis = 0 is assumed
        """

        if blocksize is not None:
            data, rwf = Data.blocking(data=data, blocksize=blocksize,rwf=rwf)
        elif rwf is not None:
            data *= rwf[:,*( (np.newaxis,)*(data.ndim-1) )]

        if rwf is None:
            # The jackknife is defined for all k = 0,1,2...,Ndata-1
            # jkn[k] = \frac{1}{Ndata-1} \sum_{n = 0, n \neq k}^{Ndata-1} dat[n]
            # Which can be translated into 
            # jkn[k] = sum - data[k] / (Ndata-1)
            # where sum = \sum_{n = 0}^{Ndata-1} dat[n]
            data_sum: np.ndarray | Number = np.sum( data, axis = 0 )
            return (data_sum - data) / (data.shape[0] - 1)
        
        else: 
            # We can perform the same trick using reweighting  for all k = 0,1,2...,Ndata-1
            # jkn[k] = \frac{1}{<rwf>} \sum_{n = 0, n \neq k}^{Ndata-1} dat[n] rwf[n]
            # Which can be translated into 
            # jkn[k] = sum - rwf[k] data[k] / (<rwf>-rwf[k])
            # where sum = \sum_{n = 0}^{Ndata-1} rwf[n] dat[n]
            data_sum: np.ndarray | Number = np.sum( data, axis = 0 )
            return (data_sum - data) / ( np.sum(rwf, axis=0) - rwf )[:,*( (np.newaxis,)*(data.ndim-1) )]

    def __jackknife(self, data:np.ndarray, rwf:np.ndarray|None = None, blocksize: int|None = None) -> None:
        r"""
            param:
                - data: np.ndarray,                 numpy array of the raw data which becomes resampled
                - rwf: np.ndarray|None,  reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
            
            Execture jackknife and set class variables

            self._rspl (jackknife resamples)
            self._rwf_rspl (rwf jackknife resmaples if rwf is provided)

            This method is intended to be used during __init__. If you simply want a jackknife of a numpy array please use 
            Data.jackknife(data=...,rwf=...)
        """
        self._rspl     = Data.jackknife(data=data,rwf=rwf, blocksize=blocksize)

        if rwf is not None:
            self._rwf_rspl = Data.jackknife(data=rwf, blocksize=blocksize)

        self.Nresample = self._rspl.shape[0]
         
    @staticmethod
    def bootstrap(data:np.ndarray, Nresample:int, rwf:np.ndarray|None = None, blocksize:int|None = None, method: Callable[[np.ndarray], np.ndarray ] | None = None) -> np.ndarray:
        r"""
            param:
                - data: np.ndarray,      numpy array of the raw data which is jackknifed 
                - Nresample: int,        number of bootstrap resamples 
                - rwf: np.ndarray|None,  reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
                - method: callable,      A method to be calculated on each bootstrap sample, the output will be stored in this class. If None, a simple arithmetic mean is taken (default: None)

            Calculate a bootstrap with repetition. On each resample the `method` is applied and output is stored. By default we store the resample means  
        """
        # we can always reconstruct a new rng with the same seed to get 
        # the same output. Therefore, it is enough to store a temporary 
        # rng for this function. 
        _rng = rng()

        # check if method is provided
        if method is None:
            method = lambda x: np.mean(x, axis = 0) 

        if blocksize is not None:
            data, rwf = Data.blocking(data=data, blocksize=blocksize,rwf=rwf)
        elif rwf is not None:
            data *= rwf[:,*( (np.newaxis,)*(data.ndim-1) )]

        # get the number of 
        Ndata:int = data.shape[0]

        if rwf is None:
            # explicitly run first bootstrap to identify shape of output
            sample_idx:np.ndarray = _rng.integers( 0, Ndata, size=Ndata)
            bst_tmp:np.ndarray = method( data[sample_idx] )

            # now we have all we need
            bst = np.empty( (Nresample, *bst_tmp.shape), dtype=bst_tmp.dtype )
            bst[0] = bst_tmp

            # Now execute the remaining bootstraps
            for k in range(1,Nresample):
                sample_idx:np.ndarray = _rng.integers( 0, Ndata, size=Ndata)
                bst[k] = method(data[sample_idx])
        else:
            # explicitly run first bootstrap to identify shape of output
            sample_idx:np.ndarray = _rng.integers( 0, Ndata, size=Ndata)
            bst_tmp:np.ndarray = method( data[sample_idx] ) / np.mean(rwf[sample_idx],axis=0)

            # now we have all we need
            bst = np.empty( (Nresample, *bst_tmp.shape), dtype=bst_tmp.dtype )
            bst[0] = bst_tmp

            # Now execute the remaining bootstraps
            for k in range(1,Nresample):
                sample_idx:np.ndarray = _rng.integers( 0, Ndata, size=Ndata)
                bst[k] = method(data[sample_idx]) / np.mean(rwf[sample_idx],axis=0)

        return bst

    def __bootstrap(self, data: np.ndarray, Nresample:int, rwf:np.ndarray|None = None, blocksize:int| None = None) -> None:
        r"""
            param:
                - data: np.ndarray,      numpy array of the raw data which becomes resampled
                - Nresample: int,        number of bootstrap resamples   
                - rwf: np.ndarray|None,  reweighting factors to be used as reweighted estimates <w O> / <w>. if None, no reweighting applied (default: None)
            
            Execture bootstrap and set class variables

            self._rspl (jackknife resamples)
            self._rwf_rspl (rwf jackknife resmaples if rwf is provided)

            This method is intended to be used during __init__. If you simply want a bootstrap of a numpy array please use 
            Data.bootstrap(data=...,Nresample=...,rwf=...,method=...)
        """
        self._rspl = Data.bootstrap(data=data,Nresample=Nresample,rwf=rwf,blocksize=blocksize)

        if rwf is not None:
            self._rwf_rspl = Data.bootstrap(data=rwf,Nresample=Nresample,blocksize=blocksize)

        self.Nresample = Nresample
        self.blocksize = blocksize

    @staticmethod
    def pseudoBootstrap(mean:float, sdev:float, Nresample:int, Ndata:int|None = None, tag:str|None = None) -> Any:
        r"""
            param: 
                - mean: np.ndarray|Number|None,     estimate of the central value. mean (location) of normal distribution 
                - sdev: np.ndarray|Number|None,     estimate of standard deviation. width of normal distribution
                - Nresample: int|None,              Number of resamples; 
                - tag: str|None=None,               A string describing the resample data. (default: None)
        """
        new:Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = "bst"
        new.Ndata = Ndata
        new.Nresample = Nresample
        new._rspl = new.get_rng().normal(mean,sdev,size=(Nresample,))
        new._mean = mean
        new._serr = sdev

        # to prevent overwriting of _serr on first call of new.serr() 
        # remove the _serr field from the caching. This way, new.serr()
        # will always simply return this value here
        # new.cache_field_names.remove('_serr')

        return new
    
    @staticmethod
    def get_rng() -> np.random.Generator:
        return rng()

    @property
    def bootstrap_sample_ids(self) -> np.ndarray:
        if self.Ndata is None:
            raise RuntimeError("Getting bootstrap sample ids requires knowledge on the number of data points being resampled (self.Ndata) which is not set.")
        _rng = rng()
        return _rng.integers( 0, self.Ndata, size=(self.Nresample,self.Ndata) )

    # =================================================================================================================
    # Internal checks and verification
    # =================================================================================================================

    def __check_other(self, other:Self|np.ndarray|Number) -> None:
        matches_flag:bool = True

        # allow type Data and type np.ndarray
        matches_flag = isinstance(other, (Data,np.ndarray,Number))
        if not matches_flag:
            raise ValueError(f"other is expected to be of type Data,np.ndarray, or Number=({Number}) but is: {type(other)}")

        # allow only same resample_type
        if isinstance(other,Data):
            #matches_flag = self.resample_type == other.resample_type 
            #if not matches_flag:
            #    raise ValueError(f"other is expected to have same resample type ({self.resample_type}) but has: {other.resample_type}")

            # allow only same number of resample 
            matches_flag = self.Nresample == other.Nresample
            if not matches_flag:
                raise ValueError(f"other is expected to have same Nresample ({self.Nresample}) but has: {other.Nresample}")
        elif isinstance(other,np.ndarray):
            # we can not check for resample type, assuming the user know what they are doing

            # allow only same number of resample 
            matches_flag = is_broadcastable(self._rspl, other)

            if not matches_flag:
                raise ValueError(f"other is expected to have broadcastable shape ({self._rspl.shape}) but has: {other.shape}")
        elif isinstance(other,Number):
            # you can always interact with a number
            pass 

    def __delete_cached(self):
        for attr in self.cache_field_names:
            if hasattr(self,attr):
                delattr(self,attr)

    def delete_cache(self):
        self.__delete_cached()

    def imported(self) -> None:
        self.delete_cache()

        if hasattr(self,"_mean"):
            delattr(self,"_mean")

        self.mean

    # =================================================================================================================
    # Arithmetic overloads
    # =================================================================================================================

    def __add__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            #esample_type:str, rspl:np.ndarray, mean: np.ndarray|Number|None = None, rwf_rspl:np.ndarray|None=None, Ndata:int|None = None, Nresample:int|None = None, tag:str | None = None
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl + other._rspl,
                mean          = self._mean+ other._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl + other,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __radd__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other._rspl + self._rspl,
                mean          = other._mean + self._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other + self._rspl,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __iadd__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            self._rspl += other._rspl
            self._mean+=other._mean
        elif isinstance(other,(np.ndarray,Number)):
            self._rspl += other
            self._mean = np.mean(self._rspl,axis=0)
        else:
            raise NotImplemented

        # self.__delete_cached()

        return self

    def __sub__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl - other._rspl,
                mean          = self._mean - other._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl - other,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __rsub__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other._rspl - self._rspl,
                mean          = other._mean - self._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other - self._rspl,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample, 
            )
        else:
            raise NotImplemented

    def __isub__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            self._rspl -= other._rspl
            self._mean -= other._mean
        elif isinstance(other,(np.ndarray,Number)):
            self._rspl -= other
            self._mean = np.mean(self._rspl,axis=0)
        else:
            raise NotImplemented

        self.__delete_cached()

        return self

    def __mul__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl * other._rspl,
                mean          = self._mean * other._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl * other,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __rmul__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other._rspl * self._rspl,
                mean          = other._mean * self._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other * self._rspl,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample, 
            )
        else:
            raise NotImplemented

    def __imul__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            self._rspl *= other._rspl
            self._mean *= other._mean
        elif isinstance(other,(np.ndarray,Number)):
            self._rspl *= other
            self._mean = np.mean(self._rspl,axis=0)
        else:
            raise NotImplemented


        self.__delete_cached()

        return self

    def __truediv__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl / other._rspl,
                mean          = self._mean / other._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl / other,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __rtruediv__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other._rspl / self._rspl,
                mean          = other._mean / self._mean,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = other / self._rspl,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented

    def __itruediv__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            self._rspl /= other._rspl
            self._mean /= other._mean
        elif isinstance(other,(np.ndarray,Number)):
            self._rspl /= other
            self._mean = np.mean(self._rspl,axis=0)
        else:
            raise NotImplemented

        self.__delete_cached()

        return self

    def __pow__(self, other:Self|np.ndarray|Number) -> Self:
        self.__check_other(other)

        if isinstance(other,Data):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl**(other._rspl),
                mean          = self._mean**(other._mean),
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        elif isinstance(other,(np.ndarray,Number)):
            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl**other,
                rwf_rspl      = self._rwf_rspl,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )
        else:
            raise NotImplemented
        
    def __neg__(self) -> Self:
        return Data.import_resamples(
            resample_type = self.resample_type,
            rspl          = -self._rspl,
            mean          = -self._mean,
            rwf_rspl      = self._rwf_rspl,
            Ndata         = self.Ndata,
            Nresample     = self.Nresample,
        )

    # =================================================================================================================
    # Comparison overloads
    # =================================================================================================================

    def __lt__(self, other:Self|np.ndarray|Real) -> np.ndarray:
        self.__check_other(other)

        if isinstance(other,Data):
            return self._rspl < other._rspl
        elif isinstance(other,(np.ndarray,Real)):
            return self._rspl < other
        else:
            raise NotImplemented

    def __le__(self, other:Self|np.ndarray|Real) -> np.ndarray:
        self.__check_other(other)

        if isinstance(other,Data):
            return self._rspl <= other._rspl
        elif isinstance(other,(np.ndarray,Real)):
            return self._rspl <= other
        else:
            raise NotImplemented

    def __gt__(self, other:Self|np.ndarray|Real) -> np.ndarray:
        self.__check_other(other)

        if isinstance(other,Data):
            return self._rspl > other._rspl
        elif isinstance(other,(np.ndarray,Real)):
            return self._rspl > other
        else:
            raise NotImplemented

    def __ge__(self, other:Self|np.ndarray|Real) -> np.ndarray:
        self.__check_other(other)

        if isinstance(other,Data):
            return self._rspl >= other._rspl
        elif isinstance(other,(np.ndarray,Real)):
            return self._rspl >= other
        else:
            raise NotImplemented

    # =================================================================================================================
    # Statistics 
    # =================================================================================================================
    
    @property
    def rspl(self) -> np.ndarray:
        if hasattr(self, "_rspl"):
            return self._rspl
        else:
            raise RuntimeError(f"Something went wrong, Data doesn't have rsmpl")

        # elements of resample may be replaced
        # entire replacement of rspl array is forbidden (unlike for mean) 

    @property
    def mean(self) -> np.ndarray|Number:
        # # elements of mean (inc case of array type array) may be replaced
        # if hasattr(self, "_mean"):
        #     return self._mean

        self._mean:np.ndarray | Number = np.mean(self._rspl, axis=0)
        return self._mean

    @mean.setter
    def mean(self, value: np.ndarray|Number) -> None:
        if hasattr(self, "_mean"):
            if isinstance(self._mean, np.ndarray) and isinstance(value, np.ndarray):
                # check for shape 
                if self._mean.shape != value.shape:
                    raise ValueError("Replacing mean with array requires same shape")
                self._mean = value

            elif isinstance(self._mean, type(None)) and isinstance(value, np.ndarray):
                self._mean = value

            elif isinstance(self._mean, np.ndarray) and isinstance(value, Number):
                raise ValueError("Replacing mean (array) with value (number) is prohibited")

            elif isinstance(value, np.ndarray) and isinstance(self._mean, Number):         
                raise ValueError("Replacing mean (number) with value (array) is prohibited")
            
            elif isinstance(self._mean, (Number,type(None))) and isinstance(value, Number):
                self._mean = value

            else:
                raise ValueError(f"Replacing mean ({type(self._mean)}) with value ({type(value)}) is prohibited")
        else:
            if isinstance(value, np.ndarray):
                if self._rspl.shape[1:] != value.shape:
                    raise ValueError("Setting mean with array requires matching shape with resamples")
                self._mean = value

            if  isinstance(value, Number):
                if self._rspl.ndim != 1:
                    raise ValueError("Setting mean with array requires matching shape with resamples")
                self._mean = value

    @property
    def serr(self) -> np.ndarray|Number:
        # # elements of serr may be replaced
        # # entire replacement of serr array is forbidden (unlike for mean) 
        # if hasattr(self,"_serr"):
        #     return self._serr
        
        if self.resample_type == 'jkn':
            self._serr = np.sqrt( (self.Nresample-1) ) * np.std( self._rspl, axis = 0 )
            # or equivalently 
            # self._serr = np.sqrt( ((self.Nresample-1)/(self.Nresample)) * np.sum( self.data, axis = 0) )
        elif self.resample_type == 'bst':
            self._serr = np.std( self._rspl, axis = 0, ddof=1 )
        else:
            raise NotImplemented

        return self._serr

    @property
    def cov(self) -> np.ndarray:
        # if hasattr(self,"_cov"):
        #     return self._cov

        if len(self._rspl.shape) != 2:
            raise RuntimeError(f"Covariance estimation is only implemented for Data of shape (N, Nobs), with N being the number of resamples, but is: {self._rspl.shape}")

        if self.resample_type == 'jkn':
            self._cov = ( self.Nresample-1 ) * np.cov( self._rspl, rowvar=False, bias = False )
        elif self.resample_type == 'bst':
            self._cov = np.cov( self._rspl, rowvar=False, bias = True )
        else:
            raise NotImplemented

        return self._cov

    def cov_uncertainty(self) -> np.ndarray:
        # Estimate the error of the covariance over a second order bootstrap
        if len(self._rspl.shape) != 2:
            raise RuntimeError(f"Correlation estimation is only implemented for Data of shape (N, Nobs), with N being the number of resamples, but is: {self._rspl.shape}")

        cov_per_bst:np.ndarray = Data.bootstrap(
            self._rspl, Nbst=self.Nbst_inner, method=lambda x: np.cov(x,rowvar=False)
        )

        return np.std(cov_per_bst, axis=0,ddof=1)

    @property
    def cor(self) -> np.ndarray:
        # if hasattr(self,"_cor"):
        #     return self._cor

        if len(self._rspl.shape) != 2:
            raise RuntimeError(f"Correlation estimation is only implemented for Data of shape (N, Nobs), with N being the number of resamples, but is: {self._rspl.shape}")

        self._cor = np.corrcoef(self._rspl, rowvar=False)

        return self._cor

    def cor_uncertainty(self) -> np.ndarray:
        # Estimate the error of the covariance over a second order bootstrap
        if len(self._rspl.shape) != 2:
            raise RuntimeError(f"Correlation estimation is only implemented for Data of shape (N, Nobs), with N being the number of resamples, but is: {self._rspl.shape}")

        cor_per_bst:np.ndarray = Data.bootstrap(
            self._rspl, Nbst=self.Nbst_inner, method=lambda x: np.corrcoef(x,rowvar=False)
        )

        return np.std(cor_per_bst, axis=0,ddof=1)

    @property
    def resample_serr(self) -> np.ndarray|Number:
        return self.serr

    @property
    def resample_cov(self) -> np.ndarray|Number:
        return self.cov

    @property
    def StN(self) -> np.ndarray | Number:
        # if hasattr(self,"_StN"):
        #     return self._StN

        if isinstance(self.mean, Number) or isinstance(self.serr, Number):
            if self.serr == 0:
                self._StN:np.ndarray | Number = np.inf
            else:
                self._StN: np.ndarray | Number = np.abs(self.mean) / self.serr
        else:
            if np.any(self.serr) == 0:
                self._StN:np.ndarray | Number = np.full_like(self.mean, np.inf)

                mask = self.serr != 0
            
                self._StN[mask] = np.abs(self.mean[mask]) / self.serr[mask] # type: ignore
            else:
                self._StN: np.ndarray | Number = np.abs(self.mean) / self.serr

        return self._StN 

    @property
    def serr_normal_approx(self):
        # Assuming jackknife (X_k) resamples are normaly distrubuted:
        # X_k ~ N(\mu, \sigma^2/Ncfg)
        # Than by definition (median m): the median absolute deviation 
        # P( |X - \mu| < m ) = 0.5 
        # or equivalently:
        # P(-m < X-\mu < m) = 0.5
        # Since, normal distribution is assumed, this can be calulated using the known 
        # cumulativ distribution function: CDF(m)
        # P( |X - \mu| < m ) = CDF(m/(\sigma/sqrt(Ncfg))) - CDF(-m/(\sigma/sqrt(Ncfg))) = 0.5
        #                    = CDF(m/(\sigma/sqrt(Ncfg))) - 1 + CDF(m/(\sigma/sqrt(Ncfg))) = 0.5
        #  =>                2*CDF(m/(\sigma/sqrt(Ncfg))) = 1.5
        #  =>                  CDF(m/(\sigma/sqrt(Ncfg))) = 0.75
        # Thus inverting the CDF gives
        # m/(\sigma/sqrt(Ncfg)) = CDF^{-1}(0.75) = 0.674....
        # or
        # sigma \approx sqrt(Ncfg) * m / 0.674...
        median_data: np.ndarray = np.median(self._rspl, axis=0)

        # Compute |X-\mu|, shape=(Nresample-M, *)
        abs_dev: np.ndarray = np.abs(self._rspl-median_data)

        # compute m, shape=(*,)
        median_abs_dev: np.ndarray = np.median(abs_dev,axis=0)

        # Compute standard deviation using the approximation derived above
        # shape=(*,)
        # The factor sqrt(N-1) translates the deviation of the jackknifes to the standard deviation
        # i.e. the standard error. this is compatible with self.serr
        if self.resample_type == "jkn":
            serr:np.ndarray = (median_abs_dev*1.48260221850560186054) * np.sqrt(self.Nresample-1)
        else:
            serr:np.ndarray = (median_abs_dev*1.48260221850560186054) 

        return serr

    def get_dist_data(self) -> np.ndarray:
        if self.resample_type == "bst":
            return self._rspl 
        elif self.resample_type == "jkn":
            return (self._rspl - self.mean) * np.sqrt(self.Ndata-1) + self.mean 
        else:
            raise NotImplemented

    # =================================================================================================================
    # Representations
    # =================================================================================================================

    def __repr__(self):
        repr:str = "Data"

        if self.tag is not None:
            repr+= f"({self.tag})"   

        repr+= f"[{self.resample_type}"

        if self._rspl is None or self.Nresample is None:
            repr+= ", unset"
        else:
            repr+= f",  Nresample={self.Nresample}"

        if self.shape:
            repr+= f", shape={self.shape}"

        repr+= "]"        

        return repr

    def gvar(self, correlated:bool = False) -> Any:
        import gvar as gv 

        if correlated:
            return gv.gvar( self.mean, self.cov )
        else:
            return gv.gvar( self.mean, self.serr )

    def to_dict(self) -> dict[str, np.ndarray|Number]:
        # to maintain some backward compatibility we can 
        # simply decompose the class into a dict object
        # with 'known' keys
        return {
            "est": self.mean,
            "err": self.serr,
            "res": self.rspl
        }

    # =================================================================================================================
    # hdf5 (de-)serialization
    # =================================================================================================================
    def serialize(self, h5f: h5.Group, node:str|None = None) -> None:
        if node is None:
            grp = h5f
        else:
            grp = h5f.create_group(node)

        grp.create_dataset("resample_type", data=self.resample_type)
        grp.create_dataset("mean", data=self._mean)
        grp.create_dataset("resamples", data=self._rspl)
        grp.create_dataset("Nresample", data=self.Nresample)
        if self._rwf_rspl is not None:
            grp.create_dataset("rwf_resamples", data=self.Ndata)
        if self.Ndata is not None:
            grp.create_dataset("Ndata", data=self.Ndata)
        if self.tag is not None:
            grp.create_dataset("tag", data=self.Ndata)
        if self.Nbst_inner != Data.Nbst_inner:
            grp.create_dataset("Nbst_inner", data=self.Nbst_inner)
        if self.cache_field_names != Data.cache_field_names:
            grp.create_dataset("cache_field_names", data=self.cache_field_names)
    
    @staticmethod
    def deserialize(h5f: h5.Group, node:str|None = None) -> Self:
        if node is None:
            grp = h5f
        else:
            grp = h5f[node]

        new:Data = Data.__new__(Data)

        new.resample_type = grp["resample_type"][()].decode('utf-8')
        new._mean = grp["mean"][()]
        new._rspl = grp["resamples"][()]
        new.Nresample = grp["Nresample"][()]

        if "rwf_resamples" in grp:
            new._rwf_rspl = grp["rwf_resamples"][()]
        if "Ndata" in grp:
            new.Ndata = grp["Ndata"][()]
        if "Nbst_inner" in grp:
            new.Nbst_inner = grp["Nbst_inner"][()]
        if "cache_field_names" in grp:
            new.cache_field_names = grp["cache_field_names"][()]

        return new

    # =================================================================================================================
    # Interoperability with numpy
    # =================================================================================================================

    def __getitem__(self, idx: Any ) -> Self|np.ndarray|Number:
        if isinstance(idx, tuple):
            # Resample of a single number may have a mean of a single float
            # One might want to broadcast the resamples and thus expand the 
            # dimensionality using np.newaxis/None
            # In this case we want to expand the float to an array containing 
            # a single number 
            if (any((item is None) for item in idx)) and not isinstance(self._mean, np.ndarray) and hasattr(self, "_mean"):
                mean_tmp = np.asarray(self._mean)[*idx]
            elif hasattr(self, "_mean"):
                mean_tmp = self._mean[*idx]
            else:
                mean_tmp = None

            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl[:, *idx],
                mean          = mean_tmp,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )

        else: # try your luckimport traceback as tb
            if (idx is None or np.newaxis == idx) and not isinstance(self._mean, np.ndarray) and hasattr(self, "_mean"):
                mean_tmp = np.asarray([self._mean])
            elif hasattr(self, "_mean"):
                mean_tmp = self._mean[idx]
            else:
                mean_tmp = None

            return Data.import_resamples(
                resample_type = self.resample_type,
                rspl          = self._rspl[:, idx],
                mean          = mean_tmp,
                Ndata         = self.Ndata,
                Nresample     = self.Nresample,
            )  

    def __setitem__(self, idx: Any, value: Self|np.ndarray|Number) -> None:
        if isinstance(value, Data):
            if self.Nresample != value.Nresample:
                raise ValueError(f"Setting requires Data with same number of resamples: {self.Nresample=} != {value.Nresample=}")
            if self.resample_type != value.resample_type:
                raise ValueError(f"Setting requires Data with same resample type: self({self.resample_type}) != value({value.resample_type})")

            self._rspl[:,idx] = value._rspl
            self._mean[idx] = value.mean

        elif isinstance(value, (np.ndarray,Number)):
            # no checkup possible, we have to trust the user 
            self._rspl[:,idx] = value
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise ValueError( f"Setting requires value to be of type Data, np.ndarray, or Number but is: {type(value)}" )

    def reshape(self, shape, *args, **kwargs):
        # often means etc may want to be in the same format hence we will simply recompute them
        self.delete_cache()

        self._rspl = self._rspl.reshape( (self.Nresample,*shape), *args,**kwargs)
        self._mean = self._mean.reshape( (*shape,), *args,**kwargs )

    @property
    def shape(self) -> tuple[int,...]:
        if isinstance(self.mean,np.ndarray):
            return self.mean.shape
        else: 
            return tuple()

    @property
    def ndim(self) -> int:
        return self.mean.ndim

    def __array__(self, copy = None, dtype=None):
        if dtype or copy:
            return np.asarray(self._rspl, copy=copy, dtype=dtype)

        return self._rspl

    def __array_ufunc__(self, ufunc, method, *inputs:tuple[Self|np.ndarray], **kwargs:dict[str,Any]) -> Any|np.ndarray|Number:
        # Handle numpy ufuncs to preserve Data type
        if method != '__call__':
            return NotImplemented
        
        resample_types: np.ndarray = np.unique( [input.resample_type for input in inputs if isinstance(input, Data)] ) 
        Nresamples: np.ndarray = np.unique( [input.Nresample for input in inputs if isinstance(input, Data)] ) 
        Ndatas: np.ndarray = np.unique( [input.Ndata for input in inputs if isinstance(input, Data)] ) 

        if len(resample_types) != 1:
            raise RuntimeError (f"All resample_types must be the same, but found: {resample_types}")
        else:
            resample_type:str = resample_types[0]

        if len(Nresamples) != 1:
            raise RuntimeError (f"All Nresamples must be the same, but found: {Nresamples}")
        else:
            Nresample:int = Nresamples[0] 

        if len(Ndatas) != 1:
            raise RuntimeError (f"All Nresamples must be the same, but found: {Ndatas}")
        else:
            Ndata:int = Ndatas[0] 

        # Extract underlying arrays for all inputs
        args = [ input._rspl if isinstance(input, Data) else input for input in inputs ]
        
        # Compute ufunc result
        result: np.ndarray = ufunc(*args, **kwargs)

        # Put result back into Data class
        if isinstance(result, np.ndarray):
            return Data.import_resamples(
                resample_type = resample_type, 
                rspl          = result,
                Ndata         = Ndata,
                Nresample     = Nresample
            )
        elif isinstance(result,Data):
            return result
        else:
            raise NotImplementedError(
                f"Dispatch of numpy ufunction not succesful: result is not type array but: {type(result)}"
            )

    def __array_function__(self, func, types, args:tuple[Any,...], kwargs:dict[str,Any]) -> Self | np.ndarray | Number:
        """
            Implements interoperability with a greater numpy ecosystem  
            https://numpy.org/neps/nep-0018-array-function-protocol.html
        """
        # By default we dispatch the function to Data.data. However, a few functions
        # may have a specific implementation. For this the following cases are considered
        if func in HANDLED_FUNCTIONS_DATA:
            return HANDLED_FUNCTIONS_DATA[func](*args,**kwargs)

        if not all(issubclass(t, (np.ndarray, Data)) for t in types):
            return NotImplemented

        # Recursively replace any Data instance with its internal ndarray
        resample_types:list  = []
        Nresamples: list = []
        Ndatas: list = []
        def unwrap(x:Any) -> Any:
            if isinstance(x, Data):
                resample_types.append(x.resample_type)
                Nresamples.append(x.Nresample)
                Ndatas.append(x.Ndata)

                return x._rspl
            elif isinstance(x, (tuple, list)):
                return type(x)(unwrap(i) for i in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            else:
                return x
            
        args = unwrap(args)
        kwargs = unwrap(kwargs)

        resample_types = list(np.unique( resample_types ))
        Nresamples = list(np.unique( Nresamples ))
        # Ndatas = list(np.unique( Ndatas ))

        if len(resample_types) != 1:
            raise RuntimeError (f"All resample_types must be the same, but found: {resample_types}")
        else:
            resample_type:str = resample_types[0]

        if len(Nresamples) != 1:
            raise RuntimeError (f"All Nresamples must be the same, but found: {Nresamples}")
        else:
            Nresample:int = Nresamples[0] 

        # if len(Ndatas) != 1:
        #     raise RuntimeError (f"All Nresamples must be the same, but found: {Ndatas}")
        # else:
        #     Ndata:int = Ndatas[0] 
        
        result:np.ndarray = func(*args,**kwargs)

        if isinstance(result, np.ndarray):
            return Data.import_resamples(
                resample_type = resample_type, 
                rspl          = result,
                # Ndata         = Ndata,
                Nresample     = Nresample
            )
        if isinstance(result, Number):
            return result
        else:
            raise NotImplementedError(
                f"Dispatch of numpy function not succesful: result is not type array but: {type(result)}"
            )

# =====================================================================================================================
# Numpy exceptions
# These functions have a specialized behviour.
# =====================================================================================================================

@implements(np.mean)
def mean(a:Data, axis=None, dtype=None, out=None, keepdims=_NoValue, *, where=_NoValue) -> Data | np.ndarray | Number:
    # compute the mean 
    if isinstance(a,Data):
        out_array: np.ndarray = np.mean(a._rspl, axis=axis, dtype=dtype,out=out,keepdims=keepdims, where=where)
    else:
        raise ValueError(f"a is expected to be of type Data or np.array but is: {type(a)}")

    # if axis == resample_axis(=0) we took a mean over the resample axis thus don't have any resample information
    # in all other cases we import the result into Data and return data
    if axis != 0:
        return Data.import_resamples(
            resample_type = a.resample_type,
            rspl          = out_array,
            Ndata         = a.Ndata,
            Nresample     = a.Nresample,
        )
    
    return out_array 
