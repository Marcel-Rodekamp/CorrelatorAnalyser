import numpy as np
from numpy import _NoValue  # type: ignore

import h5py as h5
import gvar as gv

from typing import Union, Self, Any, Callable
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

SEED: int = 88567
np.random.seed(SEED)
rng: Callable[[], np.random.Generator] = lambda: np.random.default_rng(seed=SEED)


class Data:
    # The resample type: 'bst': bootstraps (with replacement), 'jkn': leave-1-out jackknife,
    # None: Gaussian error propagation via gvar
    resample_type: str | None

    # central values (arithmetic mean over resamples, gv.mean of gvars, or provided estimate)
    _mean: np.ndarray

    # resamples of the raw data; None when resample_type is None (gvar mode)
    _rspl: np.ndarray | None

    # gvar array for Gaussian error propagation mode; None when resample_type is set
    _gvar: gv.GVar | np.ndarray  # gvar.GVar scalar or np.ndarray of gvar.GVar objects

    # number of resamples (axis=0 length of _rspl); None in gvar mode
    Nresample: int | None

    # number of data points used to be resampled. May not be given
    Ndata: int | None = None

    # determines the size of each bin if the data is blocked before resample
    blocksize: int | None = None

    # a string tag attached to the data set
    tag: str | None = None

    # resamples of reweighting factors if provided
    _rwf_rspl: np.ndarray | None = None

    # for error of error analysis a inner bootstrap is executed using this number of samples
    Nbst_inner: int = 100

    # determine whether DATA.mean recomputes the mean over resamples
    # or simply returns the ._mean
    locked_mean: bool = False

    # =================================================================================================================
    # Constructor
    # =================================================================================================================

    def __init__(self, resample_type: str | None, data: np.ndarray,
                 mean: np.ndarray | Number | None = None, rwf: np.ndarray | None = None,
                 Nresample: int | None = None, blocksize: int | None = None,
                 tag: str | None = None, locked_mean: bool = False):
        r"""
            param:
                - resample_type: str | None,            'bst', 'jkn', or None for Gaussian error propagation
                - data: np.ndarray,                     numpy array of the raw data which becomes resampled
                - mean: np.ndarray|Number|None,         estimate of the central value. If None will be estimated
                                                        from data using arithmetic mean (default: None)
                - rwf: np.ndarray|None,                 reweighting factors; not supported in gvar mode (default: None)
                - Nresample: int|None,                  Number of resamples; required if resample_type=='bst'
                - blocksize: int|None,                  Block size; in gvar mode blocks before computing mean/std
                - tag: str|None,                        A string describing the data (default: None)
                - locked_mean: bool,                    Lock mean (ignored in gvar mode) (default: False)
        """
        # check that data is numpy array
        if not isinstance(data, np.ndarray):
            raise ValueError(f"data should be np.ndarray but is: {type(data)}")

        self.tag = tag
        self.locked_mean = locked_mean
        self.Ndata = data.shape[0]
        self.blocksize = blocksize

        # ---- Gaussian error propagation mode ----
        if resample_type is None:
            self.resample_type = None
            self._rspl = None
            self._gvar = None
            self._rwf_rspl = None
            self.Nresample = None

            if blocksize is not None:
                blocked_data, _ = Data.blocking(data=data, blocksize=blocksize, rwf=rwf)
                m = np.mean(blocked_data, axis=0)
                s = np.std(blocked_data, axis=0, ddof=1) / np.sqrt(blocked_data.shape[0])
            else:
                if rwf is not None:
                    data = data * rwf[:, *(np.newaxis,) * len(data.shape[1:])] / np.mean(rwf)
                m = np.mean(data, axis=0)
                s = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])

            if mean is not None:
                m = mean

            self._gvar = gv.gvar(m, s)
            self._mean = gv.mean(self._gvar)
            return

        # ---- Resample mode ----
        if (not isinstance(resample_type, str)) or (resample_type.lower() not in ['jkn', 'bst']):
            raise ValueError(f"Data resample_type must be 'jkn', 'bst', or None but is: {resample_type}")
        self.resample_type = resample_type.lower()
        self._gvar = None

        if self.resample_type == 'jkn':
            self.__jackknife(data=data, rwf=rwf, blocksize=blocksize)
        elif self.resample_type == 'bst':
            if Nresample is None:
                raise ValueError("Data with resample_type='bst' requires parameter Nresample:int")
            self.__bootstrap(data=data, rwf=rwf, Nresample=Nresample, blocksize=blocksize)
        else:
            raise RuntimeError(f"Something went wrong initializing Data with:\n"
                               f" - resample_type={resample_type}\n - data={data}")

        if mean is None:
            self._mean = np.mean(self._rspl, axis=0)
        else:
            self._mean = mean

    # =================================================================================================================
    # Factories
    # =================================================================================================================

    @staticmethod
    def import_gvar(g: Any, mean: np.ndarray | Number | None = None, Ndata: int | None = None, tag: str | None = None,
                    locked_mean: bool = False) -> 'Data':
        r"""
            Import gvar objects directly into a gvar-mode Data instance.
            Correlations encoded in the gvar objects are fully preserved.

            param:
                - g: gvar.GVar or np.ndarray of gvar.GVar,  gvar object(s) to import
                - mean: np.array, Number or None,           mean value, if not provide taken from g
                - Ndata: int|None,                           Number of raw data points (default: None)
                - tag: str|None,                             A string tag (default: None)
                - locked_mean: bool,                         Has no effect in gvar mode (default: False)
        """
        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = None
        new._rspl = None
        new._rwf_rspl = None
        new.Nresample = None
        new.Ndata = Ndata
        new.locked_mean = locked_mean
        new.blocksize = None

        new._gvar = g
        if mean is None:
            new._mean = gv.mean(new._gvar)
        else:
            new._mean = mean

        return new

    @staticmethod
    def import_resamples(resample_type: str, rspl: np.ndarray, mean: np.ndarray | Number | None = None,
                         rwf_rspl: np.ndarray | None = None, Ndata: int | None = None,
                         Nresample: int | None = None, tag: str | None = None,
                         locked_mean: bool = False) -> 'Data':
        r"""
            param:
                - resample_type: str,               'bst' or 'jkn' for bootstrap or jackknife respectively
                - rspl: np.ndarray,                 numpy array of the already resampled data
                - mean: np.ndarray|Number|None,     estimation of the central value. If None will be estimated
                                                    from rspl using arithmetic mean (default: None)
                - rwf_rspl: np.ndarray|None,        resampled reweighting factors (default: None)
                - Ndata: int|None,                  Number of raw data points used for this resample (default: None)
                - Nresample: int|None,              Number of resamples; may be deduced from rspl.shape[0] (default: None)
                - tag: str|None,                    A string describing the resample data (default: None)
        """
        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = resample_type
        new.locked_mean = locked_mean
        new.blocksize = None

        # Resample mode — gvar is not set
        new._rspl = rspl
        new._rwf_rspl = rwf_rspl
        new._gvar = None

        if mean is None:
            new._mean = np.mean(rspl, axis=0)
        else:
            new._mean = mean

        if Nresample is None:
            new.Nresample = rspl.shape[0]
        else:
            new.Nresample = Nresample
            if Nresample != rspl.shape[0]:
                raise RuntimeError(
                    f"Data assumes resample axis=0 but Nresample ({Nresample}) does not match rspl.shape ({rspl.shape})")

        new.Ndata = Ndata

        return new

    @staticmethod
    def zeros(resample_type: str | None, shape: tuple[int] | None = None, Ndata: int | None = None,
              Nresample: int | None = None, tag: str | None = None, locked_mean: bool = False,
              **array_kwargs) -> 'Data':
        r"""
            param:
                - resample_type: str | None,        'bst', 'jkn', or None for gvar mode
                - shape: tuple[int] | None,         shape of the observable (default: None = scalar)
                - Ndata: int|None,                  Number of raw data points (default: None)
                - Nresample: int|None,              Number of resamples (default: None)
                - tag: str|None,                    A string tag (default: None)
        """
        if resample_type is None:
            new: Data = Data.__new__(Data)
            new.tag = tag
            new.resample_type = None
            new._rspl = None
            new._rwf_rspl = None
            new.Nresample = None
            new.Ndata = Ndata
            new.locked_mean = locked_mean
            new.blocksize = None
            if shape is None:
                new._gvar = gv.gvar(0.0, 0.0)
            else:
                new._gvar = gv.gvar(np.zeros(shape, **array_kwargs), np.zeros(shape, **array_kwargs))
            new._mean = gv.mean(new._gvar)
            return new

        # deduce the number of resamples if not provided
        if Nresample is not None:
            pass
        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            Nresample = Ndata
        else:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Nresample is None:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Ndata is None and resample_type == "jkn":
            Ndata = Nresample

        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = resample_type
        new.Ndata = Ndata
        new.Nresample = Nresample
        new.locked_mean = locked_mean
        new._gvar = None
        new.blocksize = None

        if shape is None:
            new._rspl = np.zeros((Nresample,), **array_kwargs)
            new._mean = 0
        else:
            new._rspl = np.zeros((Nresample, *shape), **array_kwargs)
            new._mean = np.zeros((*shape,), **array_kwargs)

        return new

    @staticmethod
    def ones(resample_type: str | None, shape: tuple[int] | None = None, Ndata: int | None = None,
             Nresample: int | None = None, tag: str | None = None, locked_mean: bool = False) -> 'Data':
        r"""
            param:
                - resample_type: str | None,        'bst', 'jkn', or None for gvar mode
                - shape: tuple[int] | None,         shape of the observable (default: None = scalar)
                - Ndata: int|None,                  Number of raw data points (default: None)
                - Nresample: int|None,              Number of resamples (default: None)
                - tag: str|None,                    A string tag (default: None)
        """
        if resample_type is None:
            new: Data = Data.__new__(Data)
            new.tag = tag
            new.resample_type = None
            new._rspl = None
            new._rwf_rspl = None
            new.Nresample = None
            new.Ndata = Ndata
            new.locked_mean = locked_mean
            new.blocksize = None
            if shape is None:
                new._gvar = gv.gvar(1.0, 0.0)
            else:
                new._gvar = gv.gvar(np.ones(shape), np.zeros(shape))
            new._mean = gv.mean(new._gvar)
            return new

        if Nresample is not None:
            pass
        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            Nresample = Ndata
        else:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Nresample is None:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Ndata is None and resample_type == "jkn":
            Ndata = Nresample

        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = resample_type
        new.Ndata = Ndata
        new.Nresample = Nresample
        new.locked_mean = locked_mean
        new._gvar = None
        new.blocksize = None

        if shape is None:
            new._rspl = np.ones((Nresample,))
            new._mean = 1
        else:
            new._rspl = np.ones((Nresample, *shape))
            new._mean = np.ones((*shape,))

        return new

    @staticmethod
    def empty(resample_type: str | None, shape: tuple[int] | None = None, Ndata: int | None = None,
              Nresample: int | None = None, tag: str | None = None, locked_mean: bool = False,
              **kwargs) -> 'Data':
        r"""
            param:
                - resample_type: str | None,        'bst', 'jkn', or None for gvar mode
                - shape: tuple[int] | None,         shape of the observable (default: None = scalar)
                - Ndata: int|None,                  Number of raw data points (default: None)
                - Nresample: int|None,              Number of resamples (default: None)
                - tag: str|None,                    A string tag (default: None)

            Note: In gvar mode this is equivalent to Data.zeros (gvar has no 'empty' notion).
        """
        if resample_type is None:
            return Data.zeros(resample_type=None, shape=shape, Ndata=Ndata, tag=tag, locked_mean=locked_mean)

        if Nresample is not None:
            pass
        elif Nresample is None and Ndata is not None and resample_type == 'jkn':
            Nresample = Ndata
        else:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Nresample is None:
            raise ValueError(
                f"Couldn't deduce Nresample with: {resample_type=}, {shape=}, {Ndata=}, {Nresample=}, {tag=}")

        if Ndata is None and resample_type == "jkn":
            Ndata = Nresample

        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = resample_type
        new.Ndata = Ndata
        new.Nresample = Nresample
        new.locked_mean = locked_mean
        new._gvar = None
        new.blocksize = None

        if shape is None:
            new._rspl = np.empty((Nresample,), **kwargs)
            new._mean = None
        else:
            new._rspl = np.empty((Nresample, *shape), **kwargs)
            new._mean = np.empty((*shape,), **kwargs)

        return new

    @staticmethod
    def zeros_like(other: 'Data', copy_rwf: bool = True, locked_mean: bool = False) -> 'Data':
        r"""
            Create a Data instance filled with zeros and all other properties deduced from other.
            param:
                - other: Data,      Source Data instance
                - copy_rwf: bool,   specify if rwf should be copied (default: True)
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.zeros_like requires other Data object but got {type(other)}")

        new: Data = Data.__new__(Data)
        new.tag = other.tag
        new.resample_type = other.resample_type
        new.Ndata = other.Ndata
        new.Nresample = other.Nresample
        new.locked_mean = locked_mean
        new.blocksize = None

        if other.resample_type is None:
            # gvar mode
            new._rspl = None
            new._rwf_rspl = None
            if isinstance(other._gvar, np.ndarray):
                new._gvar = gv.gvar(np.zeros_like(gv.mean(other._gvar)), np.zeros_like(gv.sdev(other._gvar)))
            else:
                new._gvar = gv.gvar(0.0, 0.0)
            new._mean = gv.mean(new._gvar)
        else:
            new._gvar = None
            if other._rwf_rspl is None:
                new._rwf_rspl = None
            elif copy_rwf:
                new._rwf_rspl = other._rwf_rspl.copy()
            else:
                new._rwf_rspl = None

            new._rspl = np.zeros_like(other._rspl)
            if other._rspl.ndim > 1:
                new._mean = np.zeros_like(other._mean)
            else:
                new._mean = 0

        return new

    @staticmethod
    def empty_like(other: 'Data', copy_rwf: bool = True, locked_mean: bool = False) -> 'Data':
        r"""
            Create a Data instance with uninitialised storage and all other properties deduced from other.
            In gvar mode, equivalent to zeros_like.
            param:
                - other: Data,      Source Data instance
                - copy_rwf: bool,   specify if rwf should be copied (default: True)
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.empty_like requires other Data object but got {type(other)}")

        if other.resample_type is None:
            return Data.zeros_like(other, copy_rwf=copy_rwf, locked_mean=locked_mean)

        new: Data = Data.__new__(Data)
        new.tag = other.tag
        new.resample_type = other.resample_type
        new.Ndata = other.Ndata
        new.Nresample = other.Nresample
        new.locked_mean = locked_mean
        new._gvar = None
        new.blocksize = None

        if other._rwf_rspl is None:
            new._rwf_rspl = None
        elif copy_rwf:
            new._rwf_rspl = other._rwf_rspl.copy()
        else:
            new._rwf_rspl = np.empty_like(other._rwf_rspl)

        new._rspl = np.empty_like(other._rspl)
        if other._rspl.ndim > 1:
            new._mean = np.empty_like(other._mean)
        else:
            new._mean = None

        return new

    @staticmethod
    def full_like(other: 'Data', value: Number, copy_rwf: bool = True, locked_mean: bool = False) -> 'Data':
        r"""
            Create a Data instance filled with value, with all other properties deduced from other.
            param:
                - other: Data,      Source Data instance
                - value: Number,    Fill value
                - copy_rwf: bool,   specify if rwf should be copied (default: True)
        """
        if not isinstance(other, Data):
            raise RuntimeError(f"Data.full_like requires other Data object but got {type(other)}")

        new: Data = Data.__new__(Data)
        new.tag = other.tag
        new.resample_type = other.resample_type
        new.Ndata = other.Ndata
        new.Nresample = other.Nresample
        new.locked_mean = locked_mean
        new.blocksize = None

        if other.resample_type is None:
            new._rspl = None
            new._rwf_rspl = None
            if isinstance(other._gvar, np.ndarray):
                new._gvar = gv.gvar(np.full_like(gv.mean(other._gvar), value),
                                    np.zeros_like(gv.sdev(other._gvar)))
            else:
                new._gvar = gv.gvar(float(value), 0.0)
            new._mean = gv.mean(new._gvar)

        else:
            new._gvar = None
            if other._rwf_rspl is None:
                new._rwf_rspl = None
            elif copy_rwf:
                new._rwf_rspl = other._rwf_rspl.copy()
            else:
                new._rwf_rspl = np.ones_like(other._rwf_rspl)

            new._rspl = np.full_like(other._rspl, value)
            if other._rspl.ndim > 1:
                new._mean = np.full_like(other._mean, value)
            else:
                new._mean = value

        return new

    def copy(self, deepcopy: bool = True) -> 'Data':
        r"""
            param:
                - deepcopy: bool,   specify if the copy should be deep (True) or shallow (False) (default = True)
        """
        new: Data = Data.__new__(Data)
        new.tag = self.tag
        new.resample_type = self.resample_type
        new.Ndata = self.Ndata
        new.Nresample = self.Nresample
        new.blocksize = self.blocksize
        new.locked_mean = self.locked_mean

        if self.resample_type is None:
            # gvar mode — np.copy keeps the same gvar primary variables (correlations preserved)
            new._rspl = None
            new._rwf_rspl = None
            new._gvar = np.copy(self._gvar) if isinstance(self._gvar, np.ndarray) else self._gvar
            new._mean = gv.mean(new._gvar)
        else:
            new._gvar = None
            if deepcopy:
                new._rspl = self._rspl.copy()
                new._mean = self._mean.copy() if isinstance(self._mean, np.ndarray) else self._mean
                new._rwf_rspl = self._rwf_rspl.copy() if self._rwf_rspl is not None else None
            else:
                raise NotImplementedError

        return new

    # =================================================================================================================
    # Resample techniques
    # =================================================================================================================

    @staticmethod
    def blocking(data: np.ndarray, blocksize: int, rwf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        Ncfg: int = data.shape[0]
        Nblock: int = Ncfg // blocksize
        shape = data.shape[1:]
        blocked_data = np.zeros((Nblock, *shape))

        if rwf is None:
            for k in range(Nblock):
                if k < Nblock - 1:
                    blocked_data[k] = np.mean(data[k * blocksize:(k + 1) * blocksize, ...], axis=0)
                else:
                    blocked_data[k] = np.mean(data[k * blocksize:, ...], axis=0)
            return blocked_data, None
        else:
            data = data * rwf[:, *(np.newaxis,) * len(shape)]
            blocked_rwf = np.zeros((Nblock,))
            for k in range(Nblock):
                if k < Nblock - 1:
                    blocked_data[k] = np.mean(data[k * blocksize:(k + 1) * blocksize, ...], axis=0)
                    blocked_rwf[k]  = np.mean(rwf[k * blocksize:(k + 1) * blocksize], axis=0)
                else:
                    blocked_data[k] = np.mean(data[k * blocksize:, ...], axis=0)
                    blocked_rwf[k]  = np.mean(rwf[k * blocksize:], axis=0)
            return blocked_data, blocked_rwf

    @staticmethod
    def jackknife(data: np.ndarray, rwf: np.ndarray | None = None, blocksize: int | None = None) -> np.ndarray:
        r"""
            param:
                - data: np.ndarray,     numpy array of the raw data which is jackknifed
                - rwf: np.ndarray|None  reweighting factors (default: None)
                - blocksize: int|None   block size for blocking before jackknife (default: None)
            Perform a leave-one-out jackknife on the data. Data axis = 0 is assumed.
        """
        if blocksize is not None:
            data, rwf = Data.blocking(data=data, blocksize=blocksize, rwf=rwf)
        elif rwf is not None:
            data = data * rwf[:, *((np.newaxis,) * (data.ndim - 1))]

        if rwf is None:
            data_sum: np.ndarray | Number = np.sum(data, axis=0)
            return (data_sum - data) / (data.shape[0] - 1)
        else:
            data_sum: np.ndarray | Number = np.sum(data, axis=0)
            return (data_sum - data) / (np.sum(rwf, axis=0) - rwf)[:, *((np.newaxis,) * (data.ndim - 1))]

    def __jackknife(self, data: np.ndarray, rwf: np.ndarray | None = None, blocksize: int | None = None) -> None:
        self._rspl = Data.jackknife(data=data, rwf=rwf, blocksize=blocksize)
        if rwf is not None:
            self._rwf_rspl = Data.jackknife(data=rwf, blocksize=blocksize)
        self.Nresample = self._rspl.shape[0]

    @staticmethod
    def bootstrap(data: np.ndarray, Nresample: int, rwf: np.ndarray | None = None,
                  blocksize: int | None = None,
                  method: Callable[[np.ndarray], np.ndarray] | None = None) -> np.ndarray:
        r"""
            param:
                - data: np.ndarray,      numpy array of the raw data which is bootstrapped
                - Nresample: int,        number of bootstrap resamples
                - rwf: np.ndarray|None,  reweighting factors (default: None)
                - method: callable,      method applied on each bootstrap sample (default: arithmetic mean)
        """
        _rng = rng()

        if method is None:
            method = lambda x: np.mean(x, axis=0)

        if blocksize is not None:
            data, rwf = Data.blocking(data=data, blocksize=blocksize, rwf=rwf)
        elif rwf is not None:
            data = data * rwf[:, *((np.newaxis,) * (data.ndim - 1))]

        Ndata: int = data.shape[0]

        if rwf is None:
            sample_idx: np.ndarray = _rng.integers(0, Ndata, size=Ndata)
            bst_tmp: np.ndarray = method(data[sample_idx])
            bst = np.empty((Nresample, *bst_tmp.shape), dtype=bst_tmp.dtype)
            bst[0] = bst_tmp
            for k in range(1, Nresample):
                sample_idx = _rng.integers(0, Ndata, size=Ndata)
                bst[k] = method(data[sample_idx])
        else:
            sample_idx: np.ndarray = _rng.integers(0, Ndata, size=Ndata)
            bst_tmp: np.ndarray = method(data[sample_idx]) / np.mean(rwf[sample_idx], axis=0)
            bst = np.empty((Nresample, *bst_tmp.shape), dtype=bst_tmp.dtype)
            bst[0] = bst_tmp
            for k in range(1, Nresample):
                sample_idx = _rng.integers(0, Ndata, size=Ndata)
                bst[k] = method(data[sample_idx]) / np.mean(rwf[sample_idx], axis=0)

        return bst

    def __bootstrap(self, data: np.ndarray, Nresample: int, rwf: np.ndarray | None = None,
                    blocksize: int | None = None) -> None:
        self._rspl = Data.bootstrap(data=data, Nresample=Nresample, rwf=rwf, blocksize=blocksize)
        if rwf is not None:
            self._rwf_rspl = Data.bootstrap(data=rwf, Nresample=Nresample, blocksize=blocksize)
        self.Nresample = Nresample
        self.blocksize = blocksize

    @staticmethod
    def pseudoBootstrap(mean: float, sdev: float, Nresample: int, Ndata: int | None = None,
                        tag: str | None = None) -> 'Data':
        r"""
            param:
                - mean: float,      mean (location) of normal distribution
                - sdev: float,      standard deviation of normal distribution
                - Nresample: int,   Number of resamples
                - tag: str|None,    A string tag (default: None)
        """
        new: Data = Data.__new__(Data)
        new.tag = tag
        new.resample_type = "bst"
        new.Ndata = Ndata
        new.Nresample = Nresample
        new._rspl = new.get_rng().normal(mean, sdev, size=(Nresample,))
        new._mean = mean
        new._gvar = None
        new.blocksize = None

        return new

    @staticmethod
    def get_rng() -> np.random.Generator:
        return rng()

    @property
    def bootstrap_sample_ids(self) -> np.ndarray:
        if self.Ndata is None:
            raise RuntimeError(
                "Getting bootstrap sample ids requires self.Ndata which is not set.")
        _rng = rng()
        return _rng.integers(0, self.Ndata, size=(self.Nresample, self.Ndata))

    # =================================================================================================================
    # Internal checks and verification
    # =================================================================================================================

    def __check_other(self, other: 'Data | np.ndarray | Number') -> None:
        matches_flag: bool = True

        matches_flag = isinstance(other, (Data, np.ndarray, Number))
        if not matches_flag:
            raise ValueError(f"other is expected to be of type Data, np.ndarray, or Number but is: {type(other)}")

        if isinstance(other, Data):
            # Modes must match
            if self.resample_type is None and other.resample_type is not None:
                raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
            if self.resample_type is not None and other.resample_type is None:
                raise ValueError("Cannot combine resample-mode Data with gvar-mode Data")

            # For resample mode, check Nresample
            if self.resample_type is not None:
                matches_flag = self.Nresample == other.Nresample
                if not matches_flag:
                    raise ValueError(
                        f"other is expected to have same Nresample ({self.Nresample}) but has: {other.Nresample}")

        elif isinstance(other, np.ndarray):
            if self.resample_type is not None:
                matches_flag = is_broadcastable(self._rspl, other)
                if not matches_flag:
                    raise ValueError(
                        f"other is expected to have broadcastable shape ({self._rspl.shape}) but has: {other.shape}")

        elif isinstance(other, Number):
            pass

    # =================================================================================================================
    # Arithmetic overloads
    # =================================================================================================================

    def __add__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(self._gvar + other._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(self._gvar + other, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl + other._rspl,
                mean=self._mean + other._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl + other,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __radd__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(other._gvar + self._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(other + self._gvar, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other._rspl + self._rspl,
                mean=other._mean + self._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other + self._rspl,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __iadd__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                self._gvar = self._gvar + other._gvar
            elif isinstance(other, (np.ndarray, Number)):
                self._gvar = self._gvar + other
            else:
                raise NotImplementedError
            self._mean = gv.mean(self._gvar)
            return self

        self.__check_other(other)
        if isinstance(other, Data):
            self._rspl += other._rspl
            self._mean += other._mean
        elif isinstance(other, (np.ndarray, Number)):
            self._rspl += other
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise NotImplementedError
        return self

    def __sub__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(self._gvar - other._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(self._gvar - other, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl - other._rspl,
                mean=self._mean - other._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl - other,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __rsub__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(other._gvar - self._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(other - self._gvar, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other._rspl - self._rspl,
                mean=other._mean - self._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other - self._rspl,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __isub__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                self._gvar = self._gvar - other._gvar
            elif isinstance(other, (np.ndarray, Number)):
                self._gvar = self._gvar - other
            else:
                raise NotImplementedError
            self._mean = gv.mean(self._gvar)
            return self

        self.__check_other(other)
        if isinstance(other, Data):
            self._rspl -= other._rspl
            self._mean -= other._mean
        elif isinstance(other, (np.ndarray, Number)):
            self._rspl -= other
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise NotImplementedError
        return self

    def __mul__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(self._gvar * other._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(self._gvar * other, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl * other._rspl,
                mean=self._mean * other._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl * other,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __rmul__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(other._gvar * self._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(other * self._gvar, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other._rspl * self._rspl,
                mean=other._mean * self._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other * self._rspl,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __imul__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                self._gvar = self._gvar * other._gvar
            elif isinstance(other, (np.ndarray, Number)):
                self._gvar = self._gvar * other
            else:
                raise NotImplementedError
            self._mean = gv.mean(self._gvar)
            return self

        self.__check_other(other)
        if isinstance(other, Data):
            self._rspl *= other._rspl
            self._mean *= other._mean
        elif isinstance(other, (np.ndarray, Number)):
            self._rspl *= other
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise NotImplementedError
        return self

    def __truediv__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(self._gvar / other._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(self._gvar / other, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl / other._rspl,
                mean=self._mean / other._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl / other,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __rtruediv__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(other._gvar / self._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(other / self._gvar, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other._rspl / self._rspl,
                mean=other._mean / self._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=other / self._rspl,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __itruediv__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                self._gvar = self._gvar / other._gvar
            elif isinstance(other, (np.ndarray, Number)):
                self._gvar = self._gvar / other
            else:
                raise NotImplementedError
            self._mean = gv.mean(self._gvar)
            return self

        self.__check_other(other)
        if isinstance(other, Data):
            self._rspl /= other._rspl
            self._mean /= other._mean
        elif isinstance(other, (np.ndarray, Number)):
            self._rspl /= other
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise NotImplementedError
        return self

    def __pow__(self, other: 'Data | np.ndarray | Number') -> 'Data':
        if self.resample_type is None:
            if isinstance(other, Data):
                if other.resample_type is not None:
                    raise ValueError("Cannot combine gvar-mode Data with resample-mode Data")
                return Data.import_gvar(self._gvar ** other._gvar, Ndata=self.Ndata)
            elif isinstance(other, (np.ndarray, Number)):
                return Data.import_gvar(self._gvar ** other, Ndata=self.Ndata)
            raise NotImplementedError

        self.__check_other(other)
        if isinstance(other, Data):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl ** other._rspl,
                mean=self._mean ** other._mean,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        elif isinstance(other, (np.ndarray, Number)):
            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl ** other,
                rwf_rspl=self._rwf_rspl,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        raise NotImplementedError

    def __neg__(self) -> 'Data':
        if self.resample_type is None:
            return Data.import_gvar(-self._gvar, Ndata=self.Ndata)

        return Data.import_resamples(
            resample_type=self.resample_type,
            rspl=-self._rspl,
            mean=-self._mean,
            rwf_rspl=self._rwf_rspl,
            Ndata=self.Ndata,
            Nresample=self.Nresample,
        )

    # =================================================================================================================
    # Comparison overloads
    # In gvar mode comparisons act on the mean value only (no distribution available).
    # =================================================================================================================

    def __lt__(self, other: 'Data | np.ndarray | Real') -> np.ndarray:
        if self.resample_type is None:
            other_val = other.mean if isinstance(other, Data) else other
            return self.mean < other_val

        self.__check_other(other)
        if isinstance(other, Data):
            return self._rspl < other._rspl
        elif isinstance(other, (np.ndarray, Real)):
            return self._rspl < other
        raise NotImplementedError

    def __le__(self, other: 'Data | np.ndarray | Real') -> np.ndarray:
        if self.resample_type is None:
            other_val = other.mean if isinstance(other, Data) else other
            return self.mean <= other_val

        self.__check_other(other)
        if isinstance(other, Data):
            return self._rspl <= other._rspl
        elif isinstance(other, (np.ndarray, Real)):
            return self._rspl <= other
        raise NotImplementedError

    def __gt__(self, other: 'Data | np.ndarray | Real') -> np.ndarray:
        if self.resample_type is None:
            other_val = other.mean if isinstance(other, Data) else other
            return self.mean > other_val

        self.__check_other(other)
        if isinstance(other, Data):
            return self._rspl > other._rspl
        elif isinstance(other, (np.ndarray, Real)):
            return self._rspl > other
        raise NotImplementedError

    def __ge__(self, other: 'Data | np.ndarray | Real') -> np.ndarray:
        if self.resample_type is None:
            other_val = other.mean if isinstance(other, Data) else other
            return self.mean >= other_val

        self.__check_other(other)
        if isinstance(other, Data):
            return self._rspl >= other._rspl
        elif isinstance(other, (np.ndarray, Real)):
            return self._rspl >= other
        raise NotImplementedError

    # =================================================================================================================
    # Statistics
    # =================================================================================================================

    @property
    def rspl(self) -> np.ndarray:
        if self.resample_type is None:
            raise RuntimeError(
                "Data in gvar mode (resample_type=None) has no resamples. "
                "Access the underlying gvar objects via .gvar() instead.")
        if hasattr(self, "_rspl") and self._rspl is not None:
            return self._rspl
        raise RuntimeError("Data._rspl is not set.")

    @property
    def mean(self) -> np.ndarray | Number:
        if self.locked_mean:
            return self._mean

        if self.resample_type is None:
            # Always derive from _gvar to stay consistent with propagated operations
            self._mean = gv.mean(self._gvar)
            return self._mean

        self._mean = np.mean(self._rspl, axis=0)
        return self._mean

    @mean.setter
    def mean(self, value: np.ndarray | Number) -> None:
        if hasattr(self, "_mean"):
            if isinstance(self._mean, np.ndarray) and isinstance(value, np.ndarray):
                if self._mean.shape != value.shape:
                    raise ValueError("Replacing mean with array requires same shape")
                self._mean = value
            elif isinstance(self._mean, type(None)) and isinstance(value, np.ndarray):
                self._mean = value
            elif isinstance(self._mean, np.ndarray) and isinstance(value, Number):
                raise ValueError("Replacing mean (array) with value (number) is prohibited")
            elif isinstance(value, np.ndarray) and isinstance(self._mean, Number):
                raise ValueError("Replacing mean (number) with value (array) is prohibited")
            elif isinstance(self._mean, (Number, type(None))) and isinstance(value, Number):
                self._mean = value
            else:
                raise ValueError(f"Replacing mean ({type(self._mean)}) with value ({type(value)}) is prohibited")
        else:
            if isinstance(value, np.ndarray):
                if self._rspl.shape[1:] != value.shape:
                    raise ValueError("Setting mean with array requires matching shape with resamples")
                self._mean = value
            elif isinstance(value, Number):
                if self._rspl.ndim != 1:
                    raise ValueError("Setting mean with array requires matching shape with resamples")
                self._mean = value
            else:
                raise ValueError(f"Setting mean with value ({type(value)}) is prohibited")

        if self.resample_type is None:
            self._gvar = gv.gvar(self._mean, self.serr)

    @property
    def serr(self) -> np.ndarray | Number:
        if self.resample_type is None:
            return gv.sdev(self._gvar)

        if self.resample_type == 'jkn':
            self._serr = np.sqrt(self.Nresample - 1) * np.std(self._rspl, axis=0)
        elif self.resample_type == 'bst':
            self._serr = np.std(self._rspl, axis=0, ddof=1)
        else:
            raise NotImplementedError(f"serr not implemented for resample_type={self.resample_type}")
        return self._serr

    @property
    def cov(self) -> np.ndarray:
        if self.resample_type is None:
            if not isinstance(self._gvar, np.ndarray) or self._gvar.ndim != 1:
                raise RuntimeError(
                    "Covariance requires a 1D array of gvar objects, "
                    f"but _gvar has type {type(self._gvar)} / shape {getattr(self._gvar, 'shape', '(scalar)')}.")
            cov_mat = gv.evalcov(self._gvar)
            # Check for actual cross-covariance (off-diagonal entries)
            off_diag = cov_mat - np.diag(np.diag(cov_mat))
            if not np.any(off_diag):
                raise RuntimeError(
                    "gvar objects carry no cross-covariance information (all off-diagonal elements are zero). "
                    "Operations that build correlations (e.g. combining correlated Data) will populate the "
                    "off-diagonal. Use .serr for individual standard deviations.")
            return cov_mat

        if len(self._rspl.shape) != 2:
            raise RuntimeError(
                f"Covariance estimation is only implemented for Data of shape (N, Nobs), "
                f"with N being the number of resamples, but is: {self._rspl.shape}")

        if self.resample_type == 'jkn':
            self._cov = (self.Nresample - 1) * np.cov(self._rspl, rowvar=False, bias=True)
        elif self.resample_type == 'bst':
            self._cov = np.cov(self._rspl, rowvar=False, bias=False)
        else:
            raise NotImplementedError
        return self._cov

    def cov_uncertainty(self) -> np.ndarray:
        if self.resample_type is None:
            raise NotImplementedError("cov_uncertainty is not implemented for gvar mode.")
        if len(self._rspl.shape) != 2:
            raise RuntimeError(
                f"Correlation estimation is only implemented for Data of shape (N, Nobs), "
                f"but is: {self._rspl.shape}")
        cov_per_bst: np.ndarray = Data.bootstrap(
            self._rspl, Nresample=self.Nbst_inner, method=lambda x: np.cov(x, rowvar=False)
        )
        return np.std(cov_per_bst, axis=0, ddof=1)

    @property
    def cor(self) -> np.ndarray:
        if self.resample_type is None:
            if not isinstance(self._gvar, np.ndarray) or self._gvar.ndim != 1:
                raise RuntimeError(
                    "Correlation requires a 1D array of gvar objects.")
            cov_mat = gv.evalcov(self._gvar)
            off_diag = cov_mat - np.diag(np.diag(cov_mat))
            if not np.any(off_diag):
                raise RuntimeError(
                    "gvar objects carry no cross-covariance information; correlation matrix is trivially diagonal.")
            sdev = gv.sdev(self._gvar)
            outer = np.outer(sdev, sdev)
            return cov_mat / outer

        if len(self._rspl.shape) != 2:
            raise RuntimeError(
                f"Correlation estimation is only implemented for Data of shape (N, Nobs), "
                f"but is: {self._rspl.shape}")
        self._cor = np.corrcoef(self._rspl, rowvar=False)
        return self._cor

    def cor_uncertainty(self) -> np.ndarray:
        if self.resample_type is None:
            raise NotImplementedError("cor_uncertainty is not implemented for gvar mode.")
        if len(self._rspl.shape) != 2:
            raise RuntimeError(
                f"Correlation estimation is only implemented for Data of shape (N, Nobs), "
                f"but is: {self._rspl.shape}")
        cor_per_bst: np.ndarray = Data.bootstrap(
            self._rspl, Nresample=self.Nbst_inner, method=lambda x: np.corrcoef(x, rowvar=False)
        )
        return np.std(cor_per_bst, axis=0, ddof=1)

    @property
    def resample_serr(self) -> np.ndarray | Number:
        return self.serr

    @property
    def resample_cov(self) -> np.ndarray | Number:
        return self.cov

    @property
    def StN(self) -> np.ndarray | Number:
        mean_val = self.mean
        serr_val = self.serr

        if isinstance(mean_val, Number) or isinstance(serr_val, Number):
            if serr_val == 0:
                self._StN: np.ndarray | Number = np.inf
            else:
                self._StN = np.abs(mean_val) / serr_val
        else:
            if np.any(serr_val == 0):
                self._StN = np.full_like(mean_val, np.inf)
                mask = serr_val != 0
                self._StN[mask] = np.abs(mean_val[mask]) / serr_val[mask]
            else:
                self._StN = np.abs(mean_val) / serr_val

        return self._StN

    @property
    def serr_normal_approx(self):
        if self.resample_type is None:
            raise NotImplementedError("serr_normal_approx is not implemented for gvar mode.")
        median_data: np.ndarray = np.median(self._rspl, axis=0)
        abs_dev: np.ndarray = np.abs(self._rspl - median_data)
        median_abs_dev: np.ndarray = np.median(abs_dev, axis=0)
        if self.resample_type == "jkn":
            serr: np.ndarray = (median_abs_dev * 1.48260221850560186054) * np.sqrt(self.Nresample - 1)
        else:
            serr: np.ndarray = (median_abs_dev * 1.48260221850560186054)
        return serr

    def get_dist_data(self) -> np.ndarray:
        if self.resample_type is None:
            raise NotImplementedError(
                "get_dist_data is not implemented for gvar mode. "
                "Use .gvar() to obtain the gvar objects.")
        if self.resample_type == "bst":
            return self._rspl
        elif self.resample_type == "jkn":
            return (self._rspl - self.mean) * np.sqrt(self.Ndata - 1) + self.mean
        raise NotImplementedError

    # =================================================================================================================
    # Representations
    # =================================================================================================================

    def __repr__(self):
        s: str = "Data"

        if self.tag is not None:
            s += f"({self.tag})"

        if self.resample_type is None:
            s += "[gvar"
        else:
            s += f"[{self.resample_type}"

        if self.locked_mean:
            s += "-locked mean"

        if self.resample_type is None:
            # gvar mode: _rspl is intentionally None, not "unset"
            g = self._gvar
            if g is None:
                s += ", unset"
            elif isinstance(g, np.ndarray):
                s += f", shape={g.shape}"
            else:
                s += ", scalar"
        else:
            if self._rspl is None or self.Nresample is None:
                s += ", unset"
            else:
                s += f", Nresample={self.Nresample}"
                if self.shape:
                    s += f", shape={self.shape}"

        s += "]"
        return s

    def gvar(self, correlated: bool = False) -> Any:
        """
        Return the underlying gvar objects.

        In gvar mode returns self._gvar directly (correlations intact).
        In resample mode constructs gvar objects from the estimated mean and error/covariance.

        param:
            - correlated: bool,  if True (resample mode only) include the full covariance matrix (default: False)
        """
        if self.resample_type is None:
            return self._gvar

        if correlated:
            return gv.gvar(self.mean, self.cov)
        else:
            return gv.gvar(self.mean, self.serr)

    def to_dict(self) -> dict[str, np.ndarray | Number]:
        return {
            "est": self.mean,
            "err": self.serr,
            "res": None if self.resample_type is None else self.rspl,
        }

    # =================================================================================================================
    # hdf5 (de-)serialization
    # =================================================================================================================

    def serialize(self, h5f: h5.Group, node: str | None = None) -> None:
        if node is None:
            grp = h5f
        else:
            grp = h5f.create_group(node)

        if self.resample_type is None:
            # gvar mode: use gv.dumps to preserve all correlations
            grp.create_dataset("mode", data="gvar")
            grp.create_dataset("mean", data=self._mean)
            grp.create_dataset("gvar", data=np.bytes_(gv.dumps(self._gvar)))
        else:
            grp.create_dataset("mode", data="resample")
            grp.create_dataset("resample_type", data=self.resample_type)
            grp.create_dataset("mean", data=self._mean)
            grp.create_dataset("resamples", data=self._rspl)
            grp.create_dataset("Nresample", data=self.Nresample)
            if self._rwf_rspl is not None:
                grp.create_dataset("rwf_resamples", data=self._rwf_rspl)

        if self.Ndata is not None:
            grp.create_dataset("Ndata", data=self.Ndata)
        if self.blocksize is not None:
            grp.create_dataset("blocksize", data=self.blocksize)
        if self.tag is not None:
            grp.create_dataset("tag", data=self.tag)
        if self.Nbst_inner != Data.Nbst_inner:
            grp.create_dataset("Nbst_inner", data=self.Nbst_inner)


        grp.create_dataset("locked_mean", data = self.locked_mean)

    @staticmethod
    def deserialize(h5f: h5.Group, node: str | None = None) -> 'Data':
        if node is None:
            grp = h5f
        else:
            grp = h5f[node]

        new: Data = Data.__new__(Data)

        if "blocksize" in grp:
            new.blocksize = grp["blocksize"][()]
        else:
            new.blocksize = None

        if "locked_mean" in grp:
            new.locked_mean = grp["locked_mean"][()]
        else:
            new.locked_mean = False

        # allow backwards compatibility:
        if "mode" in grp:
            mode = grp["mode"][()].decode('utf-8') if isinstance(grp["mode"][()], bytes) else grp["mode"][()]
        else:
            mode = "resample"

        if mode == "gvar":
            new.resample_type = None
            new._rspl = None
            new._rwf_rspl = None
            new.Nresample = None
            new._mean = grp["mean"][()]
            gvar_bytes = grp["gvar"][()]
            new._gvar = gv.loads(bytes(gvar_bytes))
            new._mean = gv.mean(new._gvar)
        else:
            new.resample_type = grp["resample_type"][()].decode('utf-8')
            new._mean = grp["mean"][()]
            new._rspl = grp["resamples"][()]
            new.Nresample = grp["Nresample"][()]
            new._gvar = None
            if "rwf_resamples" in grp:
                new._rwf_rspl = grp["rwf_resamples"][()]
            else:
                new._rwf_rspl = None

        new.Ndata = grp["Ndata"][()] if "Ndata" in grp else None
        new.tag = grp["tag"][()].decode('utf-8') if "tag" in grp else None
        new.Nbst_inner = grp["Nbst_inner"][()] if "Nbst_inner" in grp else Data.Nbst_inner

        return new

    # =================================================================================================================
    # Interoperability with numpy
    # =================================================================================================================

    def __getitem__(self, idx: Any) -> 'Data | np.ndarray | Number':
        if self.resample_type is None:
            # Index into the gvar array
            if isinstance(self._gvar, np.ndarray):
                if isinstance(idx, tuple):
                    result_gvar = self._gvar[*idx]
                else:
                    result_gvar = self._gvar[idx]
            else:
                raise IndexError("Cannot index scalar gvar Data")
            return Data.import_gvar(result_gvar, Ndata=self.Ndata)

        if isinstance(idx, tuple):
            if (any((item is None) for item in idx)) and not isinstance(self._mean, np.ndarray) and hasattr(self,
                                                                                                             "_mean"):
                mean_tmp = np.asarray(self._mean)[*idx]
            elif hasattr(self, "_mean"):
                mean_tmp = self._mean[*idx]
            else:
                mean_tmp = None

            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl[:, *idx],
                mean=mean_tmp,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )
        else:
            if (idx is None or np.newaxis == idx) and not isinstance(self._mean, np.ndarray) and hasattr(self, "_mean"):
                mean_tmp = np.asarray([self._mean])
            elif hasattr(self, "_mean"):
                mean_tmp = self._mean[idx]
            else:
                mean_tmp = None

            return Data.import_resamples(
                resample_type=self.resample_type,
                rspl=self._rspl[:, idx],
                mean=mean_tmp,
                Ndata=self.Ndata,
                Nresample=self.Nresample,
            )

    def __setitem__(self, idx: Any, value: 'Data | np.ndarray | Number') -> None:
        if self.resample_type is None:
            if not isinstance(self._gvar, np.ndarray):
                raise ValueError("Cannot set items on scalar gvar Data")
            if isinstance(value, Data):
                if value.resample_type is not None:
                    raise ValueError("Cannot mix gvar mode with resample mode")
                self._gvar[idx] = value._gvar
            elif isinstance(value, (np.ndarray, Number)):
                # Assign as exact values (zero uncertainty)
                self._gvar[idx] = gv.gvar(value, np.zeros_like(np.asarray(value, dtype=float)))
            else:
                raise ValueError(f"Setting requires Data, np.ndarray, or Number but is: {type(value)}")
            self._mean = gv.mean(self._gvar)
            return

        if isinstance(value, Data):
            if self.Nresample != value.Nresample:
                raise ValueError(
                    f"Setting requires Data with same Nresample: {self.Nresample=} != {value.Nresample=}")
            if self.resample_type != value.resample_type:
                raise ValueError(
                    f"Setting requires same resample type: self({self.resample_type}) != value({value.resample_type})")
            self._rspl[:, idx] = value._rspl
            self._mean[idx] = value.mean
        elif isinstance(value, (np.ndarray, Number)):
            self._rspl[:, idx] = value
            self._mean = np.mean(self._rspl, axis=0)
        else:
            raise ValueError(f"Setting requires Data, np.ndarray, or Number but is: {type(value)}")

    def reshape(self, shape, *args, **kwargs):
        if self.resample_type is None:
            if isinstance(self._gvar, np.ndarray):
                self._gvar = self._gvar.reshape(shape, *args, **kwargs)
            else:
                raise ValueError("Cannot reshape scalar Data")
            self._mean = gv.mean(self._gvar)
            return

        self._rspl = self._rspl.reshape((self.Nresample, *shape), *args, **kwargs)
        self._mean = self._mean.reshape((*shape,), *args, **kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.resample_type is None:
            if isinstance(self._gvar, np.ndarray):
                return self._gvar.shape
            return tuple()

        if isinstance(self.mean, np.ndarray):
            return self.mean.shape
        return tuple()

    @property
    def ndim(self) -> int:
        if self.resample_type is None:
            if isinstance(self._gvar, np.ndarray):
                return self._gvar.ndim
            return 0
        return self.mean.ndim

    def __array__(self, copy=None, dtype=None):
        if self.resample_type is None:
            # Return an object array of gvar elements; numpy ufuncs will dispatch to gvar's own __array_ufunc__
            arr = np.asarray(self._gvar)
            if dtype is not None:
                return arr.astype(dtype)
            if copy:
                return arr.copy()
            return arr

        if dtype or copy:
            return np.asarray(self._rspl, copy=copy, dtype=dtype)
        return self._rspl

    def __array_ufunc__(self, ufunc, method, *inputs: tuple['Data | np.ndarray'], **kwargs: dict[str, Any]) -> Any:
        # Handle numpy ufuncs to preserve Data type
        if method != '__call__':
            raise NotImplementedError

        data_inputs = [inp for inp in inputs if isinstance(inp, Data)]

        # ---- gvar mode ----
        if any(inp.resample_type is None for inp in data_inputs):
            if not all(inp.resample_type is None for inp in data_inputs):
                raise RuntimeError("Cannot mix gvar-mode and resample-mode Data in a ufunc")

            # Extract _gvar arrays; gvar handles error propagation natively via its own __array_ufunc__
            args = [inp._gvar if isinstance(inp, Data) else inp for inp in inputs]
            result = ufunc(*args, **kwargs)

            ndata = next((inp.Ndata for inp in data_inputs if inp.Ndata is not None), None)
            if isinstance(result, np.ndarray) or hasattr(result, 'sdev'):
                return Data.import_gvar(result, Ndata=ndata)
            return result

        # ---- resample mode ----
        resample_types: np.ndarray = np.unique([inp.resample_type for inp in data_inputs])
        Nresamples: np.ndarray = np.unique([inp.Nresample for inp in data_inputs])
        Ndatas: np.ndarray = np.unique([inp.Ndata for inp in data_inputs if inp.Ndata is not None])
        rwf_candidates = [inp._rwf_rspl for inp in data_inputs if inp._rwf_rspl is not None]

        if len(resample_types) != 1:
            raise RuntimeError(f"All resample_types must be the same, but found: {resample_types}")
        resample_type: str = resample_types[0]

        if len(Nresamples) != 1:
            raise RuntimeError(f"All Nresamples must be the same, but found: {Nresamples}")
        Nresample: int = Nresamples[0]

        if len(Ndatas) != 1:
            raise RuntimeError(f"All Ndatas must be the same, but found: {Ndatas}")
        Ndata: int = Ndatas[0]

        if len(rwf_candidates) == 0:
            rwf_rspl = None
        else:
            # all must be identical — check pairwise against the first
            for rwf in rwf_candidates[1:]:
                if not np.array_equal(rwf, rwf_candidates[0]):
                    raise RuntimeError(
                        "Cannot combine Data objects with different rwf_rspl arrays in a ufunc")
            rwf_rspl = rwf_candidates[0]

        args = [inp._rspl if isinstance(inp, Data) else inp for inp in inputs]
        result: np.ndarray = ufunc(*args, **kwargs)

        if isinstance(result, np.ndarray):
            return Data.import_resamples(
                resample_type=resample_type,
                rspl=result,
                # No information on _rwf_rspl provided,
                rwf_rspl=rwf_rspl,
                Ndata=Ndata,
                Nresample=Nresample,
            )
        elif isinstance(result, Data):
            return result
        raise NotImplementedError(
            f"Dispatch of numpy ufunc not successful: result is not np.ndarray but: {type(result)}")

    def __array_function__(self, func, types, args: tuple[Any, ...], kwargs: dict[str, Any]) -> 'Data | np.ndarray | Number':
        """
        Implements interoperability with a greater numpy ecosystem.
        https://numpy.org/neps/nep-0018-array-function-protocol.html
        """
        if func in HANDLED_FUNCTIONS_DATA:
            return HANDLED_FUNCTIONS_DATA[func](*args, **kwargs)

        if not all(issubclass(t, (np.ndarray, Data)) for t in types):
            raise NotImplementedError

        # Collect metadata while unwrapping
        resample_types: list = []
        Nresamples: list = []
        Ndatas: list = []
        gvar_modes: list = []

        def unwrap(x: Any) -> Any:
            if isinstance(x, Data):
                gvar_modes.append(x.resample_type is None)
                if x.resample_type is None:
                    return x._gvar
                else:
                    resample_types.append(x.resample_type)
                    Nresamples.append(x.Nresample)
                    Ndatas.append(x.Ndata)
                    return x._rspl
            elif isinstance(x, (tuple, list)):
                return type(x)(unwrap(i) for i in x)
            elif isinstance(x, dict):
                return {k: unwrap(v) for k, v in x.items()}
            return x

        unwrapped_args = unwrap(args)
        unwrapped_kwargs = unwrap(kwargs)

        # ---- gvar mode ----
        if gvar_modes and all(gvar_modes):
            ndata = next((a.Ndata for a in args if isinstance(a, Data) and a.Ndata is not None), None)
            result = func(*unwrapped_args, **unwrapped_kwargs)
            if isinstance(result, np.ndarray) or hasattr(result, 'sdev'):
                return Data.import_gvar(result, Ndata=ndata)
            return result

        if len(gvar_modes) > 1 and not all(gvar_modes):
            print(gvar_modes)
            raise RuntimeError("Cannot mix gvar-mode and resample-mode Data in a numpy function")

        # ---- resample mode ----
        resample_types = list(np.unique(resample_types))
        Nresamples = list(np.unique(Nresamples))

        if len(resample_types) != 1:
            raise RuntimeError(f"All resample_types must be the same, but found: {resample_types}")
        resample_type: str = resample_types[0]

        if len(Nresamples) != 1:
            raise RuntimeError(f"All Nresamples must be the same, but found: {Nresamples}")
        Nresample: int = Nresamples[0]

        result: np.ndarray = func(*unwrapped_args, **unwrapped_kwargs)

        if isinstance(result, np.ndarray):
            return Data.import_resamples(
                resample_type=resample_type,
                rspl=result,
                Nresample=Nresample,
            )
        if isinstance(result, Number):
            return result
        raise NotImplementedError(
            f"Dispatch of numpy function not successful: result is not np.ndarray but: {type(result)}")


# =====================================================================================================================
# Numpy exceptions
# These functions have a specialised behaviour.
# =====================================================================================================================

@implements(np.mean)
def mean(a: Data, axis=None, dtype=None, out=None, keepdims=_NoValue, *, where=_NoValue) -> 'Data | np.ndarray | Number':
    if not isinstance(a, Data):
        raise ValueError(f"a is expected to be of type Data but is: {type(a)}")

    if a.resample_type is None:
        # gvar mode: apply mean to gvar array; gvar propagates errors through np.mean
        if isinstance(a._gvar, np.ndarray):
            result = np.mean(a._gvar, axis=axis)
        else:
            result = a._gvar  # scalar

        # If we averaged away all axes the result is a gvar scalar — still wrap it
        if isinstance(result, np.ndarray) or hasattr(result, 'sdev'):
            return Data.import_gvar(result, Ndata=a.Ndata)
        return result

    out_array: np.ndarray = np.mean(a._rspl, axis=axis, dtype=dtype, out=out,
                                    keepdims=keepdims, where=where)

    # axis==0 is the resample axis; averaging over it yields a plain array
    if axis != 0:
        return Data.import_resamples(
            resample_type=a.resample_type,
            rspl=out_array,
            Ndata=a.Ndata,
            Nresample=a.Nresample,
        )
    return out_array
