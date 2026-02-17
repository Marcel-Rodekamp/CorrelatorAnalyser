import numpy as np

import gvar as gv

import h5py

from dataclasses import dataclass, field, fields

from pathlib import Path

from typing import Self, List, Dict

from .fitResult import FitResult

from .data import Data

import re

import warnings

from typing import Any, Dict

def _read_ds(ds: h5py.Dataset):
    """Read an h5py dataset, decoding strings when needed."""
    # h5py>=3: this returns str instead of bytes for string datasets
    try:
        return ds.asstr()[()]
    except Exception:
        data = ds[()]
        if isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8")
        if isinstance(data, np.ndarray) and data.dtype.kind in ("S", "O"):
            return np.array(
                [x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x for x in data],
                dtype=object,
            )
        return data


@dataclass
class FitState:
    
    # list of fit results: Defining the state of the fitter
    fit_results: List[FitResult] = field(default_factory=lambda: [])  
    
    # collection of keys in the state
    keys_all: list[str] = field(default_factory=list[str])

    def append(self, new_fit: FitResult) -> None:
        self.fit_results.append(new_fit)

        new_keys = list(new_fit.params.keys())
        self.keys_all += [key for key in new_keys if key not in self.keys_all]
        self.fit_results.sort(key=lambda x: x.AIC.mean)  # sort by AIC

        return

    def model_average(self, keys: str | list[str] = None) -> dict:
        N = len(self)

        def avg_single_key(key: str) -> None:
            param: Data = Data.zeros(
                resample_type = self.fit_results[0].resample_type,
                shape = (len(self),),
                Nresample = self.fit_results[0].Nresample,
            )
            AIC: Data = Data.zeros(
                resample_type = self.fit_results[0].resample_type,
                shape = (len(self),),
                Nresample = self.fit_results[0].Nresample,
            )

            ignored_fits = []
            for fit_id,fit in enumerate(self.fit_results):
                if key not in fit.params: 
                    ignored_fits.append(fit_id)
                    continue
                
                param[fit_id] = fit.params[key]
                AIC[fit_id] = fit.AIC
            
            if len(ignored_fits):
                mask = [ True if fit_id not in ignored_fits else False for fit_id in range(len(self)) ]
                param = param[mask]
                AIC = AIC[mask]

            # axis=0 == resample axis, axis = 1 == number fit axis 
            weights:Data = np.exp( -0.5*(AIC-np.min(AIC,axis=1)[None])) 
            weights /= np.sum(weights,axis=1)[None]

            modelAverage = np.average(
                param, weights=weights,
                axis=1
            )

            return modelAverage
        # end def avg_single_key

        # go through all parameters and do model averaging for each of them:
        if isinstance(keys, list):  # if argument was given
            out = {}
            for key in keys:

                if key not in self.keys_all:
                    raise KeyError(
                        f'The given key "{key}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                    )
                ma_param = avg_single_key(key=key)

                out[key] = ma_param
            return out

        elif isinstance(keys, str):  # if argument was given
            if keys not in self.keys_all:
                raise KeyError(
                    f'The given key "{keys}" is not a fit parameter, choose one parameter from {self.keys_all} for the model average.'
                )
            
            ma_param = avg_single_key(key=keys)

            return ma_param
        else:
            out = {}
            for key in self.keys_all:
                out[key] = avg_single_key(key=key)
            return out

    def __getitem__(self, index: int) -> FitResult:
        """method that gets a fit result based on the index in the fit_result list, list is sorted by AIC"""
        return self.fit_results[index]

    def __len__(self):
        """
        Returns the number of fits which are being tracked.
        """

        return len(self.fit_results)

    # dumps all information of the FitState in a h5 File
    def serialize_all(
        self, h5_file: h5py.File
    ) -> None:
        if not self.fit_results:
            raise Warning(f"Import FitResults first with {type(self).__name__}.append(new_fit : FitResult) before saving in an h5 file.")

        for field in fields(self):
            field_value = getattr(self, field.name)
            # print(field.name,field_value)
            # save the fit results using the serialize method from FitResult class
            if field.name == "fit_results":
                for i, fit in enumerate(field_value):
                    fit.serialize(h5_handle=h5_file, node=f"FitResults/Fit{i}")

            # save the keys
            elif field.name == "keys_all":
                h5_file.create_dataset(f"ModelAverage/KeysList", data=field_value)

        return

    def deserialize_all(self, h5_file: h5py.File):
        if "FitResults" in h5_file:
            fr_grp = h5_file["FitResults"]

            def _fit_idx(name: str) -> int:
                m = re.search(r"(\d+)$", name)
                return int(m.group(1)) if m else -1

            for node_name in sorted(fr_grp.keys(), key=_fit_idx):
                node_path = f"FitResults/{node_name}"
                self.fit_results.append(FitResult.deserialize(h5_handle=h5_file, node=node_path))
        else:
            raise RuntimeError("No 'FitResults' group found")

        # ---------- ModelAverage/KeysList ----------
        self.keys_all = []
        try:
            keys = _read_ds(h5_file["ModelAverage/KeysList"])
            if isinstance(keys, np.ndarray):
                self.keys_all = keys.tolist()
            elif np.isscalar(keys):
                self.keys_all = [keys.item() if hasattr(keys, "item") else keys]
            else:
                # iterable but not str/bytes -> list; str -> wrap
                self.keys_all = list(keys) if not isinstance(keys, (str, bytes)) else [keys]
        except KeyError:
            warnings.warn("No 'ModelAverage/KeysList' dataset found; leaving `keys_all` empty.", RuntimeWarning)

        # Optional: mirror your save-time guard as a load-time sanity note
        if not self.fit_results:
            warnings.warn(
                f"No FitResults loaded. Ensure the HDF5 file contains groups like 'FitResults/Fit0', 'FitResults/Fit1', ...",
                RuntimeWarning,
            )


