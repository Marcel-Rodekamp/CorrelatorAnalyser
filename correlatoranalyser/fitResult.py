import numpy as np

import h5py

from dataclasses import dataclass

from dill import dumps, loads

from collections.abc import Callable
from typing import Any

# We provide an import from lsqfit thus require
import lsqfit
import gvar as gv 

# We provide an import from minuit thus require

from scipy.special import gammaincc

from .data import Data

from .prior import Prior


@dataclass
class FitResult:
    # ###########################################
    # Fit Model
    # ###########################################

    # the independent variables of the fit 
    abscissa: np.ndarray | Data | None = None

    # number of degrees of freedom
    dof: int | None = None

    # Functional form of the fit model
    fcn: Callable | None = None

    # specify if resamples are fitted 
    resample_type: str | None = None
    
    # number of resamples if provided
    Nresample: int | None = None

    # ###########################################
    # Central Value Fit Result
    # ###########################################

    # result parameters of the central value fit 
    params: dict[str, Data] | None = None

    # prios of the central value fit
    # only filled if actually used
    priors: dict[str,Prior] | None = None

    # resulting χ² of the central value fit
    chi2: float | Data | None = None

    # resulting p-value of the central value fit
    p_value: float | Data | None = None

    # resulting Akaike information criterion of the central value fit
    AIC: float | Data | None = None

    # determine if a small sample correction is being calculated in the AIC
    AIC_small_sample_correction: bool = True
    
    # ###########################################
    # Functionality
    # ###########################################

    @property
    def Ndata(self) -> int:
        if self.abscissa is None:
            raise RuntimeError("Can't deduce number of fit points (Ndata) if no abscissa is provided")
        
        return np.prod(self.abscissa.shape)

    @property
    def has_resamples(self) -> bool:
        r"""
            Check if the fits are done on resamples.
        """
        # if the number of bootstraps are set we assume that bootstrap fits are be performed
        return self.resample_type is not None

    def eval(self, abscissa: np.ndarray | Data | None = None) -> Data | np.ndarray:
        r"""
            @param abscissa: np.ndarray, a abscissa to evaluate the fit model. If None fit abscissa is used 
        """
        if abscissa is None:
            # if abscissa not provided we use a linear space with 5 interval length of the original fit interval
            abscissa = self.abscissa
        
        out = Data.zeros(
            resample_type=self.resample_type,
            shape = abscissa.shape,
            Nresample = self.Nresample,
        )

        out.mean = self.fcn( abscissa, {key: self.params[key].mean for key in self.params.keys()} )
        for nres in range(self.Nresample):
            out.rspl[nres] = self.fcn( abscissa, {key: self.params[key].rspl[nres] for key in self.params.keys()} )

        return out

    def __post_init__(self) -> None:
        r"""
            Thus function, set's up the various arrays for bootstrap fits
        """

        # check resample type

        # allow "None" string 
        if self.resample_type == "None":
            raise NotImplementedError(f"Currently, gaussian error propagation is not implemented use, resample_type = 'bst' or 'jkn'")
            self.resample_type = None

        
        # check that resample type is as expected
        if self.resample_type not in [None, "bst", "jkn"]:
            raise RuntimeError(f"Resample type {self.resample_type} not recognized. Expecting None (no resample), 'bst' (bootstrap) or 'jkn' (jackknife) ")
        
        # if no resamples have to be done, we won't need to set up their respective arrays
        if self.resample_type is None:
            return
        else:
            if self.Nresample is None:
                raise RuntimeError(f"Nresample not provided even though resample_type={self.resample_type}")

        self.chi2       = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample)
        self.p_value    = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample)
        self.AIC        = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample)
        self.priors     = {}

    def __repr__(self) -> str:
        r"""
            Create a representation of the fit results
            
            The presented fit parameters 
                - deduced from central value fit and resamples
            The fit statistics are deduced from central value fits
            TODO: Add the some fit results for bootstraps
        """

        # General information on the fit model
        rep = f"FitResult[ ({self.abscissa[0]},{self.abscissa[-1]}), Ndata={self.Ndata}, resample:{ self.resample_type if self.has_resamples else False }]:\n"

        # Fit statistics (of central value fit)
        rep+= f"  𝜒²/dof [dof] = {self.chi2.mean/self.dof:.3g} [{self.dof}]\n"
        rep+= f"  p-value = {self.p_value.mean:.3g} \n"
        rep+= f"  AIC = {self.AIC.mean:.3g} \n"
        
        # Fit parameters
        for key, p in self.params.items():
            if bool(self.priors):
                rep+= f"    - {key}: {p.gvar()}  [{self.priors[key].gvar()}]\n"
            else:
                rep+= f"    - {key}: {p.gvar()} \n"

        return rep

    def serialize(self, h5_handle: h5py.File, node: str | None = None) -> None:
        r"""
            @param h5_handle: h5py.File, file object pointing to an opened h5-file 
            @param node: str, path into the h5 file relative to the h5_handle

            Store the FitResult into a node of a h5 file. 
        """

        if node is None:
            grp = h5_handle
        else:
            grp = h5_handle.create_group(node)

        # Check that the fit result is actually computed
        if self.params is None:
            raise ValueError(
                f"Import data from a fit first before saving the data in an h5 file."
            )

        # abscissa: np.ndarray | Data 
        if isinstance(self.abscissa,Data):
            self.abscissa.serialize(grp,node="abscissa")
        else:
            grp.create_dataset("abscissa", data = self.abscissa)

        # Ndata: int == len(abscissa)
        grp.create_dataset("Ndata", data = self.Ndata)

        # dof: int | None = None 
        if self.dof is not None:
            grp.create_dataset("dof", data = self.dof)

        # fcn: Callable | None = None
        if self.fcn is not None:
            grp.create_dataset("fcn", data = dumps(self.fcn,0))

        # resample_type: str | None = None
        if self.resample_type is not None:
            grp.create_dataset("resample_type", data = self.resample_type)

        # Nresample: int | None = None
        if self.Nresample is not None:
            grp.create_dataset("Nresample", data = self.Nresample)

        # params: dict[str, Data] | None = None
        for key, param in self.params.items():
            param.serialize(grp, node=f"params/{key}")

        #prior: dict[str,Prior] | None = None
        if self.priors is not None:
            for key, prior in self.priors.items():
                prior.serialize(grp, node=f"priors/{key}")

        #chi2: float | Data | None = None
        if isinstance(self.chi2,Data):
            self.chi2.serialize(grp,node="chi2")
        elif self.chi2 is not None:
            grp.create_dataset("chi2", data = self.chi2)

        #p_value: float | Data | None = None
        if isinstance(self.p_value,Data):
            self.p_value.serialize(grp,node="p_value")
        elif self.p_value is not None:
            grp.create_dataset("p_value", data = self.p_value)

        #AIC: float | Data | None = None
        if isinstance(self.AIC,Data):
            self.AIC.serialize(grp,node="AIC")
        elif self.AIC is not None:
            grp.create_dataset("AIC", data = self.AIC)

        return

    @staticmethod
    def deserialize(h5_handle: h5py.File, node: str | None = None) -> "FitResult":
        r"""
            @param h5_handle: h5py.File, file object pointing to an opened h5-file 
            @param node: str, path into the h5 file relative to the h5_handle

            Deserialize a FitResult from an h5 file. This is the inverse operation to serialize
        """
        if node is None:
            grp = h5_handle
        else:
            grp = h5_handle[node]

        abscissa = grp[f"abscissa"][()]
        Ndata:int = grp[f"Ndata"][()]

        # check if h5file contains bootstrap data:
        if "Nresample" in grp:
            Nresample:int = grp[f"Nresample"][()]
            resample_type:str = grp[f"resample_type"][()]
            out = FitResult(abscissa=abscissa, Ndata=Ndata, resample_type=resample_type, Nresample=Nresample)
        else:
            out = FitResult(abscissa=abscissa, Ndata=Ndata)

        # abscissa: np.ndarray | Data 
        if "abscissa" in grp:
            if isinstance(grp["abscissa"], h5py.Dataset):
                setattr(out, "abscissa", grp["abscissa"][()])
            else:
                setattr(out, "abscissa", Data.deserialize(grp,node="abscissa"))

        # Ndata: int == len(abscissa)
        setattr(out, "Ndata", grp["Ndata"][()])

        # dof: int | None = None 
        if "dof" in grp:
            setattr(out, "dof", grp["dof"][()])

        # fcn: Callable | None = None
        if "fcn" in grp:
            setattr(out, "fcn", loads(grp["fcn"][()]))

        # Nresample: int | None = None
        if "Nresample" in grp:
            setattr(out, "Nresample", grp["Nresample"][()])

        # params: dict[str, Data] | None = None
        if "params" in grp:
            params = {}
            for key in grp["params"].keys():
                params[key] = Data.deserialize(grp, node=f"params/{key}")
            setattr(out, "params", params)

        #prior: dict[str,Prior] | None = None
        if "priors" in grp:
            priors = {}
            for key in grp["priors"].keys():
                priors[key] = Prior.deserialize(grp, node = f"priors/{key}")
            setattr(out, "priors", priors)

        #chi2: float | Data | None = None
        if "chi2" in grp:
            if isinstance(grp["chi2"], h5py.Dataset):
                setattr(out, "chi2", grp["chi2"][()])
            else:
                setattr(out, "chi2", Data.deserialize(grp,node="chi2"))           

        #p_value: float | Data | None = None
        if "p_value" in grp:
            if isinstance(grp["p_value"], h5py.Dataset):
                setattr(out, "p_value", grp["p_value"][()])
            else:
                setattr(out, "p_value", Data.deserialize(grp,node="p_value"))  

        #AIC: float | Data | None = None
        if "AIC" in grp:
            if isinstance(grp["AIC"], h5py.Dataset):
                setattr(out, "AIC", grp["AIC"][()])
            else:
                setattr(out, "AIC", Data.deserialize(grp,node="AIC"))           

        return out

    def calculate_AIC(self, chi2):
        r"""
            @param small_sample_correction: bool, flag to add/remove a small sample correction (default: False)

            Compute the Akaike information criterion for a fit result
            based on the chi^2 obtained from lsqfit.
            The form can be found in
                https://arxiv.org/abs/2305.19417
                https://arxiv.org/abs/2208.14983
                https://arxiv.org/abs/2008.01069
            equation 3 in the first:
                AIC = -2ln L^* + 2k - 2d_K 

            A small sample correction can be applied to both by seeting small_sample_correction 
            https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
                AICc = AIC + (2k^2 + 2k)/(d_K - k -1)

            Here we compare
                1. -2*ln(L^*) = chi^2
                2. k = number of parameters
                3. d_K = number of points
                5. Model parameter \Theta
                4. p_\Theta prior for parameter \Theta
        """
        if bool(self.params):
            Nparam: int = len(self.params.keys())
        else:
            raise RuntimeError("FitResult not initialized, can not determine number of parameters for calculating AIC")

        # start with the degree of freedom. 
        # TODO: This factor 2 is highly debated as it is very aggressive for many data sets
        #       We may want to come up with a way to allow a more flexible way of calculating
        #       the AIC.
        #       For reference see issue #8
        AIC: float = 2 * (Nparam-self.Ndata)

        if self.AIC_small_sample_correction and (self.Ndata - Nparam - 1) != 0:

            # An error is raised if the number of data points is too small
            if self.Ndata <= Nparam +1:
                raise RuntimeError(f"In order to use the AIC small sample correction the number of data points and parameters should be such that Ndata>Nparam+1, but instead they have the values Ndata={self.Ndata}, Nparam={Nparam}.")

            # This corrections is negligible if Ndata >> Nparam**2 and thus often very useful
            # it effectively favours models with less parameters
            AIC += (2*Nparam**2 + 2*Nparam)/(self.Ndata - Nparam - 1)

        AIC += chi2

        return AIC

    # ###########################################
    # Importers
    # ###########################################
    def import_from_lsqfit(self, nlf: lsqfit.nonlinear_fit, nres: None | int = None) -> None:
        """
            @param nlf: lsqfit.nonlinear_fit, lsqfit fit result. 
            @param nres: int, resample ID imports the fit result into the resample arrays at position nres. If none, the central value fit fields
                              are populated
            save the results from a lsqfit, if nres is given then save in corresponding row nres of the bootstrap parameters
        """
        # if the functional form hasn't been set we populate it here
        if self.fcn is None:
            self.fcn = nlf.fcn

        # if the degree of freedom hasn't been set we populate it here
        # Notice, for lsqfit if priors are used for each parameter then
        # dof = Ndata
        # if start values are used for each parameter then
        # dof = Ndata - Nparam
        if self.dof is None:
            self.dof = nlf.dof

        # if nres is provided populate the resample fields at position nres
        if nres is not None:
            # check that the dictionary is set and fillable
            if not bool(self.params):
                self.params = {}

            # loop over all fit parameter keys
            for key in nlf.p.keys():  
                # lsqfit parameter keys can come in the form
                # "param"
                # "log(param)"
                # if the latter is provided we want to remove the log part and explicitly exponentiate the result
                # to get "param":...
                if "log" in key:
                    # delete 'log(' and ')' from 'log(X)' from to obtain X
                    # X may be a string of arbitrary length
                    key_red = key[4:-1]  

                    # check if the resample array exists. If not set it
                    # dtype = object allows to store gvar.gvar instances 
                    if key_red not in self.params.keys():
                        self.params[key_red] = Data.empty( 
                            resample_type=self.resample_type, 
                            shape = None, #each parameter is a one-dimensional object hence no additional shape
                            Nresample=self.Nresample
                        )

                    # extract the parameter and exponentiate it
                    self.params[key_red].rspl[nres] = np.exp(gv.mean(nlf.p[key]))
                    # we only save central vlaue priors for now
                    # if nlf.prior is not None:
                    #     if key_red not in self.priors:
                    #         self.priors[key_red] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                    #     self.priors[key_red].rspl[nres] = Prior.import_from_lsqfit(nlf.prior)
                else:

                    # check if the resample array exists. If not set it
                    if key not in self.params.keys():
                        self.params[key] = Data.empty( 
                            resample_type=self.resample_type, 
                            shape = None, #each parameter is a one-d object hence no additional shape
                            Nresample=self.Nresample
                        )

                    # extract the parameter
                    self.params[key].rspl[nres] = gv.mean(nlf.p[key])

                    # if nlf.prior is not None:
                    #     if key not in self.priors:
                    #         self.priors[key] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                    #     self.priors[key].rspl[nres] = Prior.import_from_lsqfit(nlf.prior)

            # Extract fit statistics
            self.chi2.rspl[nres] = nlf.chi2
            self.p_value.rspl[nres] = nlf.Q
            self.AIC.rspl[nres] = self.calculate_AIC(self.chi2.rspl[nres])

        # central value fit (if nres is None)
        else:
            if not bool(self.params):
                self.params = {}

            # loop over all fit parameter keys
            for key in list(nlf.p.keys()):  # don't want to store the log value
                # lsqfit parameter keys can come in the form
                # "param"
                # "log(param)"
                # if the latter is provided we want to remove the log part and explicitly exponentiate the result
                # to get "param":...
                if "log" in key:
                    # delete 'log(' and ')' from 'log(X)' from to obtain X
                    # X may be a string of arbitrary length
                    key_red = key[4:-1]  

                    if key_red not in self.params.keys():
                        self.params[key_red] = Data.empty( 
                            resample_type=self.resample_type, 
                            shape = None, #each parameter is a one-dimensional object hence no additional shape
                            Nresample=self.Nresample
                        )

                    # extract the parameter and exponentiate it
                    self.params[key_red].mean = gv.mean(np.exp(nlf.p[key]))

                    if nlf.prior is not None:
                        # if key_red not in self.priors:
                        #     self.priors[key_red] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                        self.priors[key_red] = Prior.import_from_lsqfit(nlf.prior)[key]

                else:
                    if key not in self.params.keys():
                        self.params[key] = Data.empty( 
                            resample_type=self.resample_type, 
                            shape = None, #each parameter is a one-dimensional object hence no additional shape
                            Nresample=self.Nresample
                        )

                    # extract the parameter
                    self.params[key].mean = gv.mean(nlf.p[key])

                    if nlf.prior is not None:
                        # if key not in self.priors:
                        #     self.priors[key] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                        self.priors[key] = Prior.import_from_lsqfit(nlf.prior)[key]

            # Extract fit statistics
            self.chi2.mean = nlf.chi2
            self.p_value.mean = nlf.Q
            self.AIC.mean = self.calculate_AIC(nlf.chi2)

    def import_from_linear_regression(self, target_data, result_params, design_matrix, weight_matrix, parameter_names, nres = None):
        if not bool(self.params):
            self.params = {}

        # deduce if we have an intercept or not by checking how many parameters we have
        has_intercept = len(result_params) == 2

        if self.fcn is None:
            if has_intercept:
                self.fcn = lambda x,p: x*p[parameter_names[0]] + p[parameter_names[1]]
            else:
                self.fcn = lambda x,p: x*p[parameter_names[0]]

        if self.dof is None:
            self.dof = self.Ndata - len(result_params)

        if nres is not None:
            # check that the dictionary is set and fillable
            if not bool(self.params):
                self.params = {}

            # calculate Gaussian error propagation
            # cov = np.linalg.inv(design_matrix.T @ weight_matrix @ design_matrix)
            
            for key_id,key in enumerate(parameter_names):
                if key not in self.params.keys():
                    self.params[key] = Data.empty( 
                        resample_type = self.resample_type, 
                        shape         = None, #each parameter is a one-dimensional object hence no additional shape
                        Nresample     = self.Nresample
                    )
                    
                self.params[key].rspl[nres] = result_params[key_id]

            result = design_matrix @ result_params
            residuals = target_data - result

            self.chi2.rspl[nres] = residuals.T @ weight_matrix @ residuals # no priors in this fit
            self.p_value.rspl[nres] = gammaincc(self.dof/2, self.chi2.rspl[nres]/2)
            self.AIC.rspl[nres] = self.calculate_AIC( self.chi2.rspl[nres] )

            self.priors = None

        # central value fit
        else: 
            # calculate Gaussian error propagation
            # ToDo: Data is currently not set up to execute gaussian error prop (resample is requred)
            # cov = np.linalg.inv(design_matrix.T @ weight_matrix @ design_matrix)
            
            for key_id,key in enumerate(parameter_names):
                # if the parameter is without the log phrase we simple add it as parameter
                # check if the resample array exists. If not set it
                # dtype = object allows to store gvar.gvar instances 
                if key not in self.params.keys():
                    self.params[key] = Data.empty( 
                        resample_type = self.resample_type, 
                        shape         = None, #each parameter is a one-dimensional object hence no additional shape
                        Nresample     = self.Nresample
                    )
                    
                self.params[key].mean = result_params[key_id]

            result = design_matrix @ result_params
            residuals = target_data - result

            self.chi2.mean = residuals.T @ weight_matrix @ residuals # no priors in this fit
            self.p_value.mean = gammaincc(self.dof/2, self.chi2.mean/2)
            self.AIC.mean = self.calculate_AIC( self.chi2.mean )

            self.prior = None
        # end else

    def import_from_iminuit(self, 
        minuit: Any, 
        Ndata: int, 
        model: Callable, 
        prior: dict[str|Prior] | None, 
        nres: None | int = None
    ) -> None:
        """
            @param nlf: lsqfit.nonlinear_fit, lsqfit fit result. 
            @param nres: int, resample ID imports the fit result into the resample arrays at position nres. If none, the central value fit fields
                              are populated
            save the results from a lsqfit, if nres is given then save in corresponding row nres of the bootstrap parameters
        """
        # if the functional form hasn't been set we populate it here
        if self.fcn is None:
            self.fcn = model

        # if the degree of freedom hasn't been set we populate it here
        if self.dof is None:
            if prior is None:
                self.dof = Ndata - len(minuit.params)
            else:
                self.dof = Ndata - len(minuit.params) + len(prior.keys())
                

        # if nres is provided populate the resample fields at position nres
        if nres is not None:
            # check that the dictionary is set and fillable
            if not bool(self.params):
                self.params = {}

            # loop over all fit parameter keys
            for param in minuit.params:  
                key = param.name 

                # check if the resample array exists. If not set it
                if key not in self.params.keys():
                    self.params[key] = Data.empty( 
                        resample_type=self.resample_type, 
                        shape = None, #each parameter is a one-d object hence no additional shape
                        Nresample=self.Nresample
                    )

                # extract the parameter
                self.params[key].rspl[nres] = param.value

                # if prior is not None:
                #     if key not in self.priors:
                #         self.priors[key] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                #     self.priors[key].rspl[nres] = prior

            # Extract fit statistics
            self.chi2.rspl[nres] = minuit.fval
            self.p_value.rspl[nres] = gammaincc(self.dof/2, self.chi2.rspl[nres]/2)
            self.AIC.rspl[nres] = self.calculate_AIC( self.chi2.rspl[nres] )
            

        # central value fit (if nres is None)
        else:
            if not bool(self.params):
                self.params = {}

            # loop over all fit parameter keys
            for param in minuit.params:
                key = param.name 

                if key not in self.params.keys():
                    self.params[key] = Data.empty( 
                        resample_type=self.resample_type, 
                        shape = None, #each parameter is a one-dimensional object hence no additional shape
                        Nresample=self.Nresample
                    )

                # extract the parameter
                self.params[key].mean = param.value

                if prior is not None:
                    # if key not in self.priors:
                    #     self.priors[key] = Data.empty(resample_type = self.resample_type, shape=None, Nresample=self.Nresample, dtype=object)
                    self.priors[key] = prior[key]

            # Extract fit statistics
            self.chi2.mean = minuit.fval
            self.p_value.mean =  gammaincc(self.dof/2, self.chi2.mean/2)
            self.AIC.mean = self.calculate_AIC(self.chi2.mean)

# end of class: FitResult 

