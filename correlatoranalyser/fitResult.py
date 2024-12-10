import numpy as np

import gvar as gv 

import h5py

from dataclasses import dataclass, fields

from dill import dumps, loads

from collections.abc import Callable

import lsqfit



@dataclass
class FitResult:
    """ToDo"""

    # ###########################################
    # Fit Model
    # ###########################################
    
    # start point of the fit range
    ts: int | np.ndarray

    # end point of the fit range
    te: int | np.ndarray

    # number of degrees of freedom
    dof: int | None = None

    # Functional form of the fit model
    fcn: Callable | None = None

    # ###########################################
    # Central Value Fit Result
    # ###########################################

    # result parameters of the central value fit 
    best_fit_param: gv.BufferDict | None = None

    # prios of the central value fit
    # only filled if actually used
    prior: gv.BufferDict | None = None

    # resulting χ² of the central value fit
    chi2: float | None = None

    # resulting χ² of the central value fit including priors
    aug_chi2: float | None = None

    # resulting Q-value of the central value fit
    Q_value: float | None = None

    # resulting Akaike information criterion of the central value fit
    AIC: float | None = None

    # resulting Akaike information critetion of the central value fit including priors
    aug_AIC: float | None = None
    
    # ###########################################
    # Resampled Fit Results
    # ###########################################

    # Number of resample (bootstrap, jackknife) fits
    Nres: int | None = None

    # Resample type: influence the way how uncertainties/confidence intervals are determined
    # assumed input:
    # None: no resample
    # "bst": bootstrap
    # "jkn": jackknife 
    resample_type: str | None = None

    # result parameter of the resampled fits
    best_fit_param_res: gv.BufferDict | None = None

    # priors of the resampled fits
    prior_res: gv.BufferDict | None = None

    # resulting χ² of the resampled fits
    chi2_res: np.ndarray | None = None

    # resulting χ² of the resampled fits including priors
    aug_chi2_res: np.ndarray | None = None

    # resulting Q-value of the resampled fits
    Q_value_res: np.ndarray | None = None

    # resulting Akaike information criterion of the resampled fits
    AIC_res: np.ndarray | None = None

    # resulting Akaike information critetion of the resampled fits including priors
    aug_AIC_res: np.ndarray | None = None
    



    # ###########################################
    # Functionality
    # ###########################################

    def has_resamples(self) -> bool:
        r"""
            Check if the fits are done on resamples.
        """
        # if the number of bootstraps are set we assume that bootstrap fits are be performed
        return self.resample_type is not None and bool(self.best_fit_param_res)

    def has_central_value(self) -> bool:
        r"""
            Check if the fit is done on the central value data.
        """
        return bool(self.best_fit_param)

    def result_params(self, key: str | None = None) -> dict:
        r"""
            @param key: either "est", "err" or "res"

            This function organizes the resulting parameters of the stored fit in a dictionary of the form
            {
              # the central value fit result or average over resamples
              "est": {"param1":float, "param2":float},  

              # the uncertainty from gaussian propagation or standard deviation over resamples
              "err": {"param1":float, "param2":float},
              
              # the results of the resample fits sorted by key then presented as an array over results
              "res": {"param1":np.array(Nres, dtype=float), "param2":np.array(Nres, dtype=float)}
            }

            if key is None, this full dictionary is returned -> {'est':..., 'err':..., 'res':...}
            if key is est, only the {...}['est'] is returned -> {'param': ..., ...}
            if key is err, only the {...}['err'] is returned -> {'param': ..., ...}
            if key is res, only the {...}['res'] is returned -> {'param': ..., ...}

        """
        # Check that the key takes one of the expected values
        if key not in [None, "est", "err", "res"]:
            raise ValueError(
                  "FitResult.result_params expects input None (entire dictionary),"
                + "'est' (estimate/central vlaue),"
                + "'err' (error/uncertainty) or 'res'(resample/bootstrap results or jackknife results)"
                +f"but got {key}"        
            )

        if self.has_central_value():
            result_params_dict = { "est": gv.mean( self.best_fit_param ) }
        else:
            result_params_dict = {}

        if self.has_resamples():
            result_params_dict["res"] = self.best_fit_param_res
            
            # Calculate errors differently for bootstrap or jackknifes
            if self.resample_type == 'bst':
                result_params_dict["err"] = {
                    key: np.std(
                            gv.mean(self.best_fit_param_res[key]), 
                            axis = 0
                         ) 
                    for key in self.best_fit_param_res.keys()  
                }
            elif self.resample_type == 'jkn': 
                result_params_dict["err"] = {
                    key: (self.Nres-1)/(self.Nres) * np.std(
                            gv.mean(self.best_fit_param_res[key]), 
                            axis = 0
                         ) 
                    for key in self.best_fit_param_res.keys()  
                }
            else:
                # this case is checked in __post_init__
                pass
            result_params_dict["est"] = {
                    key: np.mean(
                        gv.mean(self.best_fit_param_res[key]), 
                        axis = 0
                    ) 
                    for key in self.best_fit_param_res.keys()  
                }

        else:
            # For non-resampled fits we can use Gaussian error-propagation implemented via gvar
            # This also includes correlation as estimated
            result_params_dict["err"] = gv.sdev( self.best_fit_param )

        if key is None:
            return result_params_dict
        else:
            return result_params_dict[key]

    def eval(self, abscissa: np.ndarray | None= None) -> dict:
        r"""
            @param abscissa: np.ndarray, a abscissa to evaluate the fit model. If None automatically a linear-space  from ts to te is created with length 5*(te-ts).

            Evaluate the fit result function on the abscissa and return a dictionary of the form
            {
                # Central value fit result or average over resamples
                "est": np.array(fcn(abscissa).shape)
                
                # Confidence interval provided through Gaussian error-propagation of the uncertainty and correlation of the best-fit parameters or through resamples
                "err": np.array(fcn(abscissa).shape)
                
                # Model function evaluated on abscissa over all resamples
                "res": np.array((Nres, fcn(abscissa).shape) )
            }
                                
        """
        
        if abscissa is None:
            # if abscissa not provided we use a linear space with 5 interval length of the original fit interval
            abscissa = np.linspace( self.ts, self.te, 5*(self.te-self.ts) )
        
        # out dictionary which will contain 3 keys:
        # "est": central value fit result
        # "err": std confidence band either through bootstrap or Gaussian error-propagation
        # "res": fit result per bootstrap 
        out: dict = {}
        
        if self.has_central_value():
            gvar_eval: np.ndarray = self.fcn( abscissa, self.best_fit_param )
            out["est"]: np.ndarray = gv.mean(gvar_eval)

        if self.has_resamples():
            out["res"]:np.ndarray = np.zeros( (self.Nres, len(abscissa) ) )

            for nres in range(self.Nres):
                out["res"][nres] = gv.mean( # self.fcn return array of gvars, we take the central values
                    self.fcn( 
                        abscissa, 
                        # reorder the parameters to access the result on the current resample
                        { key: self.best_fit_param_res[key][nres] for key in self.best_fit_param_res.keys() } 
                    )
                )

            # Calculate errors differently for bootstrap or jackknifes
            if self.resample_type == 'bst':
                out["err"]: np.ndarray = np.std(out["res"], axis = 0) 
            elif self.resample_type == 'jkn': 
                out["err"]: np.ndarray = (self.Nres-1)/(self.Nres) * np.std(out["res"], axis = 0) 
            else:
                # this case is checked in __post_init__
                pass

            if not self.has_central_value():
                out["est"]: np.ndarray = np.mean(out["res"], axis = 0) 

        else:
            # For non-resampled fits we can use Gaussian error-propagation implemented via gvar
            # This also includes correlation as estimated
            out['err']: np.ndarray = gv.sdev( gvar_eval )

        return out

    def __post_init__(self) -> None:
        r"""
            Thus function, set's up the various arrays for bootstrap fits
        """

        # check resample type

        # allow "None" string 
        if self.resample_type == "None":
            self.resample_type = None
        
        # check that resample type is as expected
        if self.resample_type not in [None, "bst", "jkn"]:
            raise RuntimeError(f"Resample type {self.resample_type} not recognized. Expecting None (no resample), 'bst' (bootstrap) or 'jkn' (jackknife) ")
        
        # if no resamples have to be done, we won't need to set up their respective arrays
        if self.resample_type is None:
            return

        self.chi2_res       = np.empty(self.Nres)
        self.aug_chi2_res   = np.empty(self.Nres)
        self.Q_value_res    = np.empty(self.Nres)
        self.AIC_res        = np.empty(self.Nres)
        self.aug_AIC_res    = np.empty(self.Nres)
        self.prior_res      = np.empty(self.Nres, dtype=object)

    def __repr__(self) -> str:
        r"""
            Create a representation of the fit results
            
            The presented fit parameters 
                - deduced from central value fit and resamples
            The fit statistics are deduced from central value fits
            TODO: Add the some fit results for bootstraps
        """

        # General information on the fit model
        rep = f"FitResult[ ({self.ts},{self.te}), resample:{ self.resample_type if self.has_resamples() else False }]:\n"

        # Fit statistics
        if self.has_central_value():
            rep+= f"  𝜒²/dof [dof] = {self.chi2/self.dof:.3g} [{self.dof}]\n"
            rep+= f"  AIC = {self.AIC:.3g} \n"
        
        # Fit parameters
        fit_params = self.result_params()
        for key in fit_params['est'].keys():
            p = gv.gvar(fit_params['est'][key], fit_params['err'][key])
            if self.has_central_value():
                rep+= f"    - {key}: {p}  [{self.prior[key]}]\n"
            else:
                # Every bootstrap has it's own prior, we can't plot all of them here, hence we neglect this information
                rep+= f"    - {key}: {p}  \n"

        return rep

    def serialize(self, h5_handle: h5py.File, node: str) -> None:
        r"""
            @param h5_handle: h5py.File, file object pointing to an opened h5-file 
            @param node: str, path into the h5 file relative to the h5_handle

            Store the FitResult into a node of a h5 file. 
        """

        # Check that the fit result is actually computed
        if self.best_fit_param is None and self.best_fit_param_res is None:
            raise ValueError(
                f"Import data from a fit first before saving the data in an h5 file."
            )

        # Iterate over all fields defined in the beginning
        # This automatically extends if more fields are added in the future
        # Care has to be taken for special types that can not simply be dumped into h5 format. 
        # In that case it may help to pickle the object (use e.g. dill.dumps imported above)
        for field in fields(self):

            # this greps the field value of the current field
            field_value = getattr(self, field.name)
            
            if field_value is None:
            # None values can be skipped
                pass

            elif isinstance(field_value, dict) or isinstance(field_value, gv.BufferDict):
            # In case we use a dictionary store the value dictionary in a deeper node (nest the node)
            # {key: value} -> h5file[node/key] == value

                # iterate over all keys in the dictionary (depth of the dictionary is assumed to be one)
                for key, value in field_value.items():
                    # gvars loose coorelation when pickled. We follow this procedure and simply store 
                    # the standard deviation (uncertainty).
                    # TODO: Do we want to store the correlation as well?
                    # split gvar data in estimate (est) and standard deviation (err)
                    h5_handle.create_dataset(
                        f"{node}/{field.name}/{key}/est", data=gv.mean(value)
                    )
                    h5_handle.create_dataset(
                        f"{node}/{field.name}/{key}/err", data=gv.sdev(value)
                    )

            elif field.name == 'fcn':
            # callable (e.g. self.fcn) are pickeld using dill.dumps
                h5_handle.create_dataset(f"{node}/{field.name}", data = dumps(field_value,0) )
            elif field.name == 'prior_res':
                h5_handle.create_dataset(f"{node}/{field.name}", data = dumps(field_value,0) )
            else:
            # other types are usually fine to just dump into h5files.
                try:
                    h5_handle.create_dataset(f"{node}/{field.name}", data=field_value)
                except Exception as e:
                    print(f"Couldnt write: {field.name}, {type(field_value)}:\n{field_value}")

                    raise e

        return

    @staticmethod
    def deserialize(h5_handle: h5py.File, node: str) -> "FitResult":
        r"""
            @param h5_handle: h5py.File, file object pointing to an opened h5-file 
            @param node: str, path into the h5 file relative to the h5_handle

            Deserialize a FitResult from an h5 file. This is the inverse operation to serialize
        """

        te:int = h5_handle[f"{node}/te"][()]
        ts:int = h5_handle[f"{node}/ts"][()]


        # check if h5file contains bootstrap data:
        if "Nres" in h5_handle[node]:
            Nres:int = h5_handle[f"{node}/Nres"][()]
            
            out = FitResult(te=te, ts=ts, Nres=Nres)

        else:
            out = FitResult(te=te, ts=ts)

        # read in the data:
        for key in h5_handle[node]:
            # The callable function needs to be decoded (unpickled)
            if key == "fcn":
                setattr(out, key, loads(h5_handle[f"{node}/{key}"][()]))
            
            # dictionaries are stored with a level more
            elif isinstance(h5_handle[f"{node}/{key}"], h5py.Group):

                if key == "best_fit_param" or "best_fit_param_res":
                    key_value = gv.BufferDict()

                    # iterate over parameter keys
                    for item in h5_handle[f"{node}/{key}"]: 
                        # read central value (est)
                        est = h5_handle[f"{node}/{key}/{item}/est"][()]

                        # read standard deviation (err)
                        # We ignored correlation in the serialize function
                        err = h5_handle[f"{node}/{key}/{item}/err"][()]

                        # assemble the central value and error in a gvar
                        # Notice, we did not save correlations between gvars 
                        # in the first place
                        key_value[item] = gv.gvar(est, err)
                    
                    # finally set the dictionary in the class field
                    setattr(out, key, key_value)
            elif key == 'prior_res':
                setattr(out, key, loads(h5_handle[f"{node}/{key}"][()]))

            elif key == 'resample_type':
                setattr(out, key, h5_handle[f"{node}/{key}"][()].decode("utf-8"))

            # all other fields are probably fine
            elif isinstance(h5_handle[f"{node}/{key}"], h5py.Dataset):
                setattr(out, key, h5_handle[f"{node}/{key}"][()])

        return out




    # ###########################################
    # Importers
    # ###########################################
    # Currently, we only provide an interface for lsqfit. 
    # This can be extended by simply implementing a function
    # import_from_FITTER(self, *args) -> None
    # and fill the fields defined above

    def AIC_from_lsqfit(self, nlf: lsqfit.nonlinear_fit, augmented: bool = False, small_sample_correction: bool = True) -> float:
        r"""
            @param nlt: lsqfit.nonlinear_fit, Fit result from lsqfit. It stores all relevant information to
                                               compute the AIC (see below)
            @param augmented: bool, flag to calculate the augmented AIC, ie including priors (default: False)
            @param small_sample_correction: bool, flag to add/remove a small sample correction (default: False)

            This function is used to set the parameters
            self.AIC
            self.AIC_res
            self.aug_AIC
            self.aug_AIC_res

            Compute the Akaike information criterion for a fit result
            based on the chi^2 obtained from lsqfit.
            The form can be found in
                https://arxiv.org/abs/2305.19417
                https://arxiv.org/abs/2208.14983
                https://arxiv.org/abs/2008.01069
            equation 3 in the first:
                AIC = -2ln L^* + 2k - 2d_K 

                AIC_augmented = AIC + \sum_{\Theta} ( \Theta - p_\Theta )**2/var(p_\Theta) 

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
        Nparam: int = len(nlf.p)
        Ndata: int  = len(nlf.x)

        # start with the degree of freedom. 
        # TODO: This factor 2 is highly debated as it is very aggressive for many data sets
        #       We may want to come up with a way to allow a more flexible way of calculating
        #       the AIC.
        #       For reference see issue #8
        AIC: float = 2 * (Nparam - Ndata)

        if small_sample_correction:
            # This corrections is negligible if Ndata >> Nparam**2 and thus often very useful
            # it effectively favours models with less parameters
            AIC += (2*Nparam**2 + 2*Nparam)/(Ndata - Nparam - 1)

        # by default lsqfit includes priors to the chi^2 (if porvided) 
        if augmented:
            AIC += nlf.chi2
        # for non-augmented we have to explicitly recalculate chi^2
        else:
            AIC += gv.chi2( nlf.y, nlf.fcn( nlf.x, nlf.p ) )

        return AIC
    
    def chi2_from_lsqfit(self, nlf: lsqfit.nonlinear_fit, augmented: bool = False) -> float:
        r"""
            @param nlt: lsqfit.nonlinear_fit, Fit result from lsqfit. It stores all relevant information to
                                               compute the chi^2
            @param augmented: bool, flag to calculate the augmented AIi^2, ie including priors (default: False)

            This function is used to set the parameters
            self.chi2
            self.chi2_res
            self.aug_chi2
            self.aug_chi2_res
        """

        # by default lsqfit includes priors to the chi^2 (if provided) 
        if augmented:
            return nlf.chi2
        # for non-augmented we have to explicitly recalculate chi^2
        else:
            return gv.chi2( nlf.y, nlf.fcn( nlf.x, nlf.p ) )

    def import_from_lsqfit(self, nlf: lsqfit.nonlinear_fit, nres: None | int = None) -> None:
        """
            @param nlf: lsqfit.nonlinear_fit, lsqfit fit result. 
            @param nres: int, resample ID imports the fit result into the resample arrays at postion nres. If none, the central value fit fields
                              are populated
            save the interesting results from a lsqfit, if nres is given then save in corresponding row nres of the bootstrap parameters
        """
        # if the functional form hasn't been set we populate it here
        if self.fcn is None:
            self.fcn = nlf.fcn

        # if the degree of freedom hasn't been set we populate it here
        if self.dof is None:
            self.dof = nlf.dof

        # if nres is provided populate the resample fields at position nres
        if nres is not None:
            # check that the dictionary is set and fillable
            if not bool(self.best_fit_param_res):
                self.best_fit_param_res = {}

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
                    if key_red not in self.best_fit_param_res.keys():
                        self.best_fit_param_res[key_red] = np.empty(self.Nres, dtype=object)

                    # extract the parameter and exponentiate it
                    self.best_fit_param_res[key_red][nres] = np.exp(nlf.p[key])
                else:

                    # check if the resample array exists. If not set it
                    if key not in self.best_fit_param_res.keys():
                        self.best_fit_param_res[key] = np.empty(self.Nres, dtype=object)

                    # extract the parameter
                    self.best_fit_param_res[key][nres] = nlf.p[key]

            # Extract fit statistics
            self.chi2_res[nres]     = self.chi2_from_lsqfit(nlf, augmented = False)
            self.aug_chi2_res[nres] = self.chi2_from_lsqfit(nlf, augmented = True)

            self.Q_value_res[nres] = nlf.Q
            
            self.AIC_res[nres]     = self.AIC_from_lsqfit(nlf, augmented = False)
            self.aug_AIC_res[nres] = self.AIC_from_lsqfit(nlf, augmented = True)
            
            self.prior_res[nres] = nlf.prior

        # central value fit (if nres is None)
        else:
            if not bool(self.best_fit_param):
                self.best_fit_param = {}

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

                    # extract the parameter and exponentiate it
                    self.best_fit_param[key_red] = np.exp(nlf.p[key])
                else:
                    # extract the parameter
                    self.best_fit_param[key] = nlf.p[key]

            # Extract fit statistics
            self.chi2       = self.chi2_from_lsqfit(nlf, augmented = False)
            self.aug_chi2   = self.chi2_from_lsqfit(nlf, augmented = True)

            self.Q_value    = nlf.Q
            
            self.AIC        = self.AIC_from_lsqfit(nlf, augmented = False)
            self.aug_AIC    = self.AIC_from_lsqfit(nlf, augmented = True)
            
            self.prior      = nlf.prior

# end of class: FitResult 

