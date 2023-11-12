import itertools
import lsqfit 
import gvar as gv
import yaml 

@dataclass
class CorrelatorAnalyser:
    parameterFile: Path

    Nt: int
    beta: float
    U: float
    mu: float
    Nconf: int

    # p^+p, per configuration (Nconf,Nt,Nx,Nx) -> (Nconf,Nt,Nx)
    C_sp: np.ndarray
    # h^+h, per configuration (Nconf,Nt,Nx,Nx) -> (Nconf,Nt,Nx)
    C_sh: np.ndarray

    Nbst: int = 100
    Nx: int = None
    delta: float = None
    isIsleData: bool = True

    amplitudes: np.ndarray

    # bootstrap sample IDs (Nbst, Nconf)
    # We can generate bst samples of an observable using:
    # data.C_sp[data.k_bst].mean(axis=1) 
    # which has the shape (Nbst,*data.shape[1:])
    k_bst: np.ndarray = None
    
    # S, per configuration (Nconf, )
    actVals: np.ndarray = None

    # weights for the reweighting
    weights: np.ndarray = None
    weights_bst: np.ndarray = None
    weights_est: np.ndarray = None
    weights_err: np.ndarray = None
    
    # p^+p, per configuration (Nbst,Nconf,Nt,Nx,Nx) -> (Nbst,Nconf,Nt,Nx)
    C_sp_bst: np.ndarray = None
    C_sp_est: np.ndarray = None
    C_sp_err: np.ndarray = None
    C_sp_cov: np.ndarray = None

    # h^+h, per configuration (Nbst,Nconf,Nt,Nx,Nx) -> (Nbst,Nconf,Nt,Nx)
    C_sh_bst: np.ndarray = None
    C_sh_est: np.ndarray = None
    C_sh_err: np.ndarray = None
    C_sh_cov: np.ndarray = None

    # charge per site
    charge_bst: np.ndarray = None
    charge_est: np.ndarray = None
    charge_err: np.ndarray = None

    # Autocorrelations
    gamma_actVals_est: np.ndarray = None
    gamma_actVals_err: np.ndarray = None
    gamma_C_sp_est: np.ndarray = None
    gamma_C_sp_err: np.ndarray = None

    # Statistical Power
    NconfCuts: np.ndarray = None
    statistical_power_bst: np.ndarray = None
    statistical_power_est: np.ndarray = None
    statistical_power_err: np.ndarray = None

    # Effecitve Mass
    meff_bst: np.ndarray = None
    meff_est: np.ndarray = None
    meff_err: np.ndarray = None

    meff_extr_bst: np.ndarray = None
    meff_extr_est: np.ndarray = None
    meff_extr_err: np.ndarray = None

    # Single State Fit
    # { "{tstart}/{tend}": nonlinear_fit }
    # stores the fit result as returned by lsqfit.nonlinear_fit
    oneState_fit: dict = field(default_factory=dict)
    # stores the results {'A0','E0','fcn',AIC} from each bootstrap fit
    oneState_fit_bst: dict = field(default_factory=dict)
    # stores the results {'A0','E0','fcn',AIC} from the central value fit
    oneState_fit_est: dict = field(default_factory=dict)
    # stores the std {'A0','E0',''fcn} over the bootstrap samples
    oneState_fit_err: dict = field(default_factory=dict)

    # Two State Fit
    # { "{tstart}/{tend}": nonlinear_fit }
    # stores the fit result as returned by lsqfit.nonlinear_fit
    twoState_fit: dict = field(default_factory=dict)
    # stores the results {'A0','E0','A1','E1','fcn'} from each bootstrap fit
    twoState_fit_bst: dict = field(default_factory=dict)
    # stores the results {'A0','E0','A1','E1','fcn'} from the central value fit
    twoState_fit_est: dict = field(default_factory=dict)
    # stores the std {'A0','E0','A1','E1','fcn'} over the bootstrap samples
    twoState_fit_err: dict = field(default_factory=dict)

    # stores the model average for each parameter on {'A0','E0',...} bootstrap sample
    modelAverage_bst: dict = field(default_factory=dict)
    modelAverage_est: dict = field(default_factory=dict)
    modelAverage_err: dict = field(default_factory=dict)

    ignoreCache: bool = False
    Nplot = 100

    def read_data(self):
        if Path(self.analysisCache()).exists() and not self.ignoreCache:
            self.deserialize()
            return True

        with open(self.parameterFile, 'r') as file:
            meta_yml = yaml.safe_load(file)
            
            if self.isIsleData: # isle data
                adj = meta_yml["adjacency"]
                hop = meta_yml["hopping"]
            else: # NSL Data
                adj = meta_yml['system']['adjacency']
                hop = meta_yml['system']['hopping']

        kappa = np.zeros((Nx,Nx))
        for x in range(len(adj)):
            y,z = adj[x]
            try:
                kappa[y,z] = hop[x]
            except:
                kappa[y,z] = hop
        kappa += kappa.T
        
        self.freeEvals, self.amplitudes = np.linalg.eigh(kappa)
      
        return False

    def analysisCache(self):
        return f"data/AnalysisCache/Nt{self.Nt:g}_beta{self.beta:g}_U{self.U:g}_mu{self.mu:g}.h5"

    def write(self,grp,key,val):
        if isinstance(val, nonlinear_fit):
            #gv.dumps pickles the lsqfit.nonlinear_fit object and converts it to a byte string
            # casting it to np.void allows the string to be stored into the h5 file.
            # this trick was developed by Evan Berkowitz
            grp.create_dataset(key, data=np.void(gv.dumps(val)))
        else:
            grp.create_dataset(key, data=val)

    def write_dict(self,grp, d):
        for key,val in d.items():
            if isinstance(val,(dict,gv.BufferDict)):
                self.write_dict(grp.create_group(key),val)
            else:
                self.write(grp,key,val)

    def read_dict(self,grp,name, key):
        if isinstance(grp, h5.Dataset) and "fit" == name[-3:] :
            # In this case we are reading the fit result for oneState_fit, twoState_fit, ...
            # we need to load the pickled nonlinear_fit object using the gv.loads method 
            self.__dict__[name][key[1:]] = gv.loads( grp[()] )
            return
        elif isinstance(grp, h5.Dataset):
            # for normal dictionaries
            self.__dict__[name][key[1:]] = grp[()]
            return

        for grpKey in grp.keys():
            # the keypath is expected to be 
            # /tstart*/tend*/k* 
            # where the * are integers, tstart is the fit window start, tend the fit window end and k are the irrep ids
            # until that is the case for key, we recursively call read_dict (see else)

            if isinstance(grp[grpKey], h5.Dataset) and "/k" in key:
                # In this case key = /tstart*/tend*/k* 
                # the grpkeys are now the fit parameter keys "A0","E0","c",... or the fit result function key "fcn"
                # this should fill fields like oneState_fit_est, twoState_fit_est, ...

                # This is a nested dict, at each key, we store a dictionary of parameters
                # the key contains a leading / which we ignore
                if not key[1:] in self.__dict__[name].keys() :
                    self.__dict__[name][key[1:]] = {}

                self.__dict__[name][key[1:]][grpKey] = grp[grpKey][()]
            else:
                # unless in canse /k was on int the key we haven't reached the end of this keypath
                # therefore continue
                self.read_dict(grp[grpKey],name,f"{key}/{grpKey}")

    def serialize(self):
        with h5.File(self.analysisCache(), 'w') as h5file:
            for name, obj in self.__dict__.items():
                if isinstance(obj,Path):
                    obj = str(obj)
                elif isinstance(obj,(dict,gv.BufferDict)):
                    self.write_dict(h5file.create_group(name),obj)
                else:
                    self.write(h5file,name,obj)

    def deserialize(self):
        with h5.File(self.analysisCache(), 'r') as h5file:
            for name, obj in h5file.items():
                if name == h5file:
                    self.__dict__[name] = Path(obj[()])
                if isinstance(obj, h5.Group):
                    self.__dict__[name] = {}
                    self.read_dict(obj,name,'')

                else:
                    self.__dict__[name] = obj[()]

    def measure_autocorrelation(self):
        def auto_corr(obs):
            shape = obs.shape[1:]

            # With the data we are generating autocorrelation is not an issue. If the zero crossing isn't found 
            # go back and increase this up to self.Nconf//2
            W = self.Nconf//2
            
            gamma_bst = np.zeros( (self.Nbst, W, *shape) , dtype=obs.dtype )

            obs_bst = obs[self.k_bst] # shape = (Nbst,Nconf,*obs.shape[1:])
            obs_est = obs_bst.mean(axis=1) # shape = (Nbst,*obs.shape[1:])

            for tau in tqdm(range(W)):
                gamma_bst[:,tau,...] = np.mean( 
                    (obs_bst - obs_est[:,None,...]) * ( np.roll(obs_bst,tau,axis=1) - obs_est[:,None,...]).conj(), 
                    axis = 1
                ).real

            gamma_bst /= gamma_bst[:,0,...][:,None,...]

            gamma_est = gamma_bst.mean(axis=0)
            gamma_err = gamma_bst.std(axis=0)

            zero_crossing = np.argmax(gamma_est<=0,axis=0)

            W = np.max(zero_crossing)

            return gamma_est[:W+1], gamma_err[:W+1]
         

        logger.info("Measuring autocorrelation of S...")
        self.gamma_actVals_est,self.gamma_actVals_err = auto_corr(self.actVals.real)
        logger.info("Measuring autocorrelation of C_sp...")
        # We have data that is almost uncorrelated, thus we save some time by just providing the first few configurations
        self.gamma_C_sp_est,self.gamma_C_sp_err = auto_corr(self.C_sp.mean(axis=(1,2,3)))
        self.gamma_C_sp_est = self.gamma_C_sp_est.real

    def reweighting_weights(self):
        logger.info("Measuring reweighting weights...")

        self.weights = np.exp(-1j*self.actVals.imag) # shape = (Nconf,)
        self.weights_bst = self.weights[self.k_bst] # shape = (Nbst,Nconf)
        self.weights_est = self.weights_bst.mean(axis=(0,1)) # shape = (,)
        self.weights_err = self.weights_bst.std (axis=(0,1)) # shape = (,)

    def measure_statistical_power(self):
        logger.info("Measuring statistical power...")
        ImS_bst = self.actVals.imag[self.k_bst]

        self.NconfCuts = np.arange(int(0.1*self.Nconf),self.Nconf, int(0.1*self.Nconf) )

        self.statistical_power_bst = np.zeros( (self.Nbst, len(self.NconfCuts)) )
        for i,N in enumerate(self.NconfCuts):
            self.statistical_power_bst[:,i] = np.abs(np.exp(-1j*ImS_bst[:,:N]).mean(axis=1))

        self.statistical_power_est = self.statistical_power_bst.mean(axis=0)
        self.statistical_power_err = self.statistical_power_bst.std(axis=0)

    def measure_C_sp(self):
        logger.info("Measuring C_sp...")

        # (block) diagonalize
        self.C_sp = np.diagonal(
            amplitudes[None,None,:,:] @  self.C_sp @ amplitudes.T.conj()[None,None,:,:],
            axis1=2,axis2=3
        ) # shape = (Nconf,Nt,Nx)

        # bootstrap
        self.C_sp_bst = self.C_sp[self.k_bst] # shape = (Nbst,Nconf,Nt,Nx)

        # reweight
        self.C_sp_bst = np.mean( self.C_sp_bst * self.weights_bst[:,:,None,None], axis=1) / self.weights_bst.mean(1)[:,None,None] # shape = (Nbst,Nt,Nx)
        self.C_sp = self.C_sp * self.weights[:,None,None] / self.weights_est # shape = (Nconf,Nt,Nx)

        # average and uncertainty
        self.C_sp_est = self.C_sp_bst.mean(axis=0).real # shape = (Nt,Nx)
        self.C_sp_err = self.C_sp_bst.std(axis=0) # shape = (Nt,Nx)

        self.C_sp_cov = np.zeros( (self.Nt,self.Nt,self.Nx) )
        for k in range(self.Nx):
            self.C_sp_cov[:,:,k] = np.cov(self.C_sp_bst[:,:,k], rowvar=False).real

        self.C_sp_bst = self.C_sp_bst.real

    def measure_C_sh(self):
        logger.info("Measuring C_sh...")

        # (block) diagonalize
        self.C_sh = np.diagonal(
            amplitudes[None,None,:,:] @  self.C_sh @ amplitudes.T.conj()[None,None,:,:],
            axis1=2,axis2=3
        ) # shape = (Nconf,Nt,Nx)

        # bootstrap
        self.C_sh_bst = self.C_sh[self.k_bst] # shape = (Nbst,Nconf,Nt,Nx)

        # reweight
        self.C_sh_bst = np.mean( self.C_sh_bst * self.weights_bst[:,:,None,None], axis=1) / self.weights_bst.mean(1)[:,None,None] # shape = (Nbst,Nt,Nx)
        self.C_sh = self.C_sh * self.weights[:,None,None] / self.weights_est # shape = (Nconf,Nt,Nx)

        # average and uncertainty
        self.C_sh_est = self.C_sh_bst.mean(axis=0).real # shape = (Nt,Nx)
        self.C_sh_err = self.C_sh_bst.std(axis=0) # shape = (Nt,Nx)

        self.C_sh_cov = np.zeros( (self.Nt,self.Nt,self.Nx) )
        for k in range(self.Nx):
            self.C_sh_cov[:,:,k] = np.cov(self.C_sh_bst[:,:,k], rowvar=False).real

        self.C_sh_bst = self.C_sh_bst.real

    def measure_charge(self):
        logger.info("Measuring charge...")
        self.charge_bst = np.zeros( (self.Nbst,) )

        if self.mu != 0:
            # The correlators are already reweights
            # q_x = C_sp_xx(tau=0) - C_sh_xx(tau=0)
            self.charge_bst = np.sum(self.C_sp_bst[ :, 0, : ] - self.C_sh_bst[ :, 0, : ], axis=1) # shape = (Nbst,)

        self.charge_est = np.mean(self.charge_bst,axis=0) 
        self.charge_err = np.std (self.charge_bst,axis=0) 

    def measure_meff(self):
        logger.info("Measuring meff...")

        self.meff_bst = (np.log(np.roll(self.C_sp_bst[:,:,:],-1,axis=1)) - np.log(np.roll(self.C_sp_bst[:,:,:],1,axis=1)))/(2*self.delta) # shape = (Nbst,Nt,Nx)
        self.meff_est = self.meff_bst.mean(axis=0).real # shape = (Nt,Nx)
        self.meff_err = self.meff_bst.std(axis=0) # shape = (Nt,Nx)

        self.meff_extr_bst = self.meff_bst[:,self.Nt//2,:] # shape = (Nbst,Nx)
        self.meff_extr_est = self.meff_extr_bst.mean(axis=0).real # shape = (Nx,)
        self.meff_extr_err = self.meff_extr_bst.std(axis=0) # shape = (Nx,)

    def fit_energy(self):     
        logger.info("Fitting Energies...")

        def AIC(fit):
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
            L = fit.chi2
            k = 2 * len(fit.p) 
            d_K = len(fit.x)

            return L + 2*k - 2*d_K

        def fit(model, prior, tau, C_est, C_cov ):
            r"""
                Perform a fit to the correlator data using the model and prior provided.
                Compute the Akaike information criterion for the fit.
            """
            # pack the pre-processed correlator data into a gvar taking the tau-tau correlation
            # into account
            # Here the correlation is estimated from the bootstrap samples
            corr_gvar = gv.gvar(C_est, C_cov)

            # perform the fit to the central values
            fitResult = nonlinear_fit( data=(tau,corr_gvar), fcn=model, prior=prior )

            # compute the AIC for the fit and return
            return fitResult, AIC(fitResult)
        
        def oneStateFit(k,tstart,tend,sign):
            def model(tau,p):
                return p["A0"]*np.exp(sign*p["E0"]*tau)

            # declare a numpy slice to represent the fit window
            window = np.s_[tstart:tend]

            # the abscissa is time
            tau = np.arange(tstart, tend)

            #############################################################################
            ## Central Value Fit
            #############################################################################

            # we declare a prior for the fit parameters
            prior_oneState = gv.BufferDict()

            # Using the effective mass for the prior
            meff_est = np.abs(self.meff_bst[:,tstart:tend,k]).mean(axis=(0,1))*self.delta
            meff_err = np.abs(self.meff_bst[:,tstart:tend,k]).mean(axis=(0,1))*self.delta
            prior_oneState['E0'] = gv.gvar(meff_est, 100*meff_err)

            # We have no knowledge about A0, thus we start with a flat prior
            prior_oneState['A0'] = gv.gvar( 1,1 )

            oneState_fit, oneState_fit_AIC = fit(
                model, prior_oneState, 
                tau, 
                self.C_sp_est[window,k], 
                self.C_sp_cov[window,window,k]
            )

            # store the fit results, for convenience we store the resulting parameters 
            # in a separate dictionary
            self.oneState_fit[f'tstart{tstart}/tend{tend}/k{k}'] = oneState_fit
            self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}'] = oneState_fit.pmean
            self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['sign'] = sign
            # also store the AIC
            self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['AIC'] = oneState_fit_AIC
            # we prepare the fit function for plotting
            self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['fcn'] = oneState_fit.fcn(
                np.linspace(tstart,tend,self.Nplot), oneState_fit.pmean
            )

            #############################################################################
            ## Bootstrap Fit
            #############################################################################
            # we store the bootstrap results in a separate dictionary
            self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}'] = { 
                'A0': np.zeros(self.Nbst), 
                'E0': np.zeros(self.Nbst),
                'AIC': np.zeros(self.Nbst),
                'fcn': np.zeros((self.Nbst,self.Nplot)) 
            }

            # perform the bootstrap using the precomputed bootstrap samples
            for nbst in range(self.Nbst):
                fitbst,fitbst_AIC = fit(
                    model, prior_oneState, 
                    tau, 
                    self.C_sp_bst[nbst,window,k], 
                    self.C_sp_cov[window,window,k]
                )

                self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['A0'][nbst] = fitbst.pmean['A0']
                self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['E0'][nbst] = fitbst.pmean['E0']
                self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['AIC'][nbst] = fitbst_AIC
                self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['fcn'][nbst,:] = fitbst.fcn(
                    np.linspace(tstart,tend,self.Nplot), fitbst.pmean
                )
            
            # from the bootstrap results we can compute the uncertainty
            self.oneState_fit_err[f'tstart{tstart}/tend{tend}/k{k}'] = {
                'A0': np.std(self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['A0'], axis=0),
                'E0': np.std(self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['E0'], axis=0),
                'fcn': np.std(self.oneState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['fcn'], axis=0)
            }
        # end oneStateFit

        def twoStateFit(k,tstart,tend,sign):
            def model(tau,p):
                return p["A0"]*np.exp(sign* p["E0"] * tau)\
                      +p["A1"]*np.exp(sign*(p["E0"] + p["dE1"])*tau)

            # declare a numpy slice to represent the fit window
            window = np.s_[tstart:tend]

            # the abscissa is time
            tau = np.arange(tstart, tend)

            #############################################################################
            ## Central Value Fit
            #############################################################################
            # we declare a prior for the fit parameters
            prior_twoState = gv.BufferDict()
            # we start with flat priors
            # E0 is guided by the effective mass defining the fit interval, yet with large uncertainty
            prior_twoState['log(E0)'] = gv.log(gv.gvar(
                self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['E0'], 
                100*self.oneState_fit_err[f'tstart{tstart}/tend{tend}/k{k}']['E0'], 
            ))
            prior_twoState['log(dE1)'] = gv.log(gv.gvar(
                prior_twoState['E0'].mean, 1 
            ))
            prior_twoState['A0'] = gv.gvar(
                self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['A0'],
                100*self.oneState_fit_err[f'tstart{tstart}/tend{tend}/k{k}']['A0']
            )
            prior_twoState['A1'] = gv.gvar( 
                self.oneState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['A0'],1 
            )

            print(prior_twoState)

            twoState_fit,twoState_fit_AIC = fit(
                model, prior_twoState, 
                tau, 
                self.C_sp_est[window,k], 
                self.C_sp_cov[window,window,k]
            )

            # store the fit results, for convenience we store the resulting parameters 
            # in a separate dictionary
            self.twoState_fit[f'tstart{tstart}/tend{tend}/k{k}'] = twoState_fit
            self.twoState_fit_est[f'tstart{tstart}/tend{tend}/k{k}'] = {
                "E0": gv.exp(twoState_fit.pmean['log(E0)']),
                "dE1": gv.exp(twoState_fit.pmean['log(dE1)']),
                "E1":  gv.exp(twoState_fit.pmean['log(E0)'])+gv.exp(twoState_fit.pmean['log(dE1)']),
                "A0": twoState_fit.pmean['A0'],
                "A1": twoState_fit.pmean['A1'],
                "sign": sign
            }
            self.twoState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['AIC'] = twoState_fit_AIC
            # we prepare the fit function for plotting
            self.twoState_fit_est[f'tstart{tstart}/tend{tend}/k{k}']['fcn'] = twoState_fit.fcn(
                np.linspace(tstart,tend,self.Nplot), twoState_fit.pmean
            )

            #############################################################################
            ## Bootstrap Fit
            #############################################################################
            # we store the bootstrap results in a separate dictionary
            self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}'] = dict([
                (key,np.zeros(self.Nbst)) for key in self.twoState_fit_est[f'tstart{tstart}/tend{tend}/k{k}'].keys()
            ])
            self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['AIC'] = np.zeros(self.Nbst)
            self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['fcn'] = np.zeros((self.Nbst,self.Nplot)) 

            # perform the bootstrap using the precomputed bootstrap samples
            for nbst in range(self.Nbst):
                fitbst,fitbst_AIC = fit(
                    model, prior_twoState, 
                    tau, 
                    self.C_sp_bst[nbst,window,k], 
                    self.C_sp_cov[window,window,k]
                )

                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']["E0"][nbst] = gv.exp(fitbst.pmean['log(E0)'])
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']["dE1"][nbst] = gv.exp(fitbst.pmean['log(dE1)'])
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']["E1"][nbst] =  gv.exp(fitbst.pmean['log(E0)'])+gv.exp(fitbst.pmean['log(dE1)'])
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']["A0"][nbst] = fitbst.pmean['A0']
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']["A1"][nbst] = fitbst.pmean['A1']
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['AIC'][nbst] = fitbst_AIC
                self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}']['fcn'][nbst,:] = fitbst.fcn(
                    np.linspace(tstart,tend,self.Nplot), fitbst.pmean
                )
            
            # from the bootstrap results we can compute the uncertainty
            self.twoState_fit_err[f'tstart{tstart}/tend{tend}/k{k}'] = dict([
                (key, np.std(arr,axis=0)) for key,arr in self.twoState_fit_bst[f'tstart{tstart}/tend{tend}/k{k}'].items()
            ])
            del self.twoState_fit_err[f'tstart{tstart}/tend{tend}/k{k}']['AIC']
        # end twoStateFit

        for k in (pbar:=tqdm(range(self.Nx))):
            # We want to identify the ground state described by the correlator
            # Some correlators describe a forward (positive E0) and some a backward (negative E0) propagating state

            # The position of the maximum always determines which it is.
            t0 = np.argmax(self.C_sp_est[:,k])

            # If we are on the left site: positive E0; otherwise: negative E0
            if t0 == 0: sign = -1
            else: sign = +1
            
            tendList = []
            for tendLeeway in [0.05,0.1,0.15]:
                # We need to identify the maximal point that we want to include into the fit.
                # We thus attempt to find the turning point of the correlator
                # and remove 5% of the time points to negate excited states propagating in.
                tmax = np.argmin(self.C_sp_est[:,k]) + sign*int(np.ceil(tendLeeway*self.Nt))

                # This is theoretically a nice filter, but it is very strong and thus some fits don't get executed...
                # therefore we leave it out for now.

                # This can be difficult as the data becomes noisy at large times.
                # we therefore find the first point at which the signal to noise ratio
                SnT = np.abs(self.C_sp_est[:,k])/self.C_sp_err[:,k]
                # drops to 1
                # Note, this is an arbitrary choice. We might want to be more conservative with 
                # larger minimal values then it's bare minimum. On the other hand, we don't 
                # want to neglect data points that carry information.
                tSnT = np.where( SnT < 1. )[0]
                if len(tSnT) == 0:
                    # In this case we just try our luck and see what happens
                    # the resulting AIC should become extremely large.
                    tSnT = tmax
                else:
                    tSnT = tSnT[0]

                # we now simply take the minimum of the two to identify the best interval ending
                if sign == -1: tend = np.min([tmax,tSnT])
                else: tend = np.max([tmax,tSnT])

                # make sure that we don't fit the same interval twice
                if tend in tendList: 
                    continue
                tendList.append(tend)

                # To reduce the number of fit intervals we fix the number of fits
                numFits = 3
                intervalStep = np.max([int( np.abs(t0 - tend) / numFits ),1])

                # we now vary the starting point of the fit and skipping fit intervals with to little points
                for tstart in range(t0,tend,-sign*intervalStep):
                    ts,te = np.sort([tstart,tend])

                    #########################################################################
                    ## Single State Fit
                    #########################################################################
                    if np.abs(te-ts) <= 4: continue # we have 2 parameter
                    pbar.set_description( f"One State Fit: k={k}, tstart={tstart}, tend={tend}" )
                    oneStateFit(k,ts,te,sign)

                    ########################################################################
                    # Two State Fit
                    ########################################################################
                    if np.abs(te-ts) <= 6: continue # we have 4 parameters
                    pbar.set_description( f"Two State Fit: k={k}, tstart={tstart}, tend={tend}" )
                    twoStateFit(k,ts,te,sign)
                # end for tstart
            # end for tendLeeway
        # end for k
    # end fit_energy

    def modelAverage(self):
        r"""
            We average the different fit results over the fit ranges and the fit models.
            As a weight we compute the normalized akaike probability as
            .. math::
                w_i = \frac{\exp(- AIC_i / 2)}{\sum_j \exp(- AIC_j/2)}
            where i,j are indexing the different fits.
        """
        # Create room for the results
        self.modelAverage_bst = {
            "A0": np.zeros((self.Nbst,self.Nx)),
            "E0": np.zeros((self.Nbst,self.Nx)),
            "A1": np.zeros((self.Nbst,self.Nx)),
            "E1": np.zeros((self.Nbst,self.Nx)),
        }
        self.modelAverage_est = {
            "A0": np.zeros(self.Nx),
            "E0": np.zeros(self.Nx),
            "A1": np.zeros(self.Nx),
            "E1": np.zeros(self.Nx),
        }
        self.modelAverage_err = {
            "A0": np.zeros(self.Nx),
            "E0": np.zeros(self.Nx),
            "A1": np.zeros(self.Nx),
            "E1": np.zeros(self.Nx),
        }

        for k in range(self.Nx):
            # Sort the results into lists
            # first we need to get all keys belonging to the irrep k
            oneState_keys = []
            for key in self.oneState_fit_est.keys():
                if f"/k{k}" in key:
                    oneState_keys.append(key)
            twoState_keys = []
            for key in self.twoState_fit_est.keys():
                if f"/k{k}" in key:
                    twoState_keys.append(key)

            # then we find all the AIC values
            oneState_AIC = np.array([
                self.oneState_fit_est[key]['AIC'] for key in oneState_keys
            ])
            twoState_AIC = np.array([
                self.twoState_fit_est[key]['AIC'] for key in twoState_keys
            ])

            # next we find all the parameters A0,E0,A1,E1,...
            # The transpose puts the bootstrap index in the first dimension
            # Central value fits
            A0_est_list = np.array(
                [self.oneState_fit_est[key]['A0'] for key in oneState_keys] + 
                [self.twoState_fit_est[key]['A0'] for key in twoState_keys]
            )
            E0_est_list = np.array(
                [self.oneState_fit_est[key]['E0'] for key in oneState_keys] + 
                [self.twoState_fit_est[key]['E0'] for key in twoState_keys]
            )
            A1_est_list = np.array(
                [self.twoState_fit_est[key]['A1'] for key in twoState_keys]
            )
            E1_est_list = np.array(
                [self.twoState_fit_est[key]['E1'] for key in twoState_keys]
            )

            # Bootstrap fits
            A0_bst_list = np.array(
                [self.oneState_fit_bst[key]['A0'] for key in oneState_keys] + 
                [self.twoState_fit_bst[key]['A0'] for key in twoState_keys]
            ).T
            E0_bst_list = np.array(
                [self.oneState_fit_bst[key]['E0'] for key in oneState_keys] + 
                [self.twoState_fit_bst[key]['E0'] for key in twoState_keys]
            ).T
            A1_bst_list = np.array(
                [self.twoState_fit_bst[key]['A1'] for key in twoState_keys]
            ).T
            E1_bst_list = np.array(
                [self.twoState_fit_bst[key]['E1'] for key in twoState_keys]
            ).T

            # determine the un-normalized weights
            oneState_w = np.exp(-oneState_AIC/2)
            twoState_w = np.exp(-twoState_AIC/2)

            # model average the parameters
            A0_bst = np.average( A0_bst_list, axis=1, weights=np.append(oneState_w,twoState_w) )
            A0_est = np.average( A0_est_list, axis=0, weights=np.append(oneState_w,twoState_w) )
            A0_err = np.std(A0_bst, axis=0)
            E0_bst = np.average( E0_bst_list, axis=1, weights=np.append(oneState_w,twoState_w) )
            E0_est = np.average( E0_est_list, axis=0, weights=np.append(oneState_w,twoState_w) )
            E0_err = np.std(E0_bst, axis=0)
            A1_bst = np.average( A1_bst_list, axis=1, weights=twoState_w )
            A1_est = np.average( A1_est_list, axis=0, weights=twoState_w )
            A1_err = np.std(A1_bst, axis=0)
            E1_bst = np.average( E1_bst_list, axis=1, weights=twoState_w )
            E1_est = np.average( E1_est_list, axis=0, weights=twoState_w )
            E1_err = np.std(E1_bst, axis=0)

            # store the results
            self.modelAverage_bst["A0"][:,k] = A0_bst
            self.modelAverage_bst["E0"][:,k] = E0_bst
            self.modelAverage_bst["A1"][:,k] = A1_bst
            self.modelAverage_bst["E1"][:,k] = E1_bst

            self.modelAverage_est["A0"][k] = A0_est
            self.modelAverage_est["E0"][k] = E0_est
            self.modelAverage_est["A1"][k] = A1_est
            self.modelAverage_est["E1"][k] = E1_est

            self.modelAverage_err["A0"][k] = A0_err
            self.modelAverage_err["E0"][k] = E0_err
            self.modelAverage_err["A1"][k] = A1_err
            self.modelAverage_err["E1"][k] = E1_err
        # end for k
    # end modelAverage

    def __post_init__(self):
        isSerealized = self.read_data()

        if not isSerealized:
            if self.k_bst is None:
                self.k_bst = np.random.randint(0,self.Nconf, size=(self.Nbst, self.Nconf))
            else:
                Nbst_,Nconf_ = self.k_bst.shape
                if Nbst_ != self.Nbst or Nconf != self.Nconf:
                    raise RuntimeError( "Passed k_bst does not align with Nbst: {Nbst_} (k_bst) != {self.Nbst} (given) or Nconf: {Nconf_} (given) != {self.Nconf} " )
            
            self.delta = self.beta/self.Nt
    
            self.reweighting_weights()

            self.measure_autocorrelation()

            self.measure_statistical_power()

            self.measure_C_sp()

            self.measure_C_sh()

            self.measure_charge()

            self.measure_meff()

            self.fit_energy()

            self.modelAverage()
            
            self.serialize()
        else:
            logger.info(f"Data already serialized, load from {self.analysisCache()}")

    def latex_repr(self):
        return rf"Perylene$\left(N_t={self.Nt},\, \beta={self.beta:g},\, U={self.U:g},\, \mu={self.mu:g}\, \vert\, N_\mathrm{{conf}}={self.Nconf}\right)$"

    def __repr__(self):
        return rf"Perylene(Nt={self.Nt}, beta={self.beta:g}, U={self.U:g}, mu={self.mu:g}, Nconf={self.Nconf})"

