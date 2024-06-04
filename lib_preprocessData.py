from tqdm import tqdm

from dataclasses import dataclass,field, InitVar

from pathlib import Path

import numpy as np

import h5py as h5

import gvar as gv

import itertools

from lib_perylene import *

@dataclass
class Data:
    h5path: Path

    Nt: int
    beta: float
    U: float
    mu: float
    Nconf: int
    Nbst: int = 100
    delta: float = None
    Nx: int = 20
    isIsleData: bool = True

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
    
    # p p^+, per configuration (Nconf,Nt,Nx)
    C_sp: np.ndarray = None
    C_sp_bst: np.ndarray = None
    C_sp_est: np.ndarray = None
    C_sp_err: np.ndarray = None
    C_sp_cov: np.ndarray = None

    # h h^+, per configuration (Nconf,Nt,Nx)
    C_sh: np.ndarray = None
    C_sh_bst: np.ndarray = None
    C_sh_est: np.ndarray = None
    C_sh_err: np.ndarray = None
    C_sh_cov: np.ndarray = None

    # Autocorrelations
    gamma_avg: np.ndarray = None
    gamma_spr: np.ndarray = None
    tauInt_avg: np.ndarray = None
    tauInt_spr: np.ndarray = None
    tauIntCut: int = None

    gammaPhi_avg: np.ndarray = None
    gammaPhi_spr: np.ndarray = None
    tauIntPhi_avg: np.ndarray = None
    tauIntPhi_spr: np.ndarray = None
    tauIntPhiCut: int = None

    # Statistical Power
    NconfCuts: np.ndarray = None
    statistical_power_bst: np.ndarray = None
    statistical_power_est: np.ndarray = None
    statistical_power_err: np.ndarray = None

    def read_data(self):
        if self.isIsleData:
            with h5.File(self.h5path,'r') as h5f:
                print("Reading Action...")
                self.actVals = h5f["weights/actVal"][0:self.Nconf]

                print("Reading C_sp...")
                self.C_sp = np.moveaxis(h5f["correlation_functions/single_particle/destruction_creation"][0:self.Nconf,:,:,:],3,1)
                self.C_sh = np.moveaxis(h5f["correlation_functions/single_hole/destruction_creation"][0:self.Nconf,:,:,:],3,1)
                self.phi_ = np.zeros((self.Nconf,self.Nt,self.Nx),dtype=complex)
                for n_ in range(self.Nconf):
                    n = n_*10
                    self.phi_[n_,:,:] = h5f[f"configuration/{1000+n}/phi"][()].reshape(self.Nt,self.Nx)

        else:
            self.actVals = np.zeros(self.Nconf,dtype=complex)
            self.C_sp = np.zeros((self.Nconf,self.Nt,self.Nx,self.Nx),dtype=complex)
            self.C_sh = np.zeros((self.Nconf,self.Nt,self.Nx,self.Nx),dtype=complex)
            self.phi_ = np.zeros((self.Nconf,self.Nt,self.Nx),dtype=complex)
            with h5.File(self.h5path,'r') as h5f:
                for n in range(self.Nconf):
                    self.actVals[n] = h5f[f"markovChain/{n}/actVal"][()]
                    self.C_sp[n,:,:,:] = h5f[f"markovChain/{n}/correlators/single/particle"][()].reshape(self.Nt,self.Nx,self.Nx)
                    self.C_sh[n,:,:,:] = h5f[f"markovChain/{n}/correlators/single/hole"][()].reshape(self.Nt,self.Nx,self.Nx)
                    self.phi_[n] = h5f[f"markovChain/{n}/phi"][()].reshape(self.Nt,self.Nx)
    # end read_data()

    def measure_autocorrelation(self):
        def auto_corr(obs):
            r"""

            """
            # Linearize axes that contain other operators
            # here we assume axis=0 corresponds to the monte carlo time
            shape = obs.shape[1:]
            if len(shape) >= 1: 
                numObs =  np.prod(shape)
            else:
                numObs = 1
            obs = np.reshape( obs, (self.Nconf,numObs) )
            
            # reweight
            weights = np.exp(-1j*self.actVals.imag) # shape = (Nconf,)            
            obs *= weights[:,None] / weights.mean(axis=0)

            obs_err = np.std(obs,axis=0) #* np.sqrt(self.Nconf/(self.Nconf-1) )

            # Define half of the markov chain length
            W = self.Nconf//2

            # compute normalized autocorrelation and remove periodic terms
            gamma = gv.dataset.autocorr(obs.real)[:W,:]

            # Compute the integrated autocorrelation time
            max_tau_int = np.zeros((W,numObs))
            chosenCuts = np.zeros(numObs,dtype = int)
            
            # find the best cut of 
            # Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms by Sokal, after Eq 3.19
            c = 10 # a number of 4 - 10 depending on the exponentialness of gamma
            for obsID in range(numObs):
                for M in range(W):
                    # truncated estimator
                    max_tau_int[M,obsID] =  0.5 + np.sum( gamma[1:M,obsID], axis=0 ) 

                    if M >= c*max_tau_int[M,obsID]:
                        chosenCuts[obsID] = M
                        break

            # We now want to report the largest integrated autocorrelation time of our observable set
            # thus that the values at the chosen cut and take the maximum over all observables
            maxTauInt = np.argmax( max_tau_int[chosenCuts, np.arange(numObs) ] )

            tau_int_est = max_tau_int[chosenCuts,maxTauInt][0]
            chosenCut = chosenCuts[maxTauInt]

            # Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms by Sokal, Eq: 3.18 & 3.19
            tau_int_err = (0.5*np.sum(gamma[chosenCut:W,maxTauInt]))**2
            tau_int_err+= (1*(2*chosenCut + 1) / self.Nconf) * tau_int_est**2
            tau_int_err = np.sqrt(tau_int_err)

            print(f"\t - tau_int cut: {chosenCut}")

            return gamma[:,maxTauInt], obs_err[maxTauInt] * np.sqrt( self.Nconf/(self.Nconf-np.arange(W) ) ), \
                   tau_int_est, tau_int_err, chosenCut 

        print("Measuring autocorrelation of C_sp...")
        # We have data that is almost uncorrelated, thus we save some time by just providing the first few configurations
        print("\t Averaging")
        corr = self.C_sp.copy()
        corr[:,1:,:,:] += np.flip(self.C_sh.copy()[:,1:,:,:], axis=1)
        corr[:,1:,:,:] /= 2

        print("\t Measuring")
        self.gamma_avg, self.gamma_spr, self.tauInt_avg, self.tauInt_spr, self.tauIntCut = auto_corr(corr)

        print(f"\t - max(tauInt) = {gv.gvar(self.tauInt_avg, self.tauInt_spr)}")

    def reduce_raw_data(self):
        step = np.max([np.floor( 2*self.tauInt_avg ),1]).astype(int)

        self.C_sp = self.C_sp[::step]
        self.C_sh = self.C_sh[::step]
        self.actVals = self.actVals[::step]
        self.Nconf = self.actVals.shape[0]

    def reweighting_weights(self):
        print("Measuring reweighting weights...")

        self.weights = np.exp(-1j*self.actVals.imag) # shape = (Nconf,)
        self.weights_bst = self.weights[self.k_bst].mean(axis=1) # shape = (Nbst)
        self.weights_est = self.weights_bst.mean(axis=0) # shape = (,)
        self.weights_err = self.weights_bst.std (axis=0) # shape = (,)

    def measure_statistical_power(self):
        r"""
            We calculate the statistical power 
            
            |Σ| = < exp(-i Im S) > |_{Ncut}

            For various cuts of the number of configurations Nconf.
            Ncut = 0.1*Nconf, 0.2*Nconf, ..., Nconf

            The relevant value is at Ncut = Nconf. The other values are used for the convergence plot in simulation statistics.
        """
        print("Measuring statistical power...")
        ImS_bst = self.actVals.imag[self.k_bst]

        self.NconfCuts = np.arange(int(0.1*self.Nconf),self.Nconf, int(0.1*self.Nconf) )

        self.statistical_power_bst = np.zeros( (self.Nbst, len(self.NconfCuts)) )
        for i,N in enumerate(self.NconfCuts):
            self.statistical_power_bst[:,i] = np.abs(np.exp(-1j*ImS_bst[:,:N]).mean(axis=1))

        self.statistical_power_est = self.statistical_power_bst.mean(axis=0)
        self.statistical_power_err = self.statistical_power_bst.std(axis=0)
    
    def measure_correlator(self):
        r"""
        """
        def diagonalize( C ):
            r"""
                This diagonalization is based on the jacobi method
                where the largest off diagonal element is rotated into the diagonal iteratively
            """
            # We start by block diagonalizing the correlator using the symmetry of the system
            C_block = amplitudes[None,None,:,:] @ C.real @ amplitudes.T.conj()[None,None,:,:] # shape = (Nbst,Nt,Nx,Nx)
            
            C_block = np.diagonal(C_block, axis1=2, axis2=3)

            return C_block, C_block.mean(axis=0), C_block.std(axis=0)
        # end def diagonalize

        def measure( C, Csh = None ):
            r"""
                We measure the raw correlator by:
                0. Averaging Particles and Holes
                1. Bootstrapping and Reweighting
                2. Diagonalizing the correlator 
                3. Symmetrizing (Cs(t) = 0.5*(C(t) + C(Nt-t)))
                4. Covariance mean, standard deviation & Covariance for C and Cs over bootstrap
            """

            # The correlators are hermitian, we make sure that is true per configuration
            if Csh is not None:
                print("\t 0. Averaging Holes and Particles...")
                # (Nconf, Nt, Nx,Nx)
                #C[:,1:,:,:] += Csh[:,1:,:,:][:,::-1,:,:].conj().transpose((0,1,3,2))
                #C[:,1:,:,:] /= 2
                C[:,1:,:,:] += np.flip(Csh[:,1:,:,:], axis=1)
                C[:,1:,:,:] /= 2

            # 1. bootstrap the data
            print("\t 1. Bootstraping & Reweighting...")
            C_bst = np.zeros( (self.Nbst,self.Nt,self.Nx,self.Nx), dtype=complex )
            for nbst in tqdm(range(self.Nbst)):
                C_bst[nbst] = np.mean(
                    C[self.k_bst[nbst]] * self.weights[self.k_bst[nbst],None,None,None],
                    axis = 0
                ) / np.mean(self.weights[self.k_bst[nbst]], axis = 0)  

            # diagonalize the correlator and compute the bootstrap mean and standard deviation
            # C_bst.shape = (Nbst,Nt,Nx)
            print("\t 3. Diagonalizing...")
            C_bst, C_est, C_err = diagonalize( C_bst )

            # finally average and estimate uncertainty/covariance
            print("\t 4. Covariance...")

            # estimate covariance for the fitting
            C_cov = np.zeros( (self.Nt,self.Nt,self.Nx), dtype=complex )
            for k in range(self.Nx):
                C_cov[:,:,k]  = np.cov(C_bst[:,:,k], rowvar=False)

            return C.real, C_bst.real, C_est.real, C_err, C_cov.real
        # measure

        print("Measuring C_sp+C_sh[::-1]...")
        self.C_sp, self.C_sp_bst, self.C_sp_est, self.C_sp_err, self.C_sp_cov = measure(self.C_sp.copy(), self.C_sh)

        # we need these for the charge
        #print("Measuring C_sh...")
        #self.C_sh, self.C_sh_bst, self.C_sh_est, self.C_sh_err, self.C_sh_cov = measure(self.C_sh)
    # end measure_correlator()

    def __post_init__(self):
        self.read_data()

        self.delta = self.beta/self.Nt

        self.measure_autocorrelation()

        self.reduce_raw_data()

        if self.k_bst is None:
            self.k_bst = np.random.randint(0,self.Nconf, size=(self.Nbst, self.Nconf))
        else:
            Nbst_,Nconf_ = self.k_bst.shape
            if Nbst_ != self.Nbst or Nconf != self.Nconf:
                raise RuntimeError( "Passed k_bst does not align with Nbst: {Nbst_} (k_bst) != {self.Nbst} (given) or Nconf: {Nconf_} (given) != {self.Nconf} " )

        self.reweighting_weights()

        self.measure_statistical_power()

        self.measure_correlator()
