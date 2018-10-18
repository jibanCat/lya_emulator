"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import shutil
import glob
import string
import math
import json
import numpy as np
import h5py
from SimulationRunner import lyasimulation
import latin_hypercube
import flux_power
import matter_power
import lyman_data
import gpemulator
from mean_flux import ConstMeanFlux

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'As': r'A_\mathrm{s}', 'heat_slope': r'H_\mathrm{S}', 'heat_amp': r'H_\mathrm{A}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0'}
    try:
        return print_names[key]
    except KeyError:
        return key

class Emulator(object):
    """Small wrapper class to store parameter names and limits, generate simulations and get an emulator.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None, mf=None, nuvb=10):
        if param_names is None:
            self.param_names = {'ns':0, 'As':1, 'heat_slope':2, 'heat_amp':3, 'hub':4}
        else:
            self.param_names = param_names
        if param_limits is None:
            self.param_limits = np.array([[0.8, 1.1], [1.4e-9, 2.6e-9], [-0.5, 0.5],[0.3,1.8],[0.62,0.75]])
        else:
            self.param_limits = param_limits
        if kf is None:
            self.kf = lyman_data.BOSSData().get_kf()
        else:
            self.kf = kf
        if mf is None:
            self.mf = ConstMeanFlux(None)
        else:
            self.mf = mf
        self.dense_param_names = {'uvb':0 }
        self.nuvb = nuvb
        self.dense_param_limits = np.array([[1, 1]])
        self.photo_factors = np.array([1.])
        #We fix omega_m h^2 = 0.1199 (Planck best-fit) and vary omega_m and h^2 to match it.
        #h^2 itself has little effect on the forest.
        self.omegamh2 = 0.1199
        #Corresponds to omega_m = (0.23, 0.31) which should be enough.

        self.set_maxk()

        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def set_maxk(self):
        """Get the maximum k in Mpc/h that we will need."""
        #Corresponds to omega_m = (0.23, 0.31) which should be enough.

        #Maximal velfactor: the h dependence cancels but there is an omegam
        minhub = self.param_limits[self.param_names['hub'],0]
        velfac = lambda a: a * 100.0* np.sqrt(self.omegamh2/minhub**2/a**3 + (1 - self.omegamh2/minhub))
        #Maximum k value to use in comoving Mpc/h.
        #Comes out to k ~ 5, which is a bit larger than strictly necessary.
        self.maxk = np.max(self.kf) * velfac(1/(1+4.4)) * 2


    def build_dirname(self,params, include_dense=False, strsz=3):
        """Make a directory name for a given set of parameter values"""
        ndense = include_dense * len(self.dense_param_names)
        parts = ['',]*(len(self.param_names) + ndense)
        #Transform the dictionary into a list of string parts,
        #sorted in the same way as the parameter array.
        fstr = "%."+str(strsz)+"g"
        for nn,val in self.dense_param_names.items():
            parts[val] = nn+fstr % params[val]
        for nn,val in self.param_names.items():
            parts[ndense+val] = nn+fstr % params[ndense+val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def print_pnames(self):
        """Get parameter names for printing"""
        n_latex = []
        sort_names = sorted(list(self.dense_param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        sort_names = sorted(list(self.param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        return n_latex

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def _recon_one(self, pdir):
        """Get the parameters of a simulation from the SimulationICs.json file"""
        with open(os.path.join(pdir, "SimulationICs.json"), 'r') as jsin:
            sics = json.load(jsin)
        ev = np.zeros_like(self.param_limits[:,0])
        pn = self.param_names
        ev[pn['heat_slope']] = sics["rescale_slope"]
        ev[pn['heat_amp']] = sics["rescale_amp"]
        ev[pn['hub']] = sics["hubble"]
        ev[pn['ns']] = sics["ns"]
        wmap = sics["scalar_amp"]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        conv = (0.05/(2*math.pi/8.))**(sics["ns"]-1.)
        ev[pn['As']] = wmap / conv
        return ev

    def reconstruct(self):
        """Reconstruct the parameters of an emulator by loading the parameters of each simulation in turn."""
        dirs = glob.glob(os.path.join(self.basedir, "*"))
        self.sample_params = np.array([self._recon_one(pdir) for pdir in dirs])
        assert np.shape(self.sample_params) == (len(dirs), np.size(self.param_limits[:,0]))

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Backup existing parameter file
        fdump = os.path.join(self.basedir, dumpfile)
        if os.path.exists(fdump):
            backup = fdump + ".backup"
            r=1
            while os.path.exists(backup):
                backup = fdump + "_r"+str(r)+".backup"
                r+=1
            shutil.move(fdump, backup)
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(fdump, 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        save_list = {"kf", "basedir", "dense_param_names", "dense_param_limits", "photo_factors", "nuvb", "mf"}
        savedict = self.__dict__
        with open(os.path.join(self.basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        for ent in save_list:
            self.__dict__[ent] = savedict[ent]
        self.set_maxk()

    def get_outdir(self, pp, strsz=3):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp, strsz=strsz)),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples,limits = None, use_existing=False):
        """Build a list of directories and parameters from a hypercube sample"""
        if limits is None:
            limits = self.param_limits
        #Consider only prior points inside the limits
        prior_points = None
        if use_existing:
            ii = np.where(np.all(self.sample_params > limits[:,0],axis=1)*np.all(self.sample_params < limits[:,1],axis=1))
            prior_points = self.sample_params[ii]
        return latin_hypercube.get_hypercube_samples(limits, nsamples,prior_points=prior_points)

    def gen_simulations(self, nsamples, npart=256.,box=40,samples=None):
        """Initialise the emulator by generating simulations for various parameters."""
        if np.size(self.sample_params) == 0:
            self.sample_params = self.build_params(nsamples)
        if samples is None:
            samples = self.sample_params
        else:
            self.sample_params = np.vstack([self.sample_params, samples])
        #Generate ICs for each set of parameter inputs
        for ev in samples:
            self._do_ic_generation(ev, npart, box)
        self.dump()

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        rescale_slope = ev[pn['heat_slope']]
        rescale_amp = ev[pn['heat_amp']]
        hub = ev[pn['hub']]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        ns = ev[pn['ns']]
        wmap = (0.05/(2*math.pi/8.))**(ns-1.) * ev[pn['As']]
        ss = lyasimulation.LymanAlphaSim(outdir=outdir, box=box,npart=npart, ns=ns, scalar_amp=wmap, rescale_gamma=True, rescale_slope = rescale_slope, redend=2.2, rescale_amp = rescale_amp, hubble=hub, omega0=self.omegamh2/hub**2, omegab=0.0483,unitary=False)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

    def get_param_limits(self, include_dense=True):
        """Get the reprocessed limits on the parameters for the likelihood."""
        if not include_dense:
            return self.param_limits
        plimits = np.vstack([self.dense_param_limits, self.param_limits])
        assert np.shape(plimits)[1] == 2
        return plimits

    def get_nsample_params(self):
        """Get the number of sparse parameters, those sampled by simulations."""
        return np.shape(self.param_limits)[0]

    def _get_fv(self, pp,myspec, photo_factors=1.):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp, strsz=3)
        if not os.path.exists(di):
            di = self.get_outdir(pp, strsz=2)
        powerspectra = myspec.get_snapshot_list(params=pp, base=di, photo_factors=photo_factors)
        return powerspectra

    def get_emulator(self, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=gpemulator.MultiBinGP, max_z=max_z)
        return gp

    def get_uvb_factor_range(self, max_z=4.2):
        """Get the maximal desired UVB range"""
        myspec = flux_power.MySpectra(max_z=max_z, max_k=self.maxk)
        #Get a power spectrum with the default UVB.
        powers = np.ravel([self._get_fv(pp, myspec, photo_factors = 1.) for pp in self.get_parameters()])
        #Work out the largest and smallest needed mean flux.
        mean_fluxes = self.mf.get_mean_flux(myspec.zout)
        if mean_fluxes[0] is None:
            return np.array([1., 1.])
        uvbs = np.vstack([pow.get_uvb_range(mean_fluxes) for pow in powers])
        maxuvb = np.max(uvbs[:,1])
        minuvb = np.min(uvbs[:,0])
        assert maxuvb >= minuvb
        return np.array([minuvb, maxuvb])

    def get_flux_vectors(self, max_z=4.2, kfunits="kms"):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        assert nparams == len(self.param_names)
        myspec = flux_power.MySpectra(max_z=max_z, max_k=self.maxk)
        try:
            aparams = np.array([np.concatenate([[phf,], pv]) for phf in self.photo_factors for pv in pvals])
            kfmpc, kfkms, flux_vectors = self.load_flux_vectors(aparams)
        except (AssertionError, OSError):
            powers = np.ravel([self._get_fv(pp, myspec, photo_factors = self.photo_factors) for pp in pvals])
            flux_vectors = np.array([ps.get_power_native_binning(mean_fluxes = None) for ps in powers])
            #Get the parameters back again, with the mean flux added
            aparams = np.array([ps.get_params() for ps in powers])
            #'natively' binned k values in km/s units as a function of redshift
            kfkms = [ps.get_kf_kms() for ps in powers]
            #Same in all boxes
            kfmpc = powers[0].kf
            assert np.all(np.abs(powers[0].kf/ powers[-1].kf-1) < 1e-6)
            self.save_flux_vectors(aparams, kfmpc, kfkms, flux_vectors)
        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]
        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc
        return aparams, kf, flux_vectors

    def save_flux_vectors(self, aparams, kfmpc, kfkms, flux_vectors, savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        save = h5py.File(os.path.join(self.basedir, savefile), 'w')
        save.attrs["classname"] = str(self.__class__)
        save["params"] = aparams
        save["flux_vectors"] = flux_vectors
        #Save in both km/s and Mpc/h units.
        save["kfkms"] = kfkms
        save["kfmpc"] = kfmpc
        save.close()

    def load_flux_vectors(self, aparams, savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        load = h5py.File(os.path.join(self.basedir, savefile), 'r')
        inparams = np.array(load["params"])
        flux_vectors = np.array(load["flux_vectors"])
        kfkms = np.array(load["kfkms"])
        kfmpc = np.array(load["kfmpc"])
        name = str(load.attrs["classname"])
        load.close()
        assert name == str(self.__class__)
        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)
        return kfmpc, kfkms, flux_vectors

    def get_flux_vectors_batch(self, max_z=4.2):
        """Launch a set of batch scripts into the queue to compute the lyman alpha spectra and their flux vectors."""
        pvals = self.get_parameters()
        uvblim = self.get_uvb_factor_range(max_z = max_z)
        nuvb = 1.
        if uvblim[1] > uvblim[0]:
            nuvb = self.nuvb
        self.photo_factors = np.linspace(uvblim[0], uvblim[1], nuvb)
        for pp in pvals:
            di = os.path.join(self.basedir, self.build_dirname(pp, strsz=3))
            if not os.path.exists(di):
                di = os.path.join(self.basedir, self.build_dirname(pp, strsz=2))
            self.batch_script(di, photo=self.photo_factors)

    def batch_script(self, pdir, photo=1.):
        """The batch script to use. For biocluster."""
        fpfile = os.path.join(os.path.dirname(__file__),"flux_power.py")
        shutil.copy(fpfile, os.path.join(pdir, "flux_power.py"))
        phfstr = ''.join("%g " % i for i in photo)
        with open(os.path.join(pdir, "spectra_submit"),'w') as submit:
            submit.write("""#!/bin/bash\n#SBATCH --partition=short\n#SBATCH --job-name="""+pdir+"\n")
            submit.write("""#SBATCH --time=1:55:00\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --cpus-per-task=32\n#SBATCH --mem-per-cpu=4G\n""")
            submit.write( """#SBATCH --mail-type=end\n#SBATCH --mail-user=sbird@ucr.edu\n""")
            submit.write("python flux_power.py "+pdir+"/output --phf "+phfstr+"\n")

    def _get_custom_emulator(self, *, emuobj, max_z=4.2):
        """Helper to allow supporting different emulators."""
        uvblim = self.get_uvb_factor_range(max_z = max_z)
        nuvb = 1.
        if uvblim[1] > uvblim[0]:
            nuvb = self.nuvb
        self.photo_factors = np.linspace(uvblim[0], uvblim[1], nuvb)
        self.dense_param_limits[self.dense_param_names['uvb']] = uvblim
        aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, kfunits="mpc")
        plimits = self.get_param_limits(include_dense=True)
        gp = emuobj(params=aparams, kf=kf, powers = flux_vectors, param_limits = plimits)
        return gp

class KnotEmulator(Emulator):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir, nknots=4, kf=None, mf=None):
        param_names = {'heat_slope':nknots, 'heat_amp':nknots+1, 'hub':nknots+2}
        #Assign names like AA, BB, etc.
        for i in range(nknots):
            param_names[string.ascii_uppercase[i]*2] = i
        self.nknots = nknots
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),nknots,axis=0),[[-0.5, 0.5],[0.5,1.5],[0.65,0.75]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits, kf=kf, mf=mf)
        #Linearly spaced knots in k space:
        #these do not quite hit the edges of the forest region, because we want some coverage over them.
        self.knot_pos = np.linspace(0.15, 1.5,nknots)
        #Used for early iterations.
        #self.knot_pos = [0.15,0.475,0.75,1.19]

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        rescale_slope = ev[pn['heat_slope']]
        rescale_amp = ev[pn['heat_amp']]
        hub = ev[pn['hub']]
        ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_pos = self.knot_pos, knot_val=ev[0:self.nknots],hubble=hub, rescale_gamma=True, redend=2.2, rescale_slope = rescale_slope, rescale_amp = rescale_amp, omega0=self.omegamh2/hub**2, omegab=0.0483,unitary=False)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

class MatterPowerEmulator(Emulator):
    """Build an emulator based on the matter power spectrum instead of the flux power spectrum, for testing."""
    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile. Reset the k values to something sensible for matter power."""
        super().load(dumpfile=dumpfile)
        self.kf = np.logspace(np.log10(3*math.pi/60.),np.log10(2*math.pi/60.*256),20)

    def _get_fv(self, pp,myspec, photo_factors=None):
        """Helper function to get a single matter power vector."""
        di = self.get_outdir(pp)
        (_,_) = myspec
        fv = matter_power.get_matter_power(di,kk=self.kf, redshift = 3.)
        return fv
