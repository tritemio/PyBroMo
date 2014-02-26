# encoding: utf-8  μ αβγδ εζηθ κλμν ξοπρ ςστυ φχψω
"""
PyBroMo - A single molecule diffusion simulator in confocal geometry.

Copyright (C) 2013-2014 Antonino Ingargiola tritemio@gmail.com

This is the main module of PyBroMo. Import (or run) it to perform a simulation.
"""

import os
import cPickle as pickle
from glob import glob
from itertools import izip
import hashlib

import numpy.random as NR
import numpy as np
from numpy import array, arange, sqrt
import matplotlib.pyplot as plt
import tables

from path_def import *
from psflib import GaussianPSF, NumericPSF
from scroll_gui import ScrollingToolQT
import loadutils as lu
from storage import Storage
from iter_chunks import iter_chunksize, iter_chunk_index

## Avogadro constant
NA = 6.022141e23    # [mol^-1]


class Box:
    """The simulation box"""
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2
        self.b = array([[x1, x2], [y1, y2], [z1, z2]])

    def volume(self):
        """Box volume in m^3."""
        return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

    def volume_L(self):
        """Box volume in liters."""
        return self.volume()*1e3

    def __repr__(self):
        return u"Box: X %.1fum, Y %.1fum, Z %.1fum" % (
                (self.x2 - self.x1)*1e6,
                (self.y2 - self.y1)*1e6,
                (self.z2 - self.z1)*1e6)


class Particle:
    """Class to describe a single particle"""
    def __init__(self, x0=0, y0=0, z0=0):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.r0 = array([x0, y0, z0])


def wrap_periodic(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply periodic boundary contitions.
    """
    a -= a1
    wrapped = np.mod(a, a2-a1) + a1
    return wrapped


class ParticlesSimulation(object):
    """Class that performs the Brownian motion simulation of N particles.
    """
    def __init__(self, D, t_step, t_max, particles, box, psf, EID=0, ID=0):
        """Initialize the simulation parameters:
        `D`: diffusion coefficient (m/s^2)
        `t_step`: time step (s)
        `particles`: list of `Particle` objects
        `box`: a `Box` object defining the simulation boundaries
        `psf`: a "PSF" object (`GaussianPSF` or `NumericPSF`) defining the PSF
        `EID`: is an ID that identifies the engine on which the simulation
            runs. It's a way to distinguish simulations that may otherwise
            appear identical.
        `ID`: is a number that identify the simulation, for ex. if you run
            the same multiple times on the same engine you can assign different
            ID.
        The EID and ID are shown in the string representation and are used
        to save unique file names.
        """
        self.particles = particles
        self.box = box
        self.psf = psf
        
        self.D = D
        self.np = len(particles)
        self.t_step = t_step
        self.t_max = t_max
        self.ID = ID
        self.EID = EID
        
        self.n_samples = int(t_max/t_step)
        self.sigma = sqrt(2*D*3*t_step)

    def __repr__(self):
        pM = self.concentration(pM=True)
        s = repr(self.box)
        s += "\nD %.2g, #Particles %d, %.1f pM, t_step %.1fus, t_max %.1fs" %\
                (self.D, self.np, pM, self.t_step*1e6, self.t_max)
        s += " EID_ID %d %d" % (self.EID, self.ID)
        return s
    
    def hash(self):
        """Return an hash for the simulation parameters (excluding ID and EID)
        This can be used to generate unique file names for simulations
        that have the same parameters and just different ID or EID.
        """
        hash_numeric = 'D=%s, t_step=%s, t_max=%s, np=%s' % \
                (self.D, self.t_step, self.t_max, self.np)
        hash_list = [hash_numeric, repr(self.box), self.psf.hash()]
        return hashlib.md5(repr(hash_list)).hexdigest()

    def compact_name_core(self, hashdigits=6, t_max=False):
        """Compact representation of simulation params (no ID, EID and t_max)
        """
        Moles = self.concentration()
        name = "D%.2g_%dP_%dpM_step%.1fus" % (
                self.D, self.np, Moles*1e12, self.t_step*1e6)
        if hashdigits > 0:
            name = self.hash()[:hashdigits] + '_' + name
        if t_max:
            name += "_t_max%.1fs" % self.t_max
        return name

    def compact_name(self, hashdigits=6):
        """Compact representation of all simulation parameters
        """
        # this can be made more robust for ID > 9 (double digit)
        s = self.compact_name_core(hashdigits, t_max=True)
        s += "_ID%d-%d" % (self.EID, self.ID)
        return s
    
    def get_nparams(self):
        """Return a dict containing all the simulation numeric-parameters.

        The values are 2-element tuples: first element is the value and
        second element is a string describing the parameter (metadata).
        """
        nparams = dict(
            D = (self.D, 'Diffusion coefficient (m^2/s)'),
            np = (self.np, 'Number of simulated particles'),
            t_step = (self.t_step, 'Simulation time-step (s)'),
            t_max = (self.t_max, 'Simulation total time (s)'),
            ID = (self.ID, 'Simulation ID (int)'),
            EID = (self.EID, 'IPython Engine ID (int)'),
            pico_mol = (self.concentration()*1e12,
                        'Particles concentration (pM)')
            )
        return nparams

    def print_RAM(self):
        """Print RAM needed to simulate the current set of parameters."""
        float_size = 4
        MB = 1024*1024
        size_ = (self.n_samples*float_size)
        print "  Number of particles:", self.np
        print "  Number of time steps:", self.n_samples
        print "  Emission array - 1 particle (float32): %.1f MB" % (size_/MB)
        print "  Emission array (float32): %.1f MB" % (size_*self.np/MB)
        print "  Position array (float32): %.1f MB " % (3*size_*self.np/MB)

    def concentration(self, pM=False):
        """Return the concentration (in Moles) of the particles in the box.
        """
        concentr = (self.np/NA)/self.box.volume_L()
        if pM: concentr *= 1e12
        return concentr

    def reopen_store(self):
        """Reopen a closed store in read-only mode."""
        self.store.open()
        self.psf_pytables = psf_pytables
        self.emission = S.store.data_file.root.trajectories.emission
        self.emission_tot = S.store.data_file.root.trajectories.emission_tot
        self.chunksize = S.store.data_file.get_node('/parameters', 'chunksize')

    def open_store(self, prefix='pybromo_', chunksize=2**18, overwrite=True,
                   comp_filter=None):
        nparams = self.get_nparams()
        self.chunksize = chunksize
        nparams.update(chunksize=(chunksize, 'Chunksize for arrays'))     
        self.store_fname = prefix + self.compact_name() + '.hdf5'
        
        attr_params = dict(particles=self.particles, box=self.box)
        self.store = Storage(self.store_fname, nparams=nparams, 
                             attr_params=attr_params, overwrite=overwrite)
      
        self.psf_pytables = self.psf.to_hdf5(self.store.data_file, '/psf')
        self.store.data_file.create_hard_link('/psf', 'default_psf', 
                                              target=self.psf_pytables)

        kwargs = dict(chunksize=self.chunksize,
                      params=dict(psf_fname=self.psf.fname))
                      # Note psf.fname is the psf name in `data_file.root.psf`
        if comp_filter is not None: 
            kwargs.update(comp_filter=comp_filter)
        self.emission_tot = self.store.add_emission_tot(**kwargs)
        self.emission = self.store.add_emission(**kwargs)

    def sim_motion_em_chunk(self, delete_pos=True, total_emission=True, 
                            seed=1):
        """Simulate Brownian motion and emission rates in one step.
        This method simulates sequentially one particle a time (uses less RAM).
        `delete_pos` allows to discard the particle trajectories and save only
                the emission.
        `total_emission` choose to save a single emission array for all the
                particles (if True), or save the emission of each single
                particle (if False). In the latter case `.em` will be a 2D
                array (#particles x time). Otherwise `.em` is (1 x time).
        """
        if 'store' not in self.__dict__:
            self.open_store()
        em_store = self.emission_tot if total_emission else self.emission
        
        if seed is not None:
            np.random.seed(seed)
        print '[PID %d] Simulation chunk:' % os.getpid(),
        i_chunk = 0
        t_chunk_size = self.emission.chunkshape[1]
        for c_size in iter_chunksize(self.n_samples, t_chunk_size):
            print i_chunk, c_size,          
            if total_emission:
                em = np.zeros((c_size), dtype=np.float32)
            else:
                em = np.zeros((self.np, c_size), dtype=np.float32)
            POS = []
            
            for i, p in enumerate(self.particles):
                delta_pos = NR.normal(loc=0, scale=self.sigma, size=3*c_size)
                delta_pos = delta_pos.reshape(3, c_size)
                pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
                pos += p.r0.reshape(3, 1)
                # Coordinates wrapping using periodic boundary conditions
                for coord in (0, 1, 2):
                    pos[coord] = wrap_periodic(pos[coord], *self.box.b[coord])

                # Sample the PSF along i-th trajectory then square to account
                # for emission and detection PSF.
                Ro = sqrt(pos[0]**2 + pos[1]**2)  # radial pos. on x-y plane
                Z = pos[2]
                current_em = self.psf.eval_xz(Ro, Z)**2
                if total_emission:
                    # Add the current particle emission to the total emission
                    em += current_em.astype(np.float32)
                else:
                    # Store the individual emission of current particle
                    em[i] = current_em.astype(np.float32)
                if not delete_pos: POS.append(pos.reshape(1, 3, n_samples))

            ## Append em to the permanent storage
            # if total_emission is just a linear array
            # otherwise is an hstack of what is saved and em (self.np, c_size)
            em_store.append(em)
            if not delete_pos: self.pos = np.concatenate(POS)
            i_chunk += 1
        em_store.set_attr('seed', seed)
        em_store.flush()
    
    def sim_timestamps_em_list(self, max_rate=1, bg_rate=0, seed=1):
        """Compute timestamps and particles and store results in a list.
        Each element contains timestamps from one chunk of emission.
        Background computed internally.       
        """
        if seed is not None: np.random.seed(seed)
        fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
        scale = 10
        max_counts = 4
        
        self.all_times_chunks_list = []
        self.all_par_chunks_list = []
        
        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples, 
                                               self.emission.chunkshape[1]):
            counts_chunk = sim_timetrace(self.emission[:, i_start:i_end], 
                                         max_rate, self.t_step)
            counts_bg_chunk = NR.poisson(bg_rate*self.t_step, 
                                         size=counts_chunk.shape[1]
                                         ).astype('uint8')
            index = np.arange(0, counts_chunk.shape[1])
            
            # Loop for each particle to compute timestamps
            times_chunk_p = []      # <-- Try preallocating array 
            par_index_chunk_p = []  # <-- Try preallocating array 
            for p_i, counts_chunk_p_i in enumerate(counts_chunk.copy()):
                # Compute timestamps for paricle p_i for all bins with counts
                times_c_i = [(index[counts_chunk_p_i >= 1] + i_start)*scale]
                # Additional timestamps for bins with counts > 1
                for frac, v in izip(fractions, range(2, max_counts + 1)):
                    times_c_i.append(
                        (index[counts_chunk_p_i >= v] + i_start)*scale + frac
                        )
                
                # Stack the arrays from different "counts"
                t = np.hstack(times_c_i)
                times_chunk_p.append(t)
                par_index_chunk_p.append(np.full(t.size, p_i, dtype='u1'))
            
            # Simulate background for current chunk
            time_chunk_bg = (index[counts_bg_chunk >= 1] + i_start)*scale
            times_chunk_p.append(time_chunk_bg)
            par_index_chunk_p.append(np.full(time_chunk_bg.size, p_i+1, 
                                             dtype='u1'))
            
            # Merge the arrays of different particles
            times_chunk_s = np.hstack(times_chunk_p)  # <-- Try preallocating
            par_index_chunk_s = np.hstack(par_index_chunk_p)  # <-- this too
            
            # Sort timestamps inside the merged chunk
            index_sort = times_chunk_s.argsort(kind='mergesort')
            times_chunk_s = times_chunk_s[index_sort]
            par_index_chunk_s = par_index_chunk_s[index_sort]
    
            # Save (ordered) timestamps and corresponding particles
            self.all_times_chunks_list.append(times_chunk_s)
            self.all_par_chunks_list.append(par_index_chunk_s)
        
    def sim_timestamps_em_list1(self, max_rate=1, bg_rate=0, seed=None):
        """Compute timestamps and particles and store results in a list.
        Each element contains timestamps from one chunk of emission.
        Background computed in sim_timetrace_bg() as last fake particle.        
        """
        if seed is not None: np.random.seed(seed)
        fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
        scale = 10
        max_counts = 4
        
        self.all_times_chunks_list = []
        self.all_par_chunks_list = []
        
        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples, 
                                               self.emission.chunkshape[1]):
            counts_chunk = sim_timetrace_bg(self.emission[:, i_start:i_end], 
                                         max_rate, bg_rate, self.t_step)
            index = np.arange(0, counts_chunk.shape[1])
            
            # Loop for each particle to compute timestamps
            times_chunk_p = []      # <-- Try preallocating array 
            par_index_chunk_p = []  # <-- Try preallocating array 
            for p_i, counts_chunk_p_i in enumerate(counts_chunk.copy()):
                # Compute timestamps for paricle p_i for all bins with counts
                times_c_i = [(index[counts_chunk_p_i >= 1] + i_start)*scale]
                # Additional timestamps for bins with counts > 1
                for frac, v in izip(fractions, range(2, max_counts + 1)):
                    times_c_i.append(
                        (index[counts_chunk_p_i >= v] + i_start)*scale + frac
                        )
                
                # Stack the arrays from different "counts"
                t = np.hstack(times_c_i)
                times_chunk_p.append(t)
                par_index_chunk_p.append(np.full(t.size, p_i, dtype='u1'))
            
            # Merge the arrays of different particles
            times_chunk_s = np.hstack(times_chunk_p)  # <-- Try preallocating
            par_index_chunk_s = np.hstack(par_index_chunk_p)  # <-- this too
            
            # Sort timestamps inside the merged chunk
            index_sort = times_chunk_s.argsort(kind='mergesort')
            times_chunk_s = times_chunk_s[index_sort]
            par_index_chunk_s = par_index_chunk_s[index_sort]
    
            # Save (ordered) timestamps and corresponding particles
            self.all_times_chunks_list.append(times_chunk_s)
            self.all_par_chunks_list.append(par_index_chunk_s)
   
    def sim_timestamps_em_list2(self, max_rate=1, bg_rate=0, seed=None):
        """Compute timestamps and particles and store results in a list.
        Each element contains timestamps from one chunk of emission.
        Background computed in sim_timetrace_bg2() as last fake particle.        
        """
        if seed is not None: np.random.seed(seed)        
        fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
        scale = 10
        max_counts = 4
        
        self.all_times_chunks_list = []
        self.all_par_chunks_list = []
        
        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples, 
                                               self.emission.chunkshape[1]):
            counts_chunk = sim_timetrace_bg2(self.emission[:, i_start:i_end], 
                                         max_rate, bg_rate, self.t_step)
            index = np.arange(0, counts_chunk.shape[1])
            
            # Loop for each particle to compute timestamps
            times_chunk_p = []      # <-- Try preallocating array 
            par_index_chunk_p = []  # <-- Try preallocating array 
            for p_i, counts_chunk_p_i in enumerate(counts_chunk.copy()):
                # Compute timestamps for paricle p_i for all bins with counts
                times_c_i = [(index[counts_chunk_p_i >= 1] + i_start)*scale]
                # Additional timestamps for bins with counts > 1
                for frac, v in izip(fractions, range(2, max_counts + 1)):
                    times_c_i.append(
                        (index[counts_chunk_p_i >= v] + i_start)*scale + frac
                        )
                
                # Stack the arrays from different "counts"
                t = np.hstack(times_c_i)
                times_chunk_p.append(t)
                par_index_chunk_p.append(np.full(t.size, p_i, dtype='u1'))
            
            # Merge the arrays of different particles
            times_chunk_s = np.hstack(times_chunk_p)  # <-- Try preallocating
            par_index_chunk_s = np.hstack(par_index_chunk_p)  # <-- this too
            
            # Sort timestamps inside the merged chunk
            index_sort = times_chunk_s.argsort(kind='mergesort')
            times_chunk_s = times_chunk_s[index_sort]
            par_index_chunk_s = par_index_chunk_s[index_sort]
    
            # Save (ordered) timestamps and corresponding particles
            self.all_times_chunks_list.append(times_chunk_s)
            self.all_par_chunks_list.append(par_index_chunk_s)
    
    def _get_ts_name(self, max_rate=1, bg_rate=0, seed=1):
        return 'ts_max_rate%dkcps_bg%dcps_seed%s' % \
                            (max_rate*1e-3, bg_rate, seed)

    def sim_timestamps_em_store(self, max_rate=1, bg_rate=0, seed=1,
                                chunksize=2**16, comp_filter=None, 
                                overwrite=False):
        """Compute timestamps and particles and store results in a list.
        Each element contains timestamps from one chunk of emission.
        Background computed in sim_timetrace_bg() as last fake particle.        
        """
        if seed is not None: np.random.seed(seed)
        fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
        scale = 10
        max_counts = 4
        
        self.timestamps, self.tparticles = self.store.add_timestamps(
                        name=self._get_ts_name(max_rate, bg_rate, seed), 
                        clk_p=t_step/scale, 
                        max_rate=max_rate, bg_rate=bg_rate,
                        num_particles=self.np, bg_particle=self.np,
                        overwrite=overwrite, chunksize=chunksize, 
                        comp_filter=comp_filter)
        
        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples, 
                                               self.emission.chunkshape[1]):
            counts_chunk = sim_timetrace_bg(self.emission[:, i_start:i_end], 
                                         max_rate, bg_rate, self.t_step)
            index = np.arange(0, counts_chunk.shape[1])
            
            # Loop for each particle to compute timestamps
            times_chunk_p = []      # <-- Try preallocating array 
            par_index_chunk_p = []  # <-- Try preallocating array 
            for p_i, counts_chunk_p_i in enumerate(counts_chunk.copy()):
                # Compute timestamps for paricle p_i for all bins with counts
                times_c_i = [(index[counts_chunk_p_i >= 1] + i_start)*scale]
                # Additional timestamps for bins with counts > 1
                for frac, v in izip(fractions, range(2, max_counts + 1)):
                    times_c_i.append(
                        (index[counts_chunk_p_i >= v] + i_start)*scale + frac
                        )
                
                # Stack the arrays from different "counts"
                t = np.hstack(times_c_i)
                times_chunk_p.append(t)
                par_index_chunk_p.append(np.full(t.size, p_i, dtype='u1'))
            
            # Merge the arrays of different particles
            times_chunk_s = np.hstack(times_chunk_p)  # <-- Try preallocating
            par_index_chunk_s = np.hstack(par_index_chunk_p)  # <-- this too
            
            # Sort timestamps inside the merged chunk
            index_sort = times_chunk_s.argsort(kind='mergesort')
            times_chunk_s = times_chunk_s[index_sort]
            par_index_chunk_s = par_index_chunk_s[index_sort]
    
            # Save (ordered) timestamps and corresponding particles
            self.timestamps.append(times_chunk_s)
            self.tparticles.append(par_index_chunk_s)
        self.store.data_file.flush()


def sim_timetrace(emission, max_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).    
    """
    emission_rates = emission*max_rate*t_step
    return NR.poisson(lam=emission_rates).astype(np.uint8)

def sim_timetrace_bg(emission, max_rate, bg_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).
    Return an uint8 array of counts with shape[0] == emission.shape[0] + 1.
    The last row is a "fake" particle representing Poisson background.
    """
    em = np.atleast_2d(emission).astype('float64', copy=False)
    counts = np.zeros((em.shape[0] + 1, em.shape[1]), dtype='u1')   
    # In-place computation
    # NOTE: the caller will see the modification
    em *= (max_rate*t_step)
    # Use automatic type conversion int64 -> uint8
    counts[:-1] = NR.poisson(lam=em)
    counts[-1] = NR.poisson(lam=bg_rate*t_step, size=em.shape[1])
    return counts

def sim_timetrace_bg2(emission, max_rate, bg_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).
    Return an uint8 array of counts with shape[0] == emission.shape[0] + 1.
    The last row is a "fake" particle representing Poisson background.
    """
    emiss_bin_rate = np.zeros((emission.shape[0] + 1, emission.shape[1]), 
                              dtype='float64')   
    emiss_bin_rate[:-1] = emission*max_rate*t_step
    emiss_bin_rate[-1] = bg_rate*t_step
    counts = NR.poisson(lam=emiss_bin_rate).astype('uint8')
    return counts

def load_simulation(fname):
    store = Storage(fname, overwrite=False)
    nparams = store.get_sim_nparams()
    
    psf_pytables = store.data_file.get_node('/psf/default_psf')
    psf = NumericPSF(psf_pytables=psf_pytables)
    box = store.data_file.get_node_attr('/parameters', 'box')
    P = store.data_file.get_node_attr('/parameters', 'particles')
    
    names = ['D', 't_step', 't_max', 'EID', 'ID']    
    kwargs = dict()
    for n in names:
        kwargs[n] = nparams[n]    
    S = ParticlesSimulation(particles=P, box=box, psf=psf, **kwargs)
    
    # Emulate S.open_store()    
    S.store = store
    S.store_fname = fname
    S.psf_pytables = psf_pytables
    S.emission = S.store.data_file.root.trajectories.emission
    S.emission_tot = S.store.data_file.root.trajectories.emission_tot
    S.chunksize = S.store.data_file.get_node('/parameters', 'chunksize')
    return S

def load_sim_id(ID, glob_str='*', EID=None, dir_='.', prefix='bromo_sim',
        verbose=False):
    """Try to load a simulation with specified ID.
    """
    # NOTE: if EID is not passed use eid from global scope
    if EID is None: EID = eid
    Sim = None
    try:
        pattern = dir_+'/'+prefix+'%sID%d-%d.pickle' % (glob_str, EID, ID)
        if verbose: print pattern
        fname = glob(pattern)
        if len(fname) > 1: print "ERROR: More than 1 file matched!"
        assert len(fname) == 1
        Sim = pickle.load(open(fname[0], 'rb'))
        print "Loaded:\n", fname[0]
    except:
        print "No matching simulation file found."
    return Sim

##
# Functions to manage/merge multiple simulations
#
def merge_ph_times(ph_times_list, time_block):
    """Build an array of timestamps joining the arrays in `ph_times_list`.
    `time_block` is the duration of each array of timestamps.
    """
    offsets = np.arange(len(ph_times_list))*time_block
    cum_sizes = np.cumsum([ph.size for ph in ph_times_list])
    ph_times = np.zeros(cum_sizes[-1])
    i1 = 0
    for i2, ph, offset in zip(cum_sizes, ph_times_list, offsets):
        ph_times[i1:i2] = ph + offset
        i1 = i2
    return ph_times

def merge_DA_ph_times(ph_times_d, ph_times_a):
    """Returns the merged timestamps of Donor and Acceptor and a bool mask for A
    """
    ph_times = np.hstack([ph_times_d, ph_times_a])
    a_em = np.hstack([np.zeros(ph_times_d.size, dtype=np.bool),
            np.ones(ph_times_a.size, dtype=np.bool)])
    index_sort = ph_times.argsort()
    return ph_times[index_sort], a_em[index_sort]

def merge_particle_emission(SS):
    """Returns a sim object summing the emissions and particles in SS (list).
    """
    # Merge all the particles
    P = reduce(lambda x, y: x+y, [Si.particles for Si in SS])
    s = SS[0]
    S = ParticlesSimulation(D=s.D, t_step=s.t_step, t_max=t_max,
            particles=P, box=s.box, psf=s.psf)
    S.em = np.zeros(s.em.shape, dtype=np.float64)
    for Si in SS:
        S.em += Si.em
    return S

def parallel_gen_timestamps(dview, max_em_rate, bg_rate):
    """Generate timestamps from a set of remote simulations in `dview`.
    Assumes that all the engines have an `S` object already containing
    an emission trace (`S.em`). The "photons" timestamps are generated
    from these emission traces and merged into a single array of timestamps.
    `max_em_rate` and `bg_rate` are passed to `S.sim_timetrace()`.
    """
    dview.execute("S.sim_timetrace(max_em_rate=%f, bg_rate=%f)" % \
            (max_em_rate, bg_rate))
    dview.execute("S.gen_ph_times()")
    PH = dview['S.ph_times']
    # Assuming all t_max equal, just take the first
    t_max = dview['S.t_max'][0]
    t_tot = np.sum(dview['S.t_max'])
    dview.execute("sim_name = S.compact_name_core()")
    # Core names contains no ID or t_max
    sim_name = dview['sim_name'][0]
    ph_times = merge_ph_times(PH, time_block=t_max)
    return ph_times, t_tot, sim_name

##
# Plot simulation functions
#

def plot_tracks(S):
    fig, AX = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.05,
            hspace=0.05)
    plt.suptitle("%.1f ms diffusion" % (S.t_step*S.n_samples*1e3))

    for ip in range(S.np):
        x, y, z = S.pos[ip]
        x0, y0, z0 = S.particles[ip].r0
        plot_kwargs = dict(ls='', marker='o', mew=0, ms=1, alpha=0.2)
        l, = AX[1].plot(x*1e6, y*1e6, **plot_kwargs)
        AX[1].plot([x0*1e6], [y0*1e6], 'o', color=l.get_color())
        AX[0].plot(x*1e6, z*1e6, color=l.get_color(), **plot_kwargs)
        AX[0].plot([x0*1e6], [z0*1e6], 'o', color=l.get_color())

    AX[1].set_ylabel("y (um)")
    AX[1].set_xlabel("x (um)")
    AX[0].set_ylabel("z (um)")
    AX[0].grid(True); AX[1].grid(True)

    if S.psf.name == 'gauss':
        sig = S.psf.s
    else:
        sig = array([0.1, 0.1, 0.3])*1e-6
    ## Draw an outline of the PSF
    a = arange(360)/360.*2*np.pi
    rx, ry, rz = (sig)  # draw radius at 3 sigma
    AX[1].plot((rx*np.cos(a))*1e6, (ry*np.sin(a))*1e6, lw=2, color='k')
    AX[0].plot((rx*np.cos(a))*1e6, (rz*np.sin(a))*1e6, lw=2, color='k')

def plot_emission(S, dec=1, scroll_gui=False, multi=False, ms=False):
    fig = plt.figure()
    plt.title("%d Particles, %.1f s diffusion, %d pM" % (S.np,
            S.t_step*S.n_samples, S.concentration()*1e12))
    plt.xlabel("Time (s)"); plt.ylabel("Emission rate [A.U.]"); plt.grid(1)
    if ms: plt.xlabel("Time (ms)")
    if multi:
        for em in S.em:
            time = S.time(dec=dec) if not ms else S.time(dec=dec)*1e3
            plt.plot(time, em[::dec], alpha=0.5)
    else:
        time = S.time(dec=dec) if not ms else S.time(dec=dec)*1e3
        plt.plot(time, S.em.sum(axis=0)[::dec], alpha=0.5)
    s = None
    if scroll_gui: s = ScrollingToolQT(fig)
    return s

def plot_timetrace(S, rebin=2e3, scroll_gui=False):
    fig = plt.figure()
    plt.title("%d Particles, %.1f s diffusion, %d pM, bin=%.1fms" % (S.np,
            S.t_step*S.n_samples, S.concentration()*1e12, S.t_step*rebin*1e3))
    plt.xlabel("Time (s)"); plt.ylabel("Emission rate [A.U.]"); plt.grid(1)
    trace = S.tt.reshape(-1, rebin).sum(axis=1)
    t_trace = (arange(trace.size)+1)*(S.t_step*rebin)
    plt.plot(S.time(), S.em.sum(axis=0)*100, alpha=0.4)
    plt.plot(t_trace, trace, color='k', alpha=0.7, drawstyle='steps')
    if scroll_gui: return ScrollingToolQT(fig)


def gen_particles(N, box, seed=None):
    """Generate `N` Particle() objects with random position in `box`.
    """
    if seed is not None:
        np.random.seed(seed)
    X0 = NR.rand(N)*(box.x2-box.x1) + box.x1
    Y0 = NR.rand(N)*(box.y2-box.y1) + box.y1
    Z0 = NR.rand(N)*(box.z2-box.z1) + box.z1
    return [Particle(x0=x0, y0=y0, z0=z0) for x0, y0, z0 in zip(X0, Y0, Z0)]


if __name__ == '__main__':
    # Simulation time step
    t_step = 0.5e-6     # seconds

    # Diffusion coefficient
    Du = 12.0           # um^2 / s
    D = Du*(1e-6)**2

    # Time duration of the simulation
    t_max = 0.3        # seconds
    n_samples = int(t_max/t_step)

    # PSF definition
    #ss = 0.2*1e-6      # lateral dimension (sigma)
    #psf = GaussianPSF(xc=0, yc=0, zc=0, sx=ss, sy=ss, sz=3*ss)
    psf = NumericPSF()

    # Box definition
    box = Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)

    # Particles definition
    #p1 = Particle(x0=-3e-6)
    #p2 = Particle(x0=3e-6)
    #p3 = Particle(y0=-3e-6)
    #p4 = Particle(y0=3e-6)
    #P = [p1,p2,p3,p4]
    P = gen_particles(15, box)

    # Brownian motion and emission simulation
    S = ParticlesSimulation(D=D, t_step=t_step, t_max=t_max,
                            particles=P, box=box, psf=psf)
    #S.sim_motion_em(delete_pos=False)
    #S.sim_timetrace(max_em_rate=3e5, bg_rate=10e3)
    #S.gen_ph_times()

    #plot_tracks(S)
    #plot_emission(S)

