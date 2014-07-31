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
from numpy import array, sqrt
import tables

from path_def import *
from psflib import GaussianPSF, NumericPSF
import loadutils as lu
from storage import Storage
from iter_chunks import iter_chunksize, iter_chunk_index
import brownian_plot as bpl
import utils.hdf5


## Avogadro constant
NA = 6.022141e23    # [mol^-1]


def get_seed(seed, ID=0, EID=0):
    """Get a random seed that is a combination of `seed`, `ID` and `EID`.
    Provides different, but deterministic, seeds in parallel computations
    """
    return seed + EID + 100*ID

hash_  = lambda x: hashlib.sha1(repr(x)).hexdigest()

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


class Particles(list):
    """Custom list containing many Particle()"""
    def __init__(self, init_list=None, init_random_state=None):
        super(Particles, self).__init__(init_list)
        self.init_random_state = init_random_state
        self.rs_hash = hash_(init_random_state)[:3]

def gen_particles(N, box, rs=None, seed=1):
    """Generate `N` Particle() objects with random position in `box`.

    Arguments:
        N (int): number of particles to be generated
        box (Box object): the simulation box
        rs (RandomState object): random state object used as random number
            generator. If None, use a random state initialized from seed.
        seed (uint): when `rs` is None, `seed` is used to initialize the
            random state, otherwise is ignored.
    """
    if rs is None:
        rs = np.random.RandomState(seed=seed)
    init_random_state = rs.get_state()
    X0 = rs.rand(N)*(box.x2-box.x1) + box.x1
    Y0 = rs.rand(N)*(box.y2-box.y1) + box.y1
    Z0 = rs.rand(N)*(box.z2-box.z1) + box.z1
    part = [Particle(x0=x0, y0=y0, z0=z0) for x0, y0, z0 in zip(X0, Y0, Z0)]
    return Particles(part, init_random_state=init_random_state)


def wrap_periodic(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply periodic boundary conditions.
    """
    a -= a1
    wrapped = np.mod(a, a2-a1) + a1
    return wrapped

def wrap_mirror(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply mirror-like boundary conditions.
    """
    a[a > a2] = a2 - (a[a > a2] - a2)
    a[a < a1] = a1 + (a1 - a[a < a1])
    return a


class ParticlesSimulation(object):
    """Class that performs the Brownian motion simulation of N particles.
    """
    def __init__(self, D, t_step, t_max, particles, box, psf, EID=0, ID=0):
        """Initialize the simulation parameters.

        Arguments:
            D (float): diffusion coefficient (m/s^2)
            t_step (float): time step (seconds)
            particles (Particles object): initial particle position
            box (Box object): the simulation boundaries
            psf (GaussianPSF or NumericPSF object): the PSF used in simulation
            EID (int): index for the engine on which the simulation is ran.
                Used to distinguish simulations when using parallel computing.
            ID (int): an index for the simulation. Can be used to distinguish
                simulations that are run multiple times.

        Note that EID and ID are shown in the string representation and are
        used to save unique file names.
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
        self.sigma_1d = sqrt(2*D*t_step)

    def __repr__(self):
        pM = self.concentration(pM=True)
        s = repr(self.box)
        s += "\nD %.2g, #Particles %d, %.1f pM, t_step %.1fus, t_max %.1fs" %\
                (self.D, self.np, pM, self.t_step*1e6, self.t_max)
        s += " ID_EID %d %d" % (self.ID, self.EID)
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
        s += "_ID%d-%d" % (self.ID, self.EID)
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

    def print_sizes(self):
        """Print on-disk array sizes required for current set of parameters."""
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

    def _save_group_attr(self, group, attr_name, attr_value):
        """Save attribute `attr_name` containing `attr_value` in `group`.
        """
        group = self.store.data_file.get_node(group)
        group._v_attrs[attr_name] = attr_value

    def _load_group_attr(self, group, attr_name):
        """Load attribute `attr_name` from `group`.
        """
        group = self.store.data_file.get_node(group)
        return group._v_attrs[attr_name]


    def open_store(self, prefix='pybromo_', chunksize=2**19, overwrite=True,
                   comp_filter=None):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
            prefix (string): file-name prefix for the HDF5 file
            chunksize (int): chunk size used for the on-disk arrays saved
                during the brownian motion simulation. Does not apply to
                the timestamps arrays (see :method:``).
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk arrays saved during the brownian motion
                simulation.
            overwrite (bool): if True, overwrite the file if already exists.
                All the previoulsy stored data in that file will be lost.
        """
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
        # Note psf.fname is the psf name in `data_file.root.psf`
        self._save_group_attr('/trajectories', 'psf_name', self.psf.fname)
        self.traj_group = self.store.data_file.root.trajectories
        self.ts_group = self.store.data_file.root.timestamps

        kwargs = dict(chunksize=self.chunksize,)
        if comp_filter is not None:
            kwargs.update(comp_filter=comp_filter)
        self.emission_tot = self.store.add_emission_tot(**kwargs)
        self.emission = self.store.add_emission(**kwargs)
        self.position = self.store.add_position(**kwargs)

    def sim_brownian_motion(self, save_pos=False, total_emission=True,
                            rs=None, seed=1, wrap_func=wrap_periodic,
                            verbose=True):
        """Simulate Brownian motion trajectories and emission rates.

        This method performs the Brownian motion simulation using the current
        set of parameters. Before running this method you can check the
        disk-space requirements using :method:`print_sizes`.

        Results are stored to disk in HDF5 format and are accessible in
        in `self.emission`, `self.emission_tot` and `self.position` as
        pytables arrays.

        Arguments:
            save_pos (bool): if True, save the particles 3D trajectories
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).
            verbose (bool): if False, prints no output.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)

        if 'store' not in self.__dict__:
            self.open_store()
        # Save current random state for reproducibility
        self._save_group_attr('/trajectories', 'init_random_state',
                              rs.get_state())

        em_store = self.emission_tot if total_emission else self.emission

        if verbose: print '[PID %d] Simulation chunk:' % os.getpid(),
        i_chunk = 0
        t_chunk_size = self.emission.chunkshape[1]

        par_start_pos = [p.r0 for p in self.particles]
        par_start_pos = np.vstack(par_start_pos).reshape(self.np, 3, 1)
        for c_size in iter_chunksize(self.n_samples, t_chunk_size):
            if verbose: print '.',
            if total_emission:
                em = np.zeros((c_size), dtype=np.float32)
            else:
                em = np.zeros((self.np, c_size), dtype=np.float32)

            POS = []
            #pos_w = np.zeros((3, c_size))
            for i in xrange(len(self.particles)):
                delta_pos = rs.normal(loc=0, scale=self.sigma_1d,
                                      size=3*c_size)
                delta_pos = delta_pos.reshape(3, c_size)
                pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
                pos += par_start_pos[i]

                # Coordinates wrapping using periodic boundary conditions
                for coord in (0, 1, 2):
                    pos[coord] = wrap_func(pos[coord], *self.box.b[coord])

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
                if save_pos:
                    POS.append(pos.reshape(1, 3, c_size))
                # Save last position as next starting position
                par_start_pos[i] = pos[:, -1:]

            ## Append em to the permanent storage
            # if total_emission is just a linear array
            # otherwise is an hstack of what is saved and em (self.np, c_size)
            em_store.append(em)
            if save_pos:
                self.position.append(np.vstack(POS).astype('float32'))
            i_chunk += 1

        # Save current random state
        self._save_group_attr('/trajectories', 'last_random_state',
                              rs.get_state())
        self.store.data_file.flush()

    def _get_ts_name_core(self, max_rate, bg_rate):
        return 'max_rate%dkcps_bg%dcps' % (max_rate*1e-3, bg_rate)

    def _get_ts_name(self, max_rate, bg_rate, rs_state, hashsize=4):
        return self._get_ts_name_core(max_rate, bg_rate) + \
                '_rs_%s' % hash_(rs_state)[:hashsize]

    def get_timestamp(self, max_rate, bg_rate):
        ts, ts_par = [], []
        name = self._get_ts_name_core(max_rate, bg_rate)
        for node in self.ts_group._f_list_nodes():
            if name in node.name:
                if node.name.endswith('_par'):
                    ts_par.append(node)
                else:
                    ts.append(node)
        if len(ts) == 0:
            raise ValueError("No array matching '%s'" % name)
        elif len(ts) > 1:
            print 'WARNING: multiple matches, only the first is returned.'
        return ts[0], ts_par[0]

    def timestamp_names(self):
        names = []
        for node in self.ts_group._f_list_nodes():
            if node.name.endswith('_par'): continue
            names.append(node.name)
        return names

    def sim_timestamps_em_store(self, max_rate=1, bg_rate=0, rs=None, seed=1,
                                chunksize=2**16, comp_filter=None,
                                overwrite=False):
        """Compute timestamps and particles arrays and store results to disk.

        The results are accessible as pytables arrays in `.timestamps` and
        `.tparticles`. The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rate (float, cps): max emission rate for a single particle
            bg_rate (float, cps): rate for a Poisson background process
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
            # Try to set the random state from the last session to preserve
            # a single random stream when simulating timestamps multiple times
            ts_attrs = self.store.data_file.root.timestamps._v_attrs
            if 'last_random_state' in ts_attrs._f_list():
                rs.set_state(ts_attrs['last_random_state'])
                print ("INFO: Random state set to last saved state"
                       " in '/timestamps'.")
            else:
                print "INFO: Random state initialized from seed (%d)." % seed

        fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
        scale = 10
        max_counts = 4

        name = self._get_ts_name(max_rate, bg_rate, rs.get_state())
        self.timestamps, self.tparticles = self.store.add_timestamps(
                name = name,
                clk_p = self.t_step/scale,
                max_rate = max_rate,
                bg_rate = bg_rate,
                num_particles = self.np,
                bg_particle = self.np,
                overwrite = overwrite,
                chunksize = chunksize,
                comp_filter = comp_filter)
        self.timestamps.set_attr('init_random_state', rs.get_state())

        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples,
                                               self.emission.chunkshape[1]):
            counts_chunk = sim_timetrace_bg(self.emission[:, i_start:i_end],
                                         max_rate, bg_rate, self.t_step, rs=rs)
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

            # Save (ordered) timestamps and corrensponding particles
            self.timestamps.append(times_chunk_s)
            self.tparticles.append(par_index_chunk_s)

        # Save current random state so it can be resumed in the next session
        self._save_group_attr('/timestamps', 'last_random_state',
                              rs.get_state())
        self.store.data_file.flush()

def sim_timetrace(emission, max_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).
    """
    emission_rates = emission*max_rate*t_step
    return NR.poisson(lam=emission_rates).astype(np.uint8)

def sim_timetrace_bg(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from Poisson(emission_rates).
    Return an uint8 array of counts with shape[0] == emission.shape[0] + 1.
    The last row is a "fake" particle representing Poisson background.
    """
    if rs is None:
        rs = np.random.RandomState()
    em = np.atleast_2d(emission).astype('float64', copy=False)
    counts = np.zeros((em.shape[0] + 1, em.shape[1]), dtype='u1')
    # In-place computation
    # NOTE: the caller will see the modification
    em *= (max_rate*t_step)
    # Use automatic type conversion int64 -> uint8
    counts[:-1] = rs.poisson(lam=em)
    counts[-1] = rs.poisson(lam=bg_rate*t_step, size=em.shape[1])
    return counts

def sim_timetrace_bg2(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from Poisson(emission_rates).
    Return an uint8 array of counts with shape[0] == emission.shape[0] + 1.
    The last row is a "fake" particle representing Poisson background.
    """
    if rs is None:
        rs = np.random.RandomState()
    emiss_bin_rate = np.zeros((emission.shape[0] + 1, emission.shape[1]),
                              dtype='float64')
    emiss_bin_rate[:-1] = emission*max_rate*t_step
    emiss_bin_rate[-1] = bg_rate*t_step
    counts = rs.poisson(lam=emiss_bin_rate).astype('uint8')
    return counts

def load_simulation(fname):
    fnames = glob(fname)
    if len(fnames) > 1:
        raise ValueError('Glob matched more than 1 file!')
    store = Storage(fnames[0], overwrite=False)
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
    S.store_fname = fnames[0]
    S.psf_pytables = psf_pytables
    S.traj_group = S.store.data_file.root.trajectories
    S.ts_group = S.store.data_file.root.timestamps
    S.emission = S.store.data_file.root.trajectories.emission
    S.emission_tot = S.store.data_file.root.trajectories.emission_tot
    S.position = S.store.data_file.root.trajectories.position
    S.chunksize = S.store.data_file.get_node('/parameters', 'chunksize')
    if '/timestamps' in S.store.data_file:
        name_list = S.ts_group._v_children.keys()
        if len(name_list) == 2:
            for name in name_list:
                if name.endswith('_par'):
                    S.tparticles = S.ts_group._f_get_child(name)
                else:
                    S.timestamps = S.ts_group._f_get_child(name)
    return S

def merge_ts_da(ts_d, ts_par_d, ts_a, ts_par_a):
    """Merge donor and acceptor timestamps and particle arrays.

    Parameters:
        ts_d (array): donor timestamp array
        ts_par_d (array): donor particles array
        ts_a (array): acceptor timestamp array
        ts_par_a (array): acceptor particles array

    Returns:
        Arrays: timestamps, acceptor bool mask, timestamp particle
    """
    ts = np.hstack([ts_d, ts_a])
    ts_par = np.hstack([ts_par_d, ts_par_a])
    a_ch = np.hstack([np.zeros(ts_d.shape[0], dtype=bool),
                      np.ones(ts_a.shape[0], dtype=bool)])
    index_sort = ts.argsort()
    return ts[index_sort], a_ch[index_sort], ts_par[index_sort]

##
# Functions to manage/merge multiple simulations
#
def merge_ph_times(times_list, times_par_list, time_block):
    """Build an array of timestamps joining the arrays in `ph_times_list`.
    `time_block` is the duration of each array of timestamps.
    """
    offsets = np.arange(len(times_list))*time_block
    cum_sizes = np.cumsum([ts.size for ts in times_list])
    times = np.zeros(cum_sizes[-1])
    times_par = np.zeros(cum_sizes[-1], dtype='uint8')
    i1 = 0
    for i2, ts, ts_par, offset in zip(cum_sizes, times_list, times_par_list,
                                      offsets):
        times[i1:i2] = ts + offset
        times_par[i1:i2] = ts_par
        i1 = i2
    return times, times_par

def merge_DA_ph_times(ph_times_d, ph_times_a):
    """Returns a merged timestamp array for Donor+Accept. and bool mask for A.
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
    dview.execute('S.sim_timestamps_em_store(max_rate=%d, bg_rate=%d, '
                  'seed=S.EID, overwrite=True)' % (max_em_rate, bg_rate))
    dview.execute('times = S.timestamps[:]')
    dview.execute('times_par = S.timestamps_par[:]')
    Times = dview['times']
    Times_par = dview['times_par']
    # Assuming all t_max equal, just take the first
    t_max = dview['S.t_max'][0]
    t_tot = np.sum(dview['S.t_max'])
    dview.execute("sim_name = S.compact_name_core(t_max=False, hashdigit=0)")
    # Core names contains no ID or t_max
    sim_name = dview['sim_name'][0]
    times_all, times_par_all = merge_ph_times(Times, Times_par,
                                              time_block=t_max)
    return times_all, times_par_all, t_tot, sim_name


if __name__ == '__main__':
    # Simulation time step
    #t_step = 0.5e-6     # seconds

    # Diffusion coefficient
    #Du = 12.0           # um^2 / s
    #D = Du*(1e-6)**2

    # Time duration of the simulation
    #t_max = 0.3        # seconds
    #n_samples = int(t_max/t_step)

    # PSF definition
    #ss = 0.2*1e-6      # lateral dimension (sigma)
    #psf = GaussianPSF(xc=0, yc=0, zc=0, sx=ss, sy=ss, sz=3*ss)
    #psf = NumericPSF()

    # Box definition
    #box = Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)

    # Particles definition
    #p1 = Particle(x0=-3e-6)
    #p2 = Particle(x0=3e-6)
    #p3 = Particle(y0=-3e-6)
    #p4 = Particle(y0=3e-6)
    #P = [p1,p2,p3,p4]
    #P = gen_particles(15, box)

    # Brownian motion and emission simulation
    #S = ParticlesSimulation(D=D, t_step=t_step, t_max=t_max,
    #                        particles=P, box=box, psf=psf)
    #S.sim_brownian_motion(delete_pos=False)

    #plot_tracks(S)
    #plot_emission(S)
    pass

