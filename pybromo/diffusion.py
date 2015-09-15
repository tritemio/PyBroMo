#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module contains the core classes and functions to perform the
Brownian motion and timestamps simulation.
"""

from __future__ import print_function, absolute_import, division
from builtins import range, zip

import os
import hashlib
import itertools
from pathlib import Path

import numpy as np
from numpy import array, sqrt

from .storage import TrajectoryStore, TimestampStore
from .iter_chunks import iter_chunksize, iter_chunk_index
from .psflib import NumericPSF


## Avogadro constant
NA = 6.022141e23    # [mol^-1]


def get_seed(seed, ID=0, EID=0):
    """Get a random seed that is a combination of `seed`, `ID` and `EID`.
    Provides different, but deterministic, seeds in parallel computations
    """
    return seed + EID + 100 * ID

def hash_(x):
    return hashlib.sha1(repr(x).encode()).hexdigest()

class Box:
    """The simulation box"""
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2
        self.b = array([[x1, x2], [y1, y2], [z1, z2]])

    @property
    def volume(self):
        """Box volume in m^3."""
        return (self.x2 - self.x1) * (self.y2 - self.y1) * (self.z2 - self.z1)

    @property
    def volume_L(self):
        """Box volume in liters."""
        return self.volume * 1e3

    def __repr__(self):
        return u"Box: X %.1fum, Y %.1fum, Z %.1fum" % (
            (self.x2 - self.x1) * 1e6,
            (self.y2 - self.y1) * 1e6,
            (self.z2 - self.z1) * 1e6)


class Particle(object):
    """Class to describe a single particle."""
    def __init__(self, D, x0=0, y0=0, z0=0):
        self.D = D   # diffusion coefficient in SI units, m^2/s
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.r0 = np.array([x0, y0, z0])

    def __eq__(self, other_particle):
        return (self.r0 == other_particle.r0).all()


class Particles(object):
    """A list of Particle() objects and a few attributes."""

    @staticmethod
    def _generate(num_particles, D, box, rs):
        """Generate a list of `Particle` objects."""
        X0 = rs.rand(num_particles) * (box.x2 - box.x1) + box.x1
        Y0 = rs.rand(num_particles) * (box.y2 - box.y1) + box.y1
        Z0 = rs.rand(num_particles) * (box.z2 - box.z1) + box.z1
        return [Particle(D=D, x0=x0, y0=y0, z0=z0)
                for x0, y0, z0 in zip(X0, Y0, Z0)]

    def __init__(self, num_particles, D, box, rs=None, seed=1):
        """A set of `N` Particle() objects with random position in `box`.

        Arguments:
            num_particles (int): number of particles to be generated
            D (float): diffusion coefficient in S.I. units (m^2/s)
            box (Box object): the simulation box
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state. `seed` is ignored when `rs` is not None.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
        self.rs = rs
        self.init_random_state = rs.get_state()
        self.box = box
        self._plist = self._generate(num_particles, D, box, rs)
        self.rs_hash = hash_(self.init_random_state)[:3]

    def add(self, num_particles, D):
        """Add particles with diffusion coeff `D` at random positions.
        """
        self._plist += self._generate(num_particles, D, box=self.box,
                                      rs=self.rs)

    def to_list(self):
        return self._plist.copy()

    def __iter__(self):
        return iter(self._plist)

    def __len__(self):
        return len(self._plist)

    def __getitem__(self, i):
        return self._plist[i]

    def __eq__(self, other_particles):
        if len(self) != len(other_particles):
            return False
        equal = np.array([p1 == p2 for p1, p2 in zip(self, other_particles)])
        return equal.all()

    @property
    def diffusion_coeff(self):
        return np.array([par.D for par in self])

    @property
    def diffusion_coeff_counts(self):
        """List of tuples of (diffusion coefficient, counts) pairs.

        The order of the diffusion coefficients is as in self.diffusion_coeff.
        """
        return [(key, len(list(group)))
                for key, group in itertools.groupby(self.diffusion_coeff)]

    def short_repr(self):
        s = ["P%d_D%.2g" % (n, D) for D, n in self.diffusion_coeff_counts]
        return "_".join(s)

    def __repr__(self):
        s = ["#Particles: %d D: %.2g" % (n, D)
             for D, n in self.diffusion_coeff_counts]
        return ", ".join(s)


def wrap_periodic(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply periodic boundary conditions.
    """
    a -= a1
    wrapped = np.mod(a, a2 - a1) + a1
    return wrapped

def wrap_mirror(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply mirror-like boundary conditions.
    """
    a[a > a2] = a2 - (a[a > a2] - a2)
    a[a < a1] = a1 + (a1 - a[a < a1])
    return a

class NoMatchError(Exception):
    pass
class MultipleMatchesError(Exception):
    pass

class ParticlesSimulation(object):
    """Class that performs the Brownian motion simulation of N particles.
    """
    _PREFIX_TRAJ = 'pybromo'
    _PREFIX_TS = 'times'

    @staticmethod
    def datafile_from_hash(hash_, prefix, path):
        """Return pathlib.Path for a data-file with given hash and prefix.
        """
        pattern = '%s_%s*.h*' % (prefix, hash_)
        datafiles = list(path.glob(pattern))
        if len(datafiles) == 0:
            raise NoMatchError('No matches for "%s"' % pattern)
        if len(datafiles) > 1:
            raise MultipleMatchesError('More than 1 match for "%s"' % pattern)
        return datafiles[0]

    @staticmethod
    def from_datafile(hash_, path='./', ignore_timestamps=False, mode='r'):
        """Load simulation from disk trajectories and (when present) timestamps.
        """
        path = Path(path)
        assert path.exists()

        file_traj = ParticlesSimulation.datafile_from_hash(
            hash_, prefix=ParticlesSimulation._PREFIX_TRAJ, path=path)
        store = TrajectoryStore(file_traj, mode='r')

        psf_pytables = store.h5file.get_node('/psf/default_psf')
        psf = NumericPSF(psf_pytables=psf_pytables)
        box = store.h5file.get_node_attr('/parameters', 'box')
        P = store.h5file.get_node_attr('/parameters', 'particles')

        names = ['t_step', 't_max', 'EID', 'ID']
        kwargs = {name: store.numeric_params[name] for name in names}
        S = ParticlesSimulation(particles=P, box=box, psf=psf, **kwargs)

        # Emulate S.open_store_traj()
        S.store = store
        S.psf_pytables = psf_pytables
        S.traj_group = S.store.h5file.root.trajectories
        S.emission = S.store.h5file.root.trajectories.emission
        S.emission_tot = S.store.h5file.root.trajectories.emission_tot
        S.position = S.store.h5file.root.trajectories.position
        S.chunksize = S.store.h5file.get_node('/parameters', 'chunksize')
        if not ignore_timestamps:
            try:
                file_ts = ParticlesSimulation.datafile_from_hash(
                    hash_, prefix=ParticlesSimulation._PREFIX_TS, path=path)
            except NoMatchError:
                # There are no timestamps saved.
                pass
            else:
                # Load the timestamps
                S.ts_store = TimestampStore(file_ts, mode=mode)
                S.ts_group = S.ts_store.h5file.root.timestamps
        return S

    @staticmethod
    def _get_randomstate(rs, seed, group):
        if rs is None:
            rs = np.random.RandomState(seed=seed)
            # Try to set the random state from the last session to preserve
            # a single random stream when simulating timestamps multiple times
            if 'last_random_state' in group._v_attrs:
                rs.set_state(group._v_attrs['last_random_state'])
                print("INFO: Random state set to last saved state in '%s'." % \
                      group._v_name)
            else:
                print("INFO: Random state initialized from seed (%d)." % seed)
        return rs

    def __init__(self, t_step, t_max, particles, box, psf, EID=0, ID=0):
        """Initialize the simulation parameters.

        Arguments:
            D (float): diffusion coefficient (m/s^2)
            t_step (float): simulation time step (seconds)
            t_max (float): simulation time duration (seconds)
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
        self.t_step = t_step
        self.t_max = t_max
        self.ID = ID
        self.EID = EID
        self.n_samples = int(t_max / t_step)

    @property
    def diffusion_coeff(self):
        return np.array([par.D for par in self.particles])

    @property
    def num_particles(self):
        return len(self.particles)

    @property
    def sigma_1d(self):
        return [np.sqrt(2 * par.D * self.t_step) for par in self.particles]

    def __repr__(self):
        pM = self.concentration(pM=True)
        s = repr(self.box)
        s += "\n%s, %.1f pM, t_step %.1fus, t_max %.1fs" %\
             (self.particles, pM, self.t_step * 1e6, self.t_max)
        s += " ID_EID %d %d" % (self.ID, self.EID)
        return s

    def hash(self):
        """Return an hash for the simulation parameters (excluding ID and EID)
        This can be used to generate unique file names for simulations
        that have the same parameters and just different ID or EID.
        """
        hash_numeric = 't_step=%.3e, t_max=%.2f, np=%d, conc=%.2e' % \
            (self.t_step, self.t_max, self.num_particles, self.concentration())
        hash_list = [hash_numeric, self.particles.short_repr(), repr(self.box),
                     self.psf.hash()]
        return hashlib.md5(repr(hash_list).encode()).hexdigest()

    def compact_name_core(self, hashdigits=6, t_max=False):
        """Compact representation of simulation params (no ID, EID and t_max)
        """
        Moles = self.concentration()
        name = "%s_%dpM_step%.1fus" % (
            self.particles.short_repr(), Moles * 1e12, self.t_step * 1e6)
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

    @property
    def numeric_params(self):
        """A dict containing all the simulation numeric-parameters.

        The values are 2-element tuples: first element is the value and
        second element is a string describing the parameter (metadata).
        """
        nparams = dict(
            D = (self.diffusion_coeff.mean(), 'Diffusion coefficient (m^2/s)'),
            np = (self.num_particles, 'Number of simulated particles'),
            t_step = (self.t_step, 'Simulation time-step (s)'),
            t_max = (self.t_max, 'Simulation total time (s)'),
            ID = (self.ID, 'Simulation ID (int)'),
            EID = (self.EID, 'IPython Engine ID (int)'),
            pico_mol = (self.concentration() * 1e12,
                        'Particles concentration (pM)'))
        return nparams

    def print_sizes(self):
        """Print on-disk array sizes required for current set of parameters."""
        float_size = 4
        MB = 1024 * 1024
        size_ = self.n_samples * float_size
        em_size = size_ * self.num_particles / MB
        pos_size = 3 * size_ * self.num_particles / MB
        print("  Number of particles:", self.num_particles)
        print("  Number of time steps:", self.n_samples)
        print("  Emission array - 1 particle (float32): %.1f MB" % (size_ / MB))
        print("  Emission array (float32): %.1f MB" % em_size)
        print("  Position array (float32): %.1f MB " % pos_size)

    def concentration(self, pM=False):
        """Return the concentration (in Moles) of the particles in the box.
        """
        concentr = (self.num_particles / NA) / self.box.volume_L
        if pM:
            concentr *= 1e12
        return concentr

    __DOCS_STORE_ARGS___ = """
            prefix (string): file-name prefix for the HDF5 file.
            path (string): a folder where simulation data is saved.
            chunksize (int): chunk size used for the on-disk arrays saved
                during the brownian motion simulation. Does not apply to
                the timestamps arrays (see :method:``).
            chunkslice ('times' or 'bytes'): if 'bytes' (default) the chunksize
                is taken as the size in bytes of the chunks. Else, if 'times'
                chunksize is the size of the last dimension. In this latter
                case 2-D or 3-D arrays have bigger chunks than 1-D arrays.
            overwrite (bool): if True, overwrite the file if already exists.
                All the previoulsy stored data in that file will be lost.
        """[1:]

    def _open_store(self, store, prefix='', path='./', chunksize=2**19,
                    chunkslice='bytes', mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Low level method used to implement different stores.

        Arguments:
            store (one of storage.Store classes): the store class to use.
        """ + self.__DOCS_STORE_ARGS___ + """
        Returns:
            Store object.
        """
        nparams = self.numeric_params
        self.chunksize = chunksize
        nparams.update(chunksize=(chunksize, 'Chunksize for arrays'))
        store_fname = '%s_%s.hdf5' % (prefix, self.compact_name())
        attr_params = dict(particles=self.particles, box=self.box)
        kwargs = dict(path=path, nparams=nparams, attr_params=attr_params,
                      mode=mode)
        store = store(store_fname, **kwargs)
        return store

    def open_store_traj(self, path='./', chunksize=2**19, chunkslice='bytes',
                        mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
        """ + self.__DOCS_STORE_ARGS___
        if hasattr(self, 'store'):
            return
        self.store = self._open_store(TrajectoryStore,
                                      prefix=ParticlesSimulation._PREFIX_TRAJ,
                                      path=path,
                                      chunksize=chunksize,
                                      chunkslice=chunkslice,
                                      mode=mode)

        self.psf_pytables = self.psf.to_hdf5(self.store.h5file, '/psf')
        self.store.h5file.create_hard_link('/psf', 'default_psf',
                                           target=self.psf_pytables)
        # Note psf.fname is the psf name in `h5file.root.psf`
        self.traj_group = self.store.h5file.root.trajectories
        self.traj_group._v_attrs['psf_name'] = self.psf.fname

        kwargs = dict(chunksize=self.chunksize, chunkslice=chunkslice)
        self.emission_tot = self.store.add_emission_tot(**kwargs)
        self.emission = self.store.add_emission(**kwargs)
        self.position = self.store.add_position(**kwargs)

    def open_store_timestamp(self, path='./', chunksize=2**19,
                             chunkslice='bytes', mode='w'):
        """Open and setup the on-disk storage file (pytables HDF5 file).

        Arguments:
        """ + self.__DOCS_STORE_ARGS___
        if hasattr(self, 'ts_store'):
            return
        self.ts_store = self._open_store(TimestampStore,
                                         prefix=ParticlesSimulation._PREFIX_TS,
                                         path=path,
                                         chunksize=chunksize,
                                         chunkslice=chunkslice,
                                         mode=mode)
        self.ts_group = self.ts_store.h5file.root.timestamps

    def _sim_trajectories(self, time_size, start_pos, rs,
                          total_emission=False, save_pos=False,
                          wrap_func=wrap_periodic):
        """Simulate (in-memory) `time_size` steps of trajectories.

        Simulate Brownian motion diffusion and emission of all the particles.
        Uses the attrbutes: num_particles, sigma_1d, box, psf.

        Arguments:
            time_size (int): number of time steps to be simulated.
            start_pos (array): shape (num_particles, 3), particles start
                positions. This array is modified to store the end position
                after this method is called.
            rs (RandomState): a `numpy.random.RandomState` object used
                to generate the random numbers.
            total_emission (bool): if True, store only the total emission array
                containing the sum of emission of all the particles.
            save_pos (bool): if True, save the particles 3D trajectories
            wrap_func (function): the function used to apply the boundary
                condition (use :func:`wrap_periodic` or :func:`wrap_mirror`).

        Returns:
            POS (list): list of 3D trajectories arrays (3 x time_size)
            em (array): array of emission (total or per-particle)
        """
        num_particles = self.num_particles
        if total_emission:
            em = np.zeros((time_size), dtype=np.float32)
        else:
            em = np.zeros((num_particles, time_size), dtype=np.float32)

        POS = []
        # pos_w = np.zeros((3, c_size))
        for i, sigma_1d in enumerate(self.sigma_1d):
            delta_pos = rs.normal(loc=0, scale=sigma_1d,
                                  size=3 * time_size)
            delta_pos = delta_pos.reshape(3, time_size)
            pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
            pos += start_pos[i]

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
                POS.append(pos.reshape(1, 3, time_size))
            # Save last position as next starting position
            start_pos[i] = pos[:, -1:]
        return POS, em

    def simulate_diffusion(self, save_pos=False, total_emission=True,
                           rs=None, seed=1, wrap_func=wrap_periodic, path='./',
                           verbose=True, chunksize=2**19, chunkslice='bytes'):
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
            path (string): a folder where simulation data is saved.
            verbose (bool): if False, prints no output.
        """
        if rs is None:
            rs = np.random.RandomState(seed=seed)
        self.open_store_traj(chunksize=chunksize, chunkslice=chunkslice,
                             path=path)
        # Save current random state for reproducibility
        self.traj_group._v_attrs['init_random_state'] = rs.get_state()

        em_store = self.emission_tot if total_emission else self.emission

        if verbose:
            print('[PID %d] Diffusion time:' % os.getpid(), end='')
        i_chunk = 0
        t_chunk_size = self.emission.chunkshape[1]
        chunk_duration = t_chunk_size * self.t_step

        par_start_pos = [p.r0 for p in self.particles]
        par_start_pos = (np.vstack(par_start_pos)
                         .reshape(self.num_particles, 3, 1))
        for time_size in iter_chunksize(self.n_samples, t_chunk_size):
            if verbose:
                print(' %.1fs' % (chunk_duration * i_chunk), end='')

            POS, em = self._sim_trajectories(time_size, par_start_pos, rs,
                                             total_emission=total_emission,
                                             save_pos=save_pos,
                                             wrap_func=wrap_func)

            ## Append em to the permanent storage
            # if total_emission, data is just a linear array
            # otherwise is a 2-D array (self.num_particles, c_size)
            em_store.append(em)
            if save_pos:
                self.position.append(np.vstack(POS).astype('float32'))
            i_chunk += 1
            self.store.h5file.flush()

        # Save current random state
        self.traj_group._v_attrs['last_random_state'] = rs.get_state()
        self.store.h5file.flush()

    def _get_ts_name_core(self, max_rate, bg_rate):
        return 'max_rate%dkcps_bg%dcps' % (max_rate * 1e-3, bg_rate)

    def _get_ts_name(self, max_rate, bg_rate, rs_state, hashsize=4):
        return '%s_rs_%s' % (self._get_ts_name_core(max_rate, bg_rate),
                             hash_(rs_state)[:hashsize])

    def _get_ts_name_mix_core(self, max_rates, populations, bg_rate):
        pop1, pop2 = populations
        max_rate1, max_rate2 = max_rates
        s = 'Pop1_P%d_%d_max_rate%dkcps_Pop2_P%d_%d_max_rate%dkcps_bg%dcps' % (
            pop1.start, pop1.stop - 1, max_rate1 * 1e-3,
            pop2.start, pop2.stop - 1, max_rate2 * 1e-3, bg_rate)
        return s

    def _get_ts_name_mix(self, max_rates, populations, bg_rate, rs_state,
                         hashsize=4):
        s = self._get_ts_name_mix_core(max_rates, populations, bg_rate)
        return '%s_rs_%s' % (s, hash_(rs_state)[:hashsize])

    @property
    def timestamp_names(self):
        names = []
        for node in self.ts_group._f_list_nodes():
            if node.name.endswith('_par'):
                continue
            names.append(node.name)
        return names

    def timestamps_match_pattern(self, pattern):
        return [t for t in self.timestamp_names if pattern in t]

    def timestamps_match_mix(self, max_rates, populations, bg_rate):
        pattern = self._get_ts_name_mix_core(max_rates, populations, bg_rate)
        return self.timestamps_match_pattern(pattern)

    def get_timestamps_name(self, max_rate=None, bg_rate=None, hash_=''):
        """Return name of the matching timestamps array.
        """
        if len(hash_) > 0:
            pattern = hash_
        else:
            assert max_rate is not None and bg_rate is not None
            pattern = self._get_ts_name_core(max_rate, bg_rate)
            pattern += '_rs_%s' % hash_
        matches = [n for n in self.timestamp_names if pattern in n]
        if len(matches) == 0:
            raise ValueError('No timestamps matching.')
        elif len(matches) > 1:
            raise ValueError('Multiple matches, try specifying `hash_`.')
        return matches[0]

    def get_timestamps_part(self, name):
        """Return matching (timestamps, particles) pytables arrays.
        """
        par_name = name + '_par'
        timestamps = self.ts_store.h5file.get_node('/timestamps', name)
        particles = self.ts_store.h5file.get_node('/timestamps', par_name)
        return timestamps, particles

    def get_timestamps(self, max_rate=None, bg_rate=None, hash_=''):
        """Return matching (timestamps, particles) pytables arrays.
        """
        name = self.get_timestamps_name(max_rate, bg_rate, hash_)
        par_name = name + '_par'
        timestamps = self.ts_store.h5file.get_node('/timestamps', name)
        particles = self.ts_store.h5file.get_node('/timestamps', par_name)
        return timestamps, particles

    def _sim_timestamps(self, max_rate, bg_rate, emission, i_start, rs,
                        ip_start=0, scale=10, max_counts=4, sort=True):
        """Simulate timestamps from emission trajectories.

        Uses attributes: `.t_step`.

        Returns:
            A tuple of two arrays: timestamps and particles.
        """
        counts_chunk = sim_timetrace_bg(emission, max_rate, bg_rate,
                                        self.t_step, rs=rs)
        nrows = emission.shape[0]
        if bg_rate is not None:
            nrows += 1
        assert counts_chunk.shape == (nrows, emission.shape[1])

        ts_range = (np.arange(counts_chunk.shape[1]) + i_start) * scale

        # Loop for each particle to compute timestamps
        times_chunk_p = []
        par_index_chunk_p = []
        for ip, counts_chunk_ip in enumerate(counts_chunk):
            # Compute timestamps for particle ip for all bins with counts
            times_c_ip = []
            for v in range(1, max_counts + 1):
                times_c_ip.append(ts_range[counts_chunk_ip >= v])

            # Stack the timstamps from different "counts"
            t = np.hstack(times_c_ip)
            # Append current particle
            times_chunk_p.append(t)
            par_index_chunk_p.append(np.full(t.size, ip + ip_start, dtype='u1'))

        # Merge the arrays of different particles
        times_chunk = np.hstack(times_chunk_p)
        par_index_chunk = np.hstack(par_index_chunk_p)

        if sort:
            # Sort timestamps inside the merged chunk
            index_sort = times_chunk.argsort(kind='mergesort')
            times_chunk = times_chunk[index_sort]
            par_index_chunk = par_index_chunk[index_sort]

        return times_chunk, par_index_chunk

    def simulate_timestamps(self, max_rate=1, bg_rate=0, rs=None, seed=1,
                            chunksize=2**16, comp_filter=None,
                            overwrite=False, scale=10, path='./',
                            t_chunksize=None):
        """Compute timestamps and particles arrays storing results to disk.

        The results are accessible as pytables arrays in `.timestamps` and
        `.tparticles`. The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rate (float, cps): max emission rate for a single particle
            bg_rate (float, cps): rate for a Poisson background process
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
                If None use default compression.
             (bool): if True  an timestamp array with the
                same name.
            scale (int): `self.t_step` is multiplied by `scale` to obtain the
                timestamps units in seconds.
            path (string): folder where to save the data.
        """
        self.open_store_timestamp(chunksize=chunksize, path=path)
        rs = self._get_randomstate(rs, seed, self.ts_group)
        if t_chunksize is None:
            t_chunksize = self.emission.chunkshape[1]

        name = self._get_ts_name(max_rate, bg_rate, rs.get_state())
        kw = dict(name=name, clk_p=self.t_step / scale,
                  max_rate=max_rate, bg_rate=bg_rate,
                  num_particles=self.num_particles,
                  bg_particle=self.num_particles,
                  overwrite=overwrite, chunksize=chunksize)
        if comp_filter is not None:
            kw.update(comp_filter=comp_filter)
        self._timestamps, self._tparticles = self.ts_store.add_timestamps(**kw)
        self.ts_group._v_attrs['init_random_state'] = rs.get_state()

        # Load emission in chunks, and save only the final timestamps
        for i_start, i_end in iter_chunk_index(self.n_samples, t_chunksize):
            em_chunk = self.emission[:, i_start:i_end]
            times_chunk_s, par_index_chunk_s = self._sim_timestamps(
                max_rate, bg_rate, em_chunk, i_start, rs=rs, scale=scale)

            # Save sorted timestamps (suffix '_s') and corresponding particles
            self._timestamps.append(times_chunk_s)
            self._tparticles.append(par_index_chunk_s)

        # Save current random state so it can be resumed in the next session
        self.ts_group._v_attrs['last_random_state'] = rs.get_state()
        self.ts_store.h5file.flush()

    def simulate_timestamps_mix(self, max_rates, populations, bg_rate,
                                rs=None, seed=1, chunksize=2**16,
                                comp_filter=None, overwrite=False, scale=10,
                                path='./', t_chunksize=None):
        """Compute timestamps for a mixture of 2 populations.

        The results are saved to disk and accessible as pytables arrays in
        `.timestamps` and `.tparticles`.
        The background generated timestamps are assigned a
        conventional particle number (last particle index + 1).

        Arguments:
            max_rate (float, cps): max emission rate for a single particle
            populations (2-tuple of slices): slices for `self.particles` that
                define the two populations.
            bg_rate (float, cps): rate for a Poisson background process
            rs (RandomState object): random state object used as random number
                generator. If None, use a random state initialized from seed.
            seed (uint): when `rs` is None, `seed` is used to initialize the
                random state, otherwise is ignored.
            chunksize (int): chunk size used for the on-disk timestamp array
            comp_filter (tables.Filter or None): compression filter to use
                for the on-disk `timestamps` and `tparticles` arrays.
                If None use default compression.
             (bool): if True  an timestamp array with the
                same name.
            scale (int): `self.t_step` is multiplied by `scale` to obtain the
                timestamps units in seconds.
            path (string): folder where to save the data.
        """
        self.open_store_timestamp(chunksize=chunksize, path=path)
        rs = self._get_randomstate(rs, seed, self.ts_group)
        if t_chunksize is None:
            t_chunksize = self.emission.chunkshape[1]

        name = self._get_ts_name_mix(max_rates, populations, bg_rate,
                                     rs.get_state())
        kw = dict(name=name, clk_p=self.t_step / scale,
                  max_rate=max_rates, bg_rate=bg_rate, populations=populations,
                  num_particles=self.num_particles,
                  bg_particle=self.num_particles,
                  overwrite=overwrite, chunksize=chunksize)
        if comp_filter is not None:
            kw.update(comp_filter=comp_filter)
        self._timestamps, self._tparticles = self.ts_store.add_timestamps(**kw)
        self.ts_group._v_attrs['init_random_state'] = rs.get_state()

        # Load emission in chunks, and save only the final timestamps
        bg_rates = (None, bg_rate)
        for i_start, i_end in iter_chunk_index(self.n_samples, t_chunksize):
            em_chunk = self.emission[:, i_start:i_end]

            # Loop for each population
            ts_chunk_pop_list, par_index_chunk_pop_list = [], []
            for rate, pop, bg in zip(max_rates, populations, bg_rates):
                em_chunk_pop = em_chunk[pop]
                ts_chunk_pop, par_index_chunk_pop = self._sim_timestamps(
                    rate, bg, em_chunk_pop, i_start, ip_start=pop.start,
                    rs=rs, scale=scale, sort=False)

                ts_chunk_pop_list.append(ts_chunk_pop)
                par_index_chunk_pop_list.append(par_index_chunk_pop)

            # Merge populations
            times_chunk_s = np.hstack(ts_chunk_pop_list)
            par_index_chunk_s = np.hstack(par_index_chunk_pop_list)

            # Sort timestamps inside the merged chunk
            index_sort = times_chunk_s.argsort(kind='mergesort')
            times_chunk_s = times_chunk_s[index_sort]
            par_index_chunk_s = par_index_chunk_s[index_sort]

            # Save sorted timestamps (suffix '_s') and corresponding particles
            self._timestamps.append(times_chunk_s)
            self._tparticles.append(par_index_chunk_s)

        # Save current random state so it can be resumed in the next session
        self.ts_group._v_attrs['last_random_state'] = rs.get_state()
        self.ts_store.h5file.flush()


def sim_timetrace(emission, max_rate, t_step):
    """Draw random emitted photons from Poisson(emission_rates).
    """
    emission_rates = emission * max_rate * t_step
    return np.random.poisson(lam=emission_rates).astype(np.uint8)

def sim_timetrace_bg(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from r.v. ~ Poisson(emission_rates).

    Arguments:
        emission (2D array): array of normalized emission rates. One row per
            particle (axis = 0). Columns are the different time steps.
        max_rate (float): the peak emission rate in Hz.
        bg_rate (float or None): rate of a constant Poisson background (Hz).
            Background is added as an additional row in the returned array
            of counts. If None, no background simulated.
        t_step (float): duration of a time step in seconds.
        rs (RandomState or None): object used to draw the random numbers.
            If None, a new RandomState is created using a random seed.

    Returns:
        `counts` an 2D uint8 array of counts in each time bin, for each
        particle. If `bg_rate` is None counts.shape == emission.shape.
        Otherwise, `counts` has one row more than `emission` for storing
        the constant Poisson background.
    """
    if rs is None:
        rs = np.random.RandomState()
    em = np.atleast_2d(emission).astype('float64', copy=False)
    counts_nrows = em.shape[0]
    if bg_rate is not None:
        counts_nrows += 1   # add a row for poisson background
    counts = np.zeros((counts_nrows, em.shape[1]), dtype='u1')
    # In-place computation
    # NOTE: the caller will see the modification
    em *= (max_rate * t_step)
    # Use automatic type conversion int64 (counts_par) -> uint8 (counts)
    counts_par = rs.poisson(lam=em)
    if bg_rate is None:
        counts[:] = counts_par
    else:
        counts[:-1] = counts_par
        counts[-1] = rs.poisson(lam=bg_rate * t_step, size=em.shape[1])
    return counts

def sim_timetrace_bg2(emission, max_rate, bg_rate, t_step, rs=None):
    """Draw random emitted photons from r.v. ~ Poisson(emission_rates).

    This is an alternative implementation of :func:`sim_timetrace_bg`.
    """
    if rs is None:
        rs = np.random.RandomState()
    emiss_bin_rate = np.zeros((emission.shape[0] + 1, emission.shape[1]),
                              dtype='float64')
    emiss_bin_rate[:-1] = emission * max_rate * t_step
    if bg_rate is not None:
        emiss_bin_rate[-1] = bg_rate * t_step
        counts = rs.poisson(lam=emiss_bin_rate).astype('uint8')
    else:
        counts = rs.poisson(lam=emiss_bin_rate[:-1]).astype('uint8')
    return counts
