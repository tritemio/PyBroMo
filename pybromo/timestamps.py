#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module contains functions to work with timestamps.
"""

import numpy as np
from time import ctime
from pathlib import Path

import phconvert as phc

from ._version import get_versions
__version__ = get_versions()['version']

from .diffusion import hash_


def merge_da(ts_d, ts_par_d, ts_a, ts_par_a):
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
#  Timestamp simulation definitions
#

def em_rates_from_E_DA(em_rate_tot, E_values):
    """Donor and Acceptor emission rates from total emission rate and E (FRET).
    """
    E_values = np.asarray(E_values)
    em_rates_a = E_values * em_rate_tot
    em_rates_d = em_rate_tot - em_rates_a
    return em_rates_d, em_rates_a

def em_rates_from_E_unique(em_rate_tot, E_values):
    """Array of unique emission rates for given total emission and E (FRET).
    """
    em_rates_d, em_rates_a = em_rates_from_E_DA(em_rate_tot, E_values)
    return np.unique(np.hstack([em_rates_d, em_rates_a]))

def em_rates_from_E_DA_mix(em_rates_tot, E_values):
    """D and A emission rates for two populations.
    """
    em_rates_d, em_rates_a = [], []
    for em_rate_tot, E_value in zip(em_rates_tot, E_values):
        em_rate_di, em_rate_ai = em_rates_from_E_DA(em_rate_tot, E_value)
        em_rates_d.append(em_rate_di)
        em_rates_a.append(em_rate_ai)
    return em_rates_d, em_rates_a

def populations_diff_coeff(particles, populations):
    """Diffusion coefficients of the two specified populations.
    """
    D_counts = particles.diffusion_coeff_counts
    if len(D_counts) == 1:
        pop_sizes = [pop.stop - pop.start for pop in populations]
        assert D_counts[0][1] >= sum(pop_sizes)
        D_counts = [(D_counts[0][0], ps) for ps in pop_sizes]

    D_list = []
    D_pop_start = 0  # start index of diffusion-based populations
    for pop, (D, counts) in zip(populations, D_counts):
        D_list.append(D)
        assert pop.start >= D_pop_start
        assert pop.stop <= D_pop_start + counts
        D_pop_start += counts
    return D_list

def populations_slices(particles, num_pop_list):
    """2-tuple of slices for selection of two populations.
    """
    slices = []
    i_prev = 0
    for num_pop in num_pop_list:
        slices.append(slice(i_prev, i_prev + num_pop))
        i_prev += num_pop
    return slices


class TimestapSimulation:
    """Simulate timestamps for a mixture of two populations.

    Attributes set by input arguments:

    1. Sequences with one element per population:

    - `em_rates`, `E_values`, `num_particles`

    2. Scalars (mandatory):

    - `bg_rate_d`, `bg_rate_a`

    3. Scalars (optional):

    - `timeslice`

    Attributes created by __init__():

    - `em_rates_d`, `em_rates_a`, `D_values`, `populations`, `traj_filename`.

    Attributes created by .run():

    - `hash_d`, `hash_a`

    Attributes created by .merge_da():

    - `ts`, `a_ch`, `part`, `clk_p`
    """

    def __init__(self, S, em_rates, E_values, num_particles,
                 bg_rate_d, bg_rate_a, timeslice=None):
        assert np.sum(num_particles) <= S.num_particles
        if timeslice is None:
            timeslice = S.t_max
        assert timeslice <= S.t_max

        em_rates_d, em_rates_a = em_rates_from_E_DA_mix(em_rates, E_values)
        populations = populations_slices(S.particles, num_particles)
        D_values = populations_diff_coeff(S.particles, populations)
        assert (len(em_rates) == len(E_values) == len(num_particles) ==
                len(populations) == len(D_values))

        params = dict(S=S, em_rates=em_rates, E_values=E_values,
                      num_particles=num_particles, bg_rate_d=bg_rate_d,
                      bg_rate_a=bg_rate_a, timeslice=timeslice,
                      em_rates_d=em_rates_d, em_rates_a=em_rates_a,
                      D_values=D_values, populations=populations,
                      traj_filename = S.store.filepath.name)

        for k, v in params.items():
            setattr(self, k, v)

    txt_header = """
        Timestamps simulation: Mixture
        ------------------------------

        Trajectories file:
            {self.traj_filename}
            time slice: {self.timeslice} s
        """
    txt_population = """
        Population{p_i}:
            # particles:        {num_pop} (first particle {pop.start})
            D                   {D} m^2/s
            Peak emission rate: {em_rate:,.0f} cps
            FRET efficiency:    {E:7.1%}
        """
    txt_background = """
        Background:
            Donor:              {self.bg_rate_d:7,} cps
            Acceptor:           {self.bg_rate_a:7,} cps
        """
    def __str__(self):
        txt = [self.txt_header.format(self=self)]
        pop_params = (self.em_rates, self.E_values, self.num_particles,
                      self.D_values, self.populations)
        for p_i, (em_rate, E, num_pop, D, pop) in enumerate(zip(*pop_params)):
            txt.append(self.txt_population.format(p_i=p_i + 1,
                num_pop=num_pop, D=D, em_rate=em_rate, E=E, pop=pop))

        txt.append(self.txt_background.format(self=self))
        return ''.join(txt)

    def summarize(self):
        print(str(self), flush=True)

    def _compact_repr(self):
        part_seq = ('%d_s%d' % (np, pop.start)
                    for np, pop in zip(self.num_particles, self.populations))
        s1 = 'P_' + '_'.join(part_seq)
        s2 = 'D_' + '_'.join('%.1e' % D for D in self.D_values)
        s3 = 'E_' + '_'.join('%d' % (E * 100) for E in self.E_values)
        s4 = 'EmTot_' + '_'.join('%dk' % (em * 1e-3) for em in self.em_rates)
        s5 = 'BgD%d_BgA%d' % (self.bg_rate_d, self.bg_rate_a)
        s6 = 't_max_%ds' % self.timeslice
        return '_'.join((s1, s2, s3, s4, s5, s6))

    @property
    def filename(self):
        hash_ = self.S.store.filepath.stem.split('_')[1]
        return "smFRET_%s_%s.hdf5" % (hash_, self._compact_repr())

    @property
    def filepath(self):
        return Path(self.S.store.filepath.parent, self.filename)

    def _calc_hash_da(self, rs):
        """Compute hash of D and A timestamps for single-step D+A case.
        """
        self.hash_d = hash_(rs.get_state())[:6]
        self.hash_a = self.hash_d

    def run(self, rs, overwrite=True, skip_existing=False, path=None,
            chunksize=None):
        """Compute timestamps for current populations."""
        if path is None:
            path = str(self.S.store.filepath.parent)
        kwargs = dict(rs=rs, overwrite=overwrite, path=path,
                      timeslice=self.timeslice, skip_existing=skip_existing)
        if chunksize is not None:
            kwargs['chunksize'] = chunksize
        header = ' - Mixture Simulation:'

        # Donor timestamps hash is from the input RandomState
        self.hash_d = hash_(rs.get_state())[:6]   # needed by merge_da()
        print('%s Donor timestamps -    %s' % (header, ctime()), flush=True)
        self.S.simulate_timestamps_mix(
            populations = self.populations,
            max_rates = self.em_rates_d,
            bg_rate = self.bg_rate_d,
            **kwargs)

        # Acceptor timestamps hash is from 'last_random_state' attribute
        # of the donor timestamps. This allows deterministic generation of
        # donor + acceptor timestamps given the input random state.
        ts_d, _ = self.S.get_timestamps_part(self.name_timestamps_d)
        rs.set_state(ts_d.attrs['last_random_state'])
        self.hash_a = hash_(rs.get_state())[:6]   # needed by merge_da()
        print('\n%s Acceptor timestamps - %s' % (header, ctime()), flush=True)
        self.S.simulate_timestamps_mix(
            populations = self.populations,
            max_rates = self.em_rates_a,
            bg_rate = self.bg_rate_a,
            **kwargs)
        print('\n%s Completed. %s' % (header, ctime()), flush=True)

    def run_da(self, rs, overwrite=True, skip_existing=False, path=None,
               chunksize=None):
        """Compute timestamps for current populations."""
        if path is None:
            path = str(self.S.store.filepath.parent)
        kwargs = dict(rs=rs, overwrite=overwrite, path=path,
                      timeslice=self.timeslice, skip_existing=skip_existing)
        if chunksize is not None:
            kwargs['chunksize'] = chunksize
        header = ' - Mixture Simulation:'

        # Donor timestamps hash is from the input RandomState
        self._calc_hash_da(rs)
        print('%s Donor + Acceptor timestamps - %s' %
              (header, ctime()), flush=True)
        self.S.simulate_timestamps_mix_da(
            max_rates_d = self.em_rates_d,
            max_rates_a = self.em_rates_a,
            populations = self.populations,
            bg_rate_d = self.bg_rate_d,
            bg_rate_a = self.bg_rate_a,
            **kwargs)
        print('\n%s Completed. %s' % (header, ctime()), flush=True)


    @property
    def name_timestamps_d(self):
        names_d = self.S.timestamps_match_mix(self.em_rates_d, self.populations,
                                              self.bg_rate_d, self.hash_d)
        assert len(names_d) == 1
        return names_d[0]

    @property
    def name_timestamps_a(self):
        names_a = self.S.timestamps_match_mix(self.em_rates_a, self.populations,
                                              self.bg_rate_a, self.hash_a)
        assert len(names_a) == 1
        return names_a[0]


    def merge_da(self):
        """Merge donor and acceptor timestamps, computes `ts`, `a_ch`, `part`.
        """
        print(' - Merging D and A timestamps', flush=True)
        ts_d, ts_par_d = self.S.get_timestamps_part(self.name_timestamps_d)
        ts_a, ts_par_a = self.S.get_timestamps_part(self.name_timestamps_a)
        ts, a_ch, part = merge_da(ts_d, ts_par_d, ts_a, ts_par_a)
        assert a_ch.sum() == ts_a.shape[0]
        assert (-a_ch).sum() == ts_d.shape[0]
        assert a_ch.size == ts_a.shape[0] + ts_d.shape[0]
        self.ts, self.a_ch, self.part = ts, a_ch, part
        self.clk_p = ts_d.attrs['clk_p']

    def _make_photon_hdf5(self, identity=None):

        # globals: S.ts_store.filename, S.t_max
        photon_data = dict(
            timestamps = self.ts,
            timestamps_specs = dict(timestamps_unit=self.clk_p),
            detectors = self.a_ch.view('uint8'),
            particles = self.part,
            measurement_specs = dict(
                measurement_type = 'smFRET',
                detectors_specs = dict(spectral_ch1 = np.atleast_1d(0),
                                       spectral_ch2 = np.atleast_1d(1))))

        setup = dict(
            num_pixels = 2,
            num_spots = 1,
            num_spectral_ch = 2,
            num_polarization_ch = 1,
            num_split_ch = 1,
            modulated_excitation = False,
            lifetime = False)

        provenance = dict(filename=self.S.ts_store.filename,
                          software='PyBroMo', software_version=__version__)

        if identity is None:
            identity = dict()

        description = self.__str__()
        acquisition_duration = self.timeslice
        data = dict(
            acquisition_duration = round(acquisition_duration),
            description = description,
            photon_data = photon_data,
            setup=setup,
            provenance=provenance,
            identity=identity)
        return data

    def save_photon_hdf5(self, identity=None, overwrite=True, path=None):
        """Create a smFRET Photon-HDF5 file with current timestamps."""
        filepath = self.filepath
        if path is not None:
            filepath = Path(path, filepath.name)
        self.merge_da()
        data = self._make_photon_hdf5(identity=identity)
        phc.hdf5.save_photon_hdf5(data, h5_fname=str(filepath),
                                  overwrite=overwrite)
