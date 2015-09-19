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

def populations_diff_coeff(particles, num_pop):
    """Diffusion coefficients of the two specified populations.
    """
    D_counts = particles.diffusion_coeff_counts
    if len(D_counts) == 1:
        D_list = [D_counts[0][0], D_counts[0][0]]
    else:
        D_list = []
        for p_i, (D, counts) in zip(num_pop, D_counts):
            D_list.append(D)
            assert p_i == counts
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
    """Simulate timestamps for a mixture of two populations."""

    def __init__(self, S, params):
        self.S = S
        self.params = params
        assert np.sum(params['num_pop']) <= S.num_particles

        em_rates_d, em_rates_a = em_rates_from_E_DA_mix(params['em_rates'],
                                                        params['E'])
        params.update(
            em_rates_d=em_rates_d,
            em_rates_a=em_rates_a,
            D_list=populations_diff_coeff(S.particles, params['num_pop']))

        self.bg_rates = [params['bg_rate_d'], params['bg_rate_a']]
        self.populations = populations_slices(S.particles,
                                              params['num_pop'])
        self.traj_filename = S.store.filepath.name

    txt_header = """
        Timestamps simulation: Mixture
        ------------------------------

        Trajectories file:
            {self.traj_filename}
        """
    txt_population = """
        Population{p_i}:
            # particles:        {num_pop}
            D                   {D} m^2/s
            Peak emission rate: {em_rate:,.0f} cps
            FRET efficiency:    {E:7.1%}
        """
    txt_background = """
        Background:
            Donor:              {bg_rate_d:7,} cps
            Acceptor:           {bg_rate_a:7,} cps
        """
    def __str__(self):
        txt = [self.txt_header.format(self=self)]
        p = self.params

        pop_params = (p['num_pop'], p['D_list'], p['em_rates'], p['E'])
        for p_i, (num_pop, D, em_rate, E) in enumerate(zip(*pop_params)):
            txt.append(self.txt_population.format(p_i=p_i + 1,
                num_pop=num_pop, D=D, em_rate=em_rate, E=E * 100))

        txt.append(self.txt_background.format(**self.params))
        return ''.join(txt)

    def summarize(self):
        print(str(self), flush=True)

    def _compact_repr(self):
        p = self.params
        s1 = 'P_' + '_'.join(str(n_p) for n_p in p['num_pop'])
        s2 = 'D_' + '_'.join('%.1e' % D for D in p['D_list'])
        s3 = 'E_' + '_'.join('%d' % (E * 100) for E in p['E'])
        s4 = 'EmTot_' + '_'.join('%d' % (em * 1e-3) for em in p['em_rates'])
        s5 = 'BgD%d_BgA%d' % (p['bg_rate_d'], p['bg_rate_a'])
        return '_'.join((s1, s2, s3, s4, s5))

    @property
    def filename(self):
        hash_ = self.S.store.filepath.stem.split('_')[1]
        return "smFRET_%s_%s.hdf5" % (hash_, self._compact_repr())

    @property
    def filepath(self):
        return Path(self.S.store.filepath.parent, self.filename)

    def run(self, rs, overwrite=True, path=None, chunksize=None,
            timeslice=None):
        if path is None:
            path = str(self.S.store.filepath.parent)
        kwargs = dict(rs=rs, overwrite=overwrite, path=path, timeslice=timeslice)
        if chunksize is not None:
            kwargs['chunksize'] = chunksize
        header = ' - Mixture Simulation:'
        print('%s Donor timestamps - %s' % (header, ctime()), flush=True)
        self.S.simulate_timestamps_mix(
            populations = self.populations,
            max_rates = self.max_rates_d,
            bg_rate = self.bg_rate_d,
            **kwargs)
        print('%s Acceptor timestamps - %s' % (header, ctime()), flush=True)
        self.S.simulate_timestamps_mix(
            populations = self.populations,
            max_rates = self.max_rates_a,
            bg_rate = self.bg_rate_a,
            **kwargs)
        print('%s Completed. %s' % (header, ctime()), flush=True)

    def merge_da(self):
        print(' - Merging D and A timestamps', flush=True)
        name_d = self.S.timestamps_match_mix(self.max_rates_d,
                                             self.populations,
                                             self.bg_rate_d)[0]
        name_a = self.S.timestamps_match_mix(self.max_rates_a,
                                             self.populations,
                                             self.bg_rate_a)[0]
        ts_d, ts_par_d = self.S.get_timestamps_part(name_d)
        ts_a, ts_par_a = self.S.get_timestamps_part(name_a)
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
        acquisition_duration = self.S.t_max
        data = dict(
            acquisition_duration = round(acquisition_duration),
            description = description,
            photon_data = photon_data,
            setup=setup,
            provenance=provenance,
            identity=identity)
        return data

    def save_photon_hdf5(self, identity=None, overwrite=True, path=None):
        filepath = self.filepath
        if path is not None:
            filepath = Path(path, filepath.name)
        self.merge_da()
        data = self._make_photon_hdf5(identity=identity)
        phc.hdf5.save_photon_hdf5(data, h5_fname=str(filepath),
                                  overwrite=overwrite)
