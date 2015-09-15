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

def em_rates_from_E_DA_mix(em_rate_tot1, em_rate_tot2, E_values1, E_values2):
    """D and A emission rates for two populations.
    """
    em_rates_d1, em_rates_a1 = em_rates_from_E_DA(em_rate_tot1, E_values1)
    em_rates_d2, em_rates_a2 = em_rates_from_E_DA(em_rate_tot2, E_values2)
    return em_rates_d1, em_rates_a1, em_rates_d2, em_rates_a2

def populations_diff_coeff(particles, num_pop1, num_pop2):
    """Diffusion coefficients of the two specified populations.
    """
    D_counts = particles.diffusion_coeff_counts
    if len(D_counts) == 1:
        D1 = D2 = D_counts[0][0]
    elif len(D_counts) == 2:
        # Multiple diffusion coefficients
        (D1, _num_pop1), (D2, _num_pop2) = D_counts
        assert _num_pop1 == num_pop1
        assert _num_pop2 == num_pop2
    return D1, D2

def populations_slices(particles, num_pop1, num_pop2):
    """2-tuple of slices for selection of two populations.
    """
    return slice(0, num_pop1), slice(num_pop1, num_pop1 + num_pop2)

class MixtureSimulation:
    """Simulate timestamps for a mixture of two populations."""
    def __init__(self, S, params):
        self.S = S
        for k, v in params.items():
            setattr(self, k, v)
        self.E1p, self.E2p = self.E1 * 100, self.E2 * 100
        rates = em_rates_from_E_DA_mix(self.em_rate_tot1, self.em_rate_tot2,
                                       self.E1, self.E2)
        self.em_rate_d1, self.em_rate_a1 = rates[:2]
        self.em_rate_d2, self.em_rate_a2 = rates[2:]
        self.em_rate_d1k, self.em_rate_a1k = rates[0] * 1e-3, rates[1] * 1e-3
        self.em_rate_d2k, self.em_rate_a2k = rates[2] * 1e-3, rates[3] * 1e-3
        self.max_rates_d = (self.em_rate_d1, self.em_rate_d2)
        self.max_rates_a = (self.em_rate_a1, self.em_rate_a2)
        self.bg_rates = [self.bg_rate_a, self.bg_rate_d]
        self.populations = populations_slices(S.particles,
                                              self.num_pop1, self.num_pop2)
        self.D1, self.D2 = populations_diff_coeff(S.particles,
                                                  self.num_pop1, self.num_pop2)
        self.traj_filename = S.store.filepath.name

    def __str__(self):
        s = """
        Timestamps simulation: Mixture
        ------------------------------

        Trajectories file:
            {self.traj_filename}

        Population1:
            # particles:        {self.num_pop1:7}
            D                   {self.D1} m^2/s
            Peak emission rate: {self.em_rate_tot1:,.0f} cps
            FRET efficiency:    {self.E1:7.1%}

        Population2:
            # particles:        {self.num_pop2:7}
            D                   {self.D2} m^2/s
            Peak emission rate: {self.em_rate_tot2:,.0f} cps
            FRET efficiency:    {self.E2:7.1%}

        Background:
            Donor:              {self.bg_rate_d:7,} cps
            Acceptor:           {self.bg_rate_a:7,} cps
        """.format(self=self)
        return s

    def summarize(self):
        print(str(self), flush=True)

    def _compact_repr(self):
        return ('_E1_{self.E1p:.0f}_D1Em{self.em_rate_d1k:.0f}k_A1Em{self.em_rate_a1k:.0f}'
                '_E2_{self.E2p:.0f}_D2Em{self.em_rate_d2k:.0f}k_A2Em{self.em_rate_a2k:.0f}'
                '_BgD{self.bg_rate_d}_BgA{self.bg_rate_a}'
               ).format(self=self)

    @property
    def filename(self):
        basename = self.S.store.filepath.stem.replace('pybromo', 'smFRET')
        return "%s%s.hdf5" % (basename, self._compact_repr())

    @property
    def filepath(self):
        return Path(self.filename)

    def run(self, rs, overwrite=True, path=None, chunksize=None):
        if path is None:
            path = str(self.S.store.filepath.parent)
        kwargs = dict(rs=rs, overwrite=overwrite, path=path)
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

    def save_photon_hdf5(self, identity=None, overwrite=True):
        self.merge_da()
        data = self._make_photon_hdf5(identity=identity)
        phc.hdf5.save_photon_hdf5(data, h5_fname=str(self.filepath),
                                  overwrite=overwrite)
