#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
Functions to manage simulations.
"""
from __future__ import print_function, absolute_import, division

import numpy as np
from glob import glob

from .storage import TrajectoryStore
from .psflib import NumericPSF
from .diffusion import ParticlesSimulation


def load_trajectories(fname, path='./'):
    fnames = glob(fname)
    if len(fnames) > 1:
        raise ValueError('Glob matched more than 1 file!')
    store = TrajectoryStore(fnames[0], overwrite=False)

    psf_pytables = store.data_file.get_node('/psf/default_psf')
    psf = NumericPSF(psf_pytables=psf_pytables)
    box = store.data_file.get_node_attr('/parameters', 'box')
    P = store.data_file.get_node_attr('/parameters', 'particles')

    names = ['t_step', 't_max', 'EID', 'ID']
    kwargs = {name: store.numeric_params[name] for name in names}
    S = ParticlesSimulation(particles=P, box=box, psf=psf, **kwargs)

    # Emulate S.open_store()
    S.store = store
    S.store_fname = fnames[0]
    S.psf_pytables = psf_pytables
    S.traj_group = S.store.data_file.root.trajectories
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
    offsets = np.arange(len(times_list)) * time_block
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
    P = reduce(lambda x, y: x + y, [Si.particles for Si in SS])
    s = SS[0]
    S = ParticlesSimulation(t_step=s.t_step, t_max=s.t_max,
                            particles=P, box=s.box, psf=s.psf)
    S.em = np.zeros(s.em.shape, dtype=np.float64)
    for Si in SS:
        S.em += Si.em
    return S
