#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#
"""
Legacy functions not used in recent versions.
"""

import numpy as np


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
