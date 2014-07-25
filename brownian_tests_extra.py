"""
PyBroMo - A single molecule diffusion simulator in confocal geometry.

Copyright (C) 2013-2014 Antonino Ingargiola tritemio@gmail.com

This module contains alternative (all-in-RAM) implementations of the
chunked timestamp simulation in brownian.py. These versions are used for
testing the pure algorithm of chunk-by-chunk timestamp generation without
the added complexity of using pytables.
"""

import numpy as np
from itertools import izip
from brownian import (iter_chunk_index, sim_timetrace, sim_timetrace_bg,
                      sim_timetrace_bg2,)


def sim_timestamps_em_list(S, max_rate=1, bg_rate=0, rs=None, seed=None):
    """Compute timestamps and particles and store results in a list.
    Each element contains timestamps from one chunk of emission.
    Background computed internally.
    """
    if rs is None:
        rs = np.random.RandomState(seed=seed)
    fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
    scale = 10
    max_counts = 4

    S.all_times_chunks_list = []
    S.all_par_chunks_list = []

    # Load emission in chunks, and save only the final timestamps
    for i_start, i_end in iter_chunk_index(S.n_samples,
                                           S.emission.chunkshape[1]):
        counts_chunk = sim_timetrace(S.emission[:, i_start:i_end],
                                     max_rate, S.t_step)
        counts_bg_chunk = rs.poisson(bg_rate*S.t_step,
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
        S.all_times_chunks_list.append(times_chunk_s)
        S.all_par_chunks_list.append(par_index_chunk_s)

def sim_timestamps_em_list1(S, max_rate=1, bg_rate=0, rs=None, seed=None):
    """Compute timestamps and particles and store results in a list.
    Each element contains timestamps from one chunk of emission.
    Background computed in sim_timetrace_bg() as last fake particle.
    """
    if rs is None:
        rs = np.random.RandomState(seed=seed)
    fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
    scale = 10
    max_counts = 4

    S.all_times_chunks_list = []
    S.all_par_chunks_list = []

    # Load emission in chunks, and save only the final timestamps
    for i_start, i_end in iter_chunk_index(S.n_samples,
                                           S.emission.chunkshape[1]):
        counts_chunk = sim_timetrace_bg(S.emission[:, i_start:i_end],
                                     max_rate, bg_rate, S.t_step, rs=rs)
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
        S.all_times_chunks_list.append(times_chunk_s)
        S.all_par_chunks_list.append(par_index_chunk_s)

def sim_timestamps_em_list2(S, max_rate=1, bg_rate=0, rs=None, seed=None):
    """Compute timestamps and particles and store results in a list.
    Each element contains timestamps from one chunk of emission.
    Background computed in sim_timetrace_bg2() as last fake particle.
    """
    if rs is None:
        rs = np.random.RandomState(seed=seed)
    fractions = [5, 2, 8, 4, 9, 1, 7, 3, 6, 9, 0, 5, 2, 8, 4, 9]
    scale = 10
    max_counts = 4

    S.all_times_chunks_list = []
    S.all_par_chunks_list = []

    # Load emission in chunks, and save only the final timestamps
    for i_start, i_end in iter_chunk_index(S.n_samples,
                                           S.emission.chunkshape[1]):
        counts_chunk = sim_timetrace_bg2(S.emission[:, i_start:i_end],
                                     max_rate, bg_rate, S.t_step, rs=rs)
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
        S.all_times_chunks_list.append(times_chunk_s)
        S.all_par_chunks_list.append(par_index_chunk_s)