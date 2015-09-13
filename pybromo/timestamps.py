#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module contains functions to work with timestamps.
"""

import numpy as np


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
