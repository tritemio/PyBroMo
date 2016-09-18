#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
PyBroMo - A single molecule diffusion simulator in confocal geometry.

Copyright (C) 2013-2014 Antonino Ingargiola tritemio@gmail.com

This module implements iterator functions to loop over arrays in chunks.
"""

import numpy as np


def iter_chunksize(num_samples, chunksize):
    """Iterator used to iterate in chunks over an array of size `num_samples`.
    At each iteration returns `chunksize` except for the last iteration.
    """
    last_chunksize = int(np.mod(num_samples, chunksize))
    chunksize = int(chunksize)
    for _ in range(int(num_samples) // chunksize):
        yield chunksize
    if last_chunksize > 0:
        yield last_chunksize


def iter_chunk_slice(num_samples, chunksize):
    """Iterator used to iterate in chunks over an array of size `num_samples`.

    At each iteration returns a slice of size `chunksize`. In the last
    iteration the slice may be smaller.
    """
    i = 0
    for c_size in iter_chunksize(num_samples, chunksize):
        yield slice(i, i + c_size)
        i += c_size


def iter_chunk_index(num_samples, chunksize):
    """Iterator used to iterate in chunks over an array of size `num_samples`.

    At each iteration returns a start and stop index for a slice of size
    `chunksize`. In the last iteration the slice may be smaller.
    """
    i = 0
    for c_size in iter_chunksize(num_samples, chunksize):
        yield i, i + c_size
        i += c_size


def reduce_chunk(func, array):
    """Reduce with `func`, chunk by chunk, the passed pytable `array`.
    """
    res = []
    for slice in iter_chunk_slice(array.shape[-1], array.chunkshape[-1]):
        res.append(func(array[..., slice]))
    return func(res)


def map_chunk(func, array, out_array):
    """Map with `func`, chunk by chunk, the input pytable `array`.
    The result is stored in the output pytable array `out_array`.
    """
    for slice in iter_chunk_slice(array.shape[-1], array.chunkshape[-1]):
        out_array.append(func(array[..., slice]))
    return out_array
