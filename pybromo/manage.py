#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
Functions to manage simulations.
"""
from __future__ import print_function, absolute_import, division

from pathlib import Path

from .storage import TrajectoryStore
from .psflib import NumericPSF
from .diffusion import ParticlesSimulation


def load_trajectories(fname, path='./'):
    path = Path(path)
    assert path.exists()
    fnames = list(path.glob(fname))
    if len(fnames) > 1:
        raise ValueError('Glob matched more than 1 file!')
    store = TrajectoryStore(str(fnames[0]), mode='r')

    psf_pytables = store.h5file.get_node('/psf/default_psf')
    psf = NumericPSF(psf_pytables=psf_pytables)
    box = store.h5file.get_node_attr('/parameters', 'box')
    P = store.h5file.get_node_attr('/parameters', 'particles')

    names = ['t_step', 't_max', 'EID', 'ID']
    kwargs = {name: store.numeric_params[name] for name in names}
    S = ParticlesSimulation(particles=P, box=box, psf=psf, **kwargs)

    # Emulate S.open_store_traj()
    S.store = store
    S.store_fname = fnames[0]
    S.psf_pytables = psf_pytables
    S.traj_group = S.store.h5file.root.trajectories
    S.emission = S.store.h5file.root.trajectories.emission
    S.emission_tot = S.store.h5file.root.trajectories.emission_tot
    S.position = S.store.h5file.root.trajectories.position
    S.chunksize = S.store.h5file.get_node('/parameters', 'chunksize')
    return S
