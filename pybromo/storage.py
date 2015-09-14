#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module implements functions to store simulation results to a file.
The module uses the HDF5 file format through the PyTables library.

File part of PyBroMo: a single molecule diffusion simulator.
Copyright (C) 2013-2014 Antonino Ingargiola tritemio@gmail.com
"""
from __future__ import print_function, absolute_import, division
from builtins import range, zip, dict

from pathlib import Path
import tables


# Compression filter used by default for arrays
default_compression = tables.Filters(complevel=5, complib='blosc')


class BaseStore(object):

    @staticmethod
    def calc_chunkshape(chunksize, shape, kind='bytes'):
        assert kind in ['times', 'bytes']
        if chunksize is None:
            return None

        if kind == 'bytes':
            divisor = 1
            for dimsize in shape[:-1]:
                divisor *= dimsize

        if len(shape) == 1:
            chunkshape = (chunksize,)
        elif len(shape) == 2:
            chunkshape = (shape[0], chunksize / divisor)
        elif len(shape) == 3:
            chunkshape = (shape[0], shape[1], chunksize / divisor)
        return chunkshape

    def __init__(self, datafile, path='./', nparams=dict(), attr_params=dict(),
                 mode='r'):
        """Return a new HDF5 file to store simulation results.

        The HDF5 file has two groups:
        '/parameters'
            containing all the simulation numeric-parameters

        If `mode='w'`, `datafile` will be overwritten (if exists).
        """
        if isinstance(datafile, Path):
            self.filepath = datafile
        else:
            assert Path(path).exists()
            self.filepath = Path(path, datafile)
        self.h5file = tables.open_file(str(self.filepath), mode=mode)
        self.filename = str(self.filepath)
        if mode != 'r':
            self.h5file.title = "PyBroMo simulation file"

            # Create the groups
            self.h5file.create_group('/', 'parameters', 'Simulation parameters')
            # Set the simulation parameters
            self.set_sim_params(nparams, attr_params)

    def close(self):
        self.h5file.close()

    def open(self):
        """Reopen a file after has been closed (uses the store filename)."""
        self.__init__(self.h5file.filename, mode='r')

    def set_sim_params(self, nparams, attr_params):
        """Store parameters in `params` in `h5file.root.parameters`.

        `nparams` (dict)
            A dict as returned by `get_params()` in `ParticlesSimulation()`
            The format is:
            keys:
                used as parameter name
            values: (2-elements tuple)
                first element is the parameter value
                second element is a string used as "title" (description)
        `attr_params` (dict)
            A dict whole items are stored as attributes in '/parameters'
        """
        for name, value in nparams.items():
            val = value[0] if value[0] is not None else 'none'
            self.h5file.create_array('/parameters', name, obj=val,
                                     title=value[1])
        for name, value in attr_params.items():
                self.h5file.set_node_attr('/parameters', name, value)

    @property
    def numeric_params(self):
        """Return a dict containing all (key, values) stored in '/parameters'
        """
        nparams = dict()
        for p in self.h5file.root.parameters:
            nparams[p.name] = p.read()
        return nparams

    @property
    def numeric_params_meta(self):
        """Return a dict with all parameters and metadata in '/parameters'.

        This returns the same dict format as returned by get_params() method
        in ParticlesSimulation().
        """
        nparams = dict()
        for p in self.h5file.root.parameters:
            nparams[p.name] = (p.read(), p.title)
        return nparams


class TrajectoryStore(BaseStore):
    def __init__(self, datafile, path='./', nparams=dict(), attr_params=dict(),
                 mode='r'):
        """Return a new HDF5 file to store simulation results.

        The HDF5 file has two groups:
        '/parameters'
            containing all the simulation numeric-parameters

        '/trajectories'
            containing simulation trajectories (positions, emission traces)

        If `mode='w'`, `datafile` will be overwritten (if exists).
        """
        super().__init__(datafile, path=path, nparams=nparams,
                         attr_params=attr_params, mode=mode)
        if mode != 'r':
            # Create the groups
            self.h5file.create_group('/', 'trajectories',
                                     'Simulated trajectories')
            self.h5file.create_group('/', 'psf', 'PSFs used in the simulation')

    def add_trajectory(self, name, overwrite=False, shape=(0,), title='',
                       chunksize=2**19, comp_filter=default_compression,
                       atom=tables.Float64Atom(), params=dict(),
                       chunkslice='bytes'):
        """Add an trajectory array in '/trajectories'.
        """
        group = self.h5file.root.trajectories
        if name in group:
            print("%s already exists ..." % name, end='')
            if overwrite:
                self.h5file.remove_node(group, name)
                print(" deleted.")
            else:
                print(" old returned.")
                return group.get_node(name)

        nparams = self.numeric_params
        num_t_steps = nparams['t_max'] / nparams['t_step']

        chunkshape = self.calc_chunkshape(chunksize, shape, kind=chunkslice)
        store_array = self.h5file.create_earray(
            group, name, atom=atom,
            shape = shape,
            chunkshape = chunkshape,
            expectedrows = num_t_steps,
            filters = comp_filter,
            title = title)

        # Set the array parameters/attributes
        for key, value in params.items():
            store_array.set_attr(key, value)
        return store_array

    def add_emission_tot(self, chunksize=2**19, comp_filter=default_compression,
                         overwrite=False, params=dict(),
                         chunkslice='bytes'):
        """Add the `emission_tot` array in '/trajectories'.
        """
        kwargs = dict(overwrite=overwrite, chunksize=chunksize, params=params,
                      comp_filter=comp_filter, atom=tables.Float32Atom(),
                      title='Summed emission trace of all the particles')
        return self.add_trajectory('emission_tot', **kwargs)

    def add_emission(self, chunksize=2**19, comp_filter=default_compression,
                     overwrite=False, params=dict(), chunkslice='bytes'):
        """Add the `emission` array in '/trajectories'.
        """
        nparams = self.numeric_params
        num_particles = nparams['np']

        return self.add_trajectory('emission', shape=(num_particles, 0),
                                   overwrite=overwrite, chunksize=chunksize,
                                   comp_filter=comp_filter,
                                   atom=tables.Float32Atom(),
                                   title='Emission trace of each particle',
                                   params=params)

    def add_position(self, chunksize=2**19, comp_filter=default_compression,
                     overwrite=False, params=dict(), chunkslice='bytes'):
        """Add the `position` array in '/trajectories'.
        """
        nparams = self.numeric_params
        num_particles = nparams['np']

        return self.add_trajectory('position', shape=(num_particles, 3, 0),
                                   overwrite=overwrite, chunksize=chunksize,
                                   comp_filter=comp_filter,
                                   atom=tables.Float32Atom(),
                                   title='3-D position trace of each particle',
                                   params=params)


class TimestampStore(BaseStore):
    def __init__(self, datafile, path='./', nparams=dict(), attr_params=dict(),
                 mode='r'):
        """Return a new HDF5 file to store simulation results.

        The HDF5 file has two groups:
        '/parameters'
            containing all the simulation numeric-parameters

        '/timestamps'
            containing simulated timestamps

        If `overwrite=True` (default) `datafile` is overwritten (if exists).
        """
        super().__init__(datafile, path=path, nparams=nparams,
                         attr_params=attr_params, mode=mode)
        if mode != 'r':
            # Create the groups
            self.h5file.create_group('/', 'timestamps', 'Simulated timestamps')

    def add_timestamps(self, name, clk_p, max_rate, bg_rate,
                       num_particles, bg_particle, populations=None,
                       overwrite=False, chunksize=2**16,
                       comp_filter=default_compression):
        if name in self.h5file.root.timestamps:
            if overwrite:
                self.h5file.remove_node('/timestamps', name=name)
                self.h5file.remove_node('/timestamps', name=name + '_par')
            else:
                raise ValueError('Timestamp array already exist (%s)' % name)

        times_array = self.h5file.create_earray(
            '/timestamps', name, atom=tables.Int64Atom(),
            shape = (0,),
            chunkshape = (chunksize,),
            filters = comp_filter,
            title = 'Simulated photon timestamps')
        times_array.set_attr('clk_p', clk_p)
        times_array.set_attr('max_rate', max_rate)
        times_array.set_attr('bg_rate', bg_rate)
        times_array.set_attr('populations', populations)
        particles_array = self.h5file.create_earray(
            '/timestamps', name + '_par', atom=tables.UInt8Atom(),
            shape = (0,),
            chunkshape = (chunksize,),
            filters = comp_filter,
            title = 'Particle number for each timestamp')
        particles_array.set_attr('num_particles', num_particles)
        particles_array.set_attr('bg_particle', bg_particle)
        return times_array, particles_array


if __name__ == '__main__':
    d = {'D': (1.2e-11, 'Diffusion coefficient (m^2/s)'),
         'EID': (0, 'IPython engine ID (int)'),
         'ID': (0, 'Simulation ID (int)'),
         'np': (40, 'Number of simulated particles'),
         'pico_mol': (86.4864063019005, 'Particles concentration (pM)'),
         't_max': (0.1, 'Simulation total time (s)'),
         't_step': (5e-07, 'Simulation time-step (s)')}
    store = TrajectoryStore('h2.h5', d)

#    em_tot_array = add_em_tot_array(hf)
#    em_array = add_em_array(hf)
#
#    #%%timeit -n1 -r1
#    for i in range(0, int(n_rows/chunksize)):
#        em_tot_array.append(np.random.rand(chunksize))
#    em_tot_array.flush()
#
#
#    #%%timeit -n1 -r1
#    for i in xrange(0, int(n_rows/chunksize)):
#        em_array.append(np.random.rand(chunksize, num_particles))
#    em_array.flush()
#
