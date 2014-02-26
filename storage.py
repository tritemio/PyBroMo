# -*- coding: utf-8 -*-
"""
PyBroMo - A single molecule diffusion simulator in confocal geometry.

Copyright (C) 2013 Antonino Ingargiola tritemio@gmail.com

This module implements functions to store simulation results to a file.
The module uses the HDF5 file format through the PyTables library.
"""

import tables
import numpy as np


# Compression filter used by default for arrays
default_compression = tables.Filters(complevel=6, complib='blosc')


class Storage(object):
    def __init__(self, fname, nparams=dict(), attr_params=dict(), 
                 overwrite=True):
        """Return a new HDF5 file to store simulation results.

        The HDF5 file has two groups:
        '/parameters'
            containing all the simulation numeric-parameters

        '/trajectories'
            containing simulation trajectories (positions, emission traces)

        If `oldfile=False` (default) `fname` will be overwritten (if exists).
        """
        if not overwrite:
            # Create a new empty file
            self.data_file = tables.open_file(fname, mode = "a")
        else:
            # Create a new empty file
            self.data_file = tables.open_file(fname, mode = "w",
                               title = "Brownian motion simulation")
            # Create the groups
            self.data_file.create_group('/', 'trajectories',
                                   'Simulated trajectories')
            self.data_file.create_group('/', 'parameters',
                                   'Simulation parameters')
            self.data_file.create_group('/', 'psf',
                                   'PSFs used in the simulation')
            self.data_file.create_group('/', 'timestamps',
                                   'Timestamps of emitted photons')   
            # Set the simulation parameters
            self.set_sim_params(nparams, attr_params)

    def close(self):
        self.data_file.close()

    def open(self):
        """Reopen a file after has been closed (uses the store filename)."""
        self.__init__(self.data_file.filename, overwrite=False)

    def set_sim_params(self, nparams, attr_params):
        """Store parameters in `params` in `data_file.root.parameters`.

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
        for name, value in nparams.iteritems():
            self.data_file.create_array('/parameters', name, obj=value[0],
                                        title=value[1])
        for name, value in attr_params.items():
                self.data_file.set_node_attr('/parameters', name, value)
    
    def get_sim_nparams(self):
        """Return a dict containing all (key, values) stored in '/parameters'
        """
        nparams = dict()
        for p in self.data_file.root.parameters:
            nparams[p.name] = p.read()
        return nparams

    def get_sim_nparams_meta(self):
        """Return a dict with all parameters and metadata in '/parameters'.

        This returns the same dict format as returned by get_params() method
        in ParticlesSimulation().
        """
        nparams = dict()
        for p in self.data_file.root.parameters:
            nparams[p.name] = (p.read(), p.title)
        return nparams

    def add_timestamps(self, name, clk_p, max_rate, bg_rate,
                       num_particles, bg_particle,
                       overwrite=False, chunksize=2**16, 
                       comp_filter=default_compression):
        if name in self.data_file.root.timestamps:
            if overwrite:
                self.data_file.remove_node('/timestamps', name=name)
                self.data_file.remove_node('/timestamps', name=name+'_par')
            else:
                raise ValueError('Timestam array already exist (%s)' % name)

        times_array = self.data_file.create_earray(
            '/timestamps', name, atom=tables.Int64Atom(),
            shape = (0,),
            chunkshape = (chunksize,),
            filters = comp_filter,
            title = 'Simulated photon timestamps')
        times_array.set_attr('clk_p', clk_p)
        times_array.set_attr('max_rate', max_rate)
        times_array.set_attr('bg_rate', bg_rate)        
        particles_array = self.data_file.create_earray(
            '/timestamps', name+'_par', atom=tables.UInt8Atom(),
            shape = (0,),
            chunkshape = (chunksize,),
            filters = comp_filter,
            title = 'Particle number for each timestamp')
        particles_array.set_attr('num_particles', num_particles)
        particles_array.set_attr('bg_particle', bg_particle)
        return times_array, particles_array

    def add_trajectory(self, name, overwrite=False, shape=(0,), title='',
                  chunksize=2**19, comp_filter=default_compression,
                  atom=tables.Float64Atom(), params=dict()):
        """Add an trajectory array in '/trajectories'.
        """
        group = self.data_file.root.trajectories
        if name in group:
            print "%s already exists ..." % name,
            if overwrite:
                self.data_file.remove_node(group, name)
                print " deleted."
            else:
                print " old returned."
                return group.get_node(name)

        nparams = self.get_sim_nparams()
        num_t_steps = nparams['t_max']/nparams['t_step']

        if len(shape) == 1:
            chunkshape = (chunksize,)
        elif len(shape) == 2:
            chunkshape = (shape[0], chunksize/shape[0],)
        elif len(shape) == 3:
            chunkshape = (shape[0], shape[1], chunksize/(shape[0]*shape[1]),)

        store_array = self.data_file.create_earray(
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

    def add_emission_tot(self, chunksize=2**19, 
                         comp_filter=default_compression,
                         overwrite=False, params=dict()):
        """Add the `emission_tot` array in '/trajectories'.
        """
        return self.add_trajectory('emission_tot', overwrite=overwrite,
                chunksize=chunksize, comp_filter=comp_filter,
                atom=tables.Float32Atom(),
                title = 'Summed emission trace of all the particles',
                params=params)

    def add_emission(self, chunksize=2**19, comp_filter=default_compression,
                     overwrite=False, params=dict()):
        """Add the `emission` array in '/trajectories'.
        """
        nparams = self.get_sim_nparams()
        num_particles = nparams['np']

        return self.add_trajectory('emission', shape=(num_particles, 0),
                overwrite=overwrite, chunksize=chunksize,
                comp_filter=comp_filter, atom=tables.Float32Atom(),
                title = 'Emission trace of each particle',
                params=params)
    
    def add_position(self, chunksize=2**19, comp_filter=default_compression,
                     overwrite=False, params=dict()):
        """Add the `position` array in '/trajectories'.
        """
        nparams = self.get_sim_nparams()
        num_particles = nparams['np']

        return self.add_trajectory('emission', shape=(num_particles, 3, 0),
                overwrite=overwrite, chunksize=chunksize,
                comp_filter=comp_filter, atom=tables.Float32Atom(),
                title = 'Position trace of each particle',
                params=params)

    def add_timetrace_tot(self, chunksize=2**19, 
                          comp_filter=default_compression,
                          overwrite=False):
        """Add the `timetrace_tot` array in '/trajectories'.
        """
        return self.add_trajectory('timetrace_tot', overwrite=overwrite,
                chunksize=chunksize, comp_filter=comp_filter,
                atom=tables.UInt8Atom(),
                title = 'Timetrace of emitted photons with bin = t_step')

    def add_timetrace(self, chunksize=2**19, comp_filter=default_compression,
                      overwrite=False):
        """Add the `timetrace` array in '/trajectories'.
        """
        group = self.data_file.root.trajectories
        nparams = self.get_sim_nparams()
        num_particles = nparams['np']
        num_t_steps = nparams['t_max']/nparams['t_step']
        dt = np.dtype([('counts', 'u1')])
        timetrace_p = []
        for particle in xrange(num_particles):
            name = 'timetrace_p' + str(particle)
            if name in group:
                print "%s already exists ..." % name,
                if overwrite:
                    self.data_file.remove_node(group, name)
                    print " deleted."
                else:
                    print " using the old one."
                    timetrace_p.append(group.get_node(name))
                    continue
            timetrace_p.append(
                    self.data_file.create_table(
                        group, name, description=dt, chunkshape=chunksize,
                        expectedrows=num_t_steps,
                        title='Binned timetrace of emitted ph (bin = t_step)'
                            ' - particle_%d' % particle)
                    )
        return timetrace_p


if __name__ == '__main__':
    store = Storage('h2.h5', {'D': (1.2e-11, 'Diffusion coefficient (m^2/s)'),
                           'EID': (0, 'IPython engine ID (int)'),
                           'ID': (0, 'Simulation ID (int)'),
                           'np': (40, 'Number of simulated particles'),
                           'pico_mol': (86.4864063019005,
                                        'Particles concentration (pM)'),
                           't_max': (0.1, 'Simulation total time (s)'),
                           't_step': (5e-07, 'Simulation time-step (s)')})

#    em_tot_array = add_em_tot_array(hf)
#    em_array = add_em_array(hf)
#
#    #%%timeit -n1 -r1
#    for i in xrange(0, int(n_rows/chunksize)):
#        em_tot_array.append(np.random.rand(chunksize))
#    em_tot_array.flush()
#
#
#    #%%timeit -n1 -r1
#    for i in xrange(0, int(n_rows/chunksize)):
#        em_array.append(np.random.rand(chunksize, num_particles))
#    em_array.flush()
#
