"""
Module containing automated unit tests for PyBroMo.

Running the tests requires `py.test`.
"""

from __future__ import division
from builtins import range, zip

#import pytest
import numpy as np

import pybromo as pbm

_SEED = 2345654342


def randomstate_equal(rs1, rs2):
    if isinstance(rs1, np.random.RandomState):
        rs1 = rs1.get_state()
    assert isinstance(rs1, tuple)
    if isinstance(rs2, np.random.RandomState):
        rs2 = rs2.get_state()
    assert isinstance(rs1, tuple)
    assert len(rs1) == len(rs2)
    equal = True
    for x1, x2 in zip(rs1, rs2):
        test = x1 == x2
        if hasattr(test, '__array__'):
            test = test.all()
        equal &= test
    return equal

def test_Particles():
    rs = np.random.RandomState(_SEED)
    box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)
    D1 = 12e-12
    D2 = D1 / 2
    P = pbm.Particles(num_particles=20, D=D1, box=box, rs=rs)
    P.add(num_particles=15, D=D2)

    Di, counts = zip(*P.diffusion_coeff_counts)
    rs2 = np.random.RandomState()
    rs2.set_state(P.init_random_state)
    P2_list = pbm.Particles._generate(num_particles=counts[0], D=Di[0],
                                      box=P.box, rs=rs2)
    P2_list += pbm.Particles._generate(num_particles=counts[1], D=Di[1],
                                       box=P.box, rs=rs2)
    assert P.to_list() == P2_list

    # Test Particles random states
    assert randomstate_equal(P.rs, rs.get_state())
    assert randomstate_equal(P.init_random_state, np.random.RandomState(_SEED))
    assert not randomstate_equal(P.init_random_state, P.rs)

def test_diffusion_sim_random_state():
        # Initialize the random state
    rs = np.random.RandomState(_SEED)

    # Diffusion coefficient
    Du = 12.0            # um^2 / s
    D1 = Du * (1e-6)**2    # m^2 / s
    D2 = D1 / 2

    # Simulation box definition
    box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)

    # PSF definition
    psf = pbm.NumericPSF()

    # Particles definition
    P = pbm.Particles(num_particles=20, D=D1, box=box, rs=rs)
    P.add(num_particles=15, D=D2)

    # Simulation time step (seconds)
    t_step = 0.5e-6

    # Time duration of the simulation (seconds)
    t_max = 0.01

    # Particle simulation definition
    S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max,
                                particles=P, box=box, psf=psf)

    rs_prediffusion = rs.get_state()
    S.simulate_diffusion(total_emission=False, save_pos=True, verbose=True,
                         rs=rs, chunksize=2**13, chunkslice='times')
    rs_postdiffusion = rs.get_state()

    # Test diffusion random states
    saved_rs = S._load_group_attr('/trajectories', 'init_random_state')
    assert randomstate_equal(saved_rs, rs_prediffusion)
    saved_rs = S._load_group_attr('/trajectories', 'last_random_state')
    assert randomstate_equal(saved_rs, rs_postdiffusion)
