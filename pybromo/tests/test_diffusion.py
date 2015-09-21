"""
Module containing automated unit tests for PyBroMo.

Running the tests requires `py.test`.
"""

from __future__ import division
from builtins import range, zip

import pytest
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

def create_diffusion_sim():
    rs = np.random.RandomState(_SEED)
    Du = 12.0            # um^2 / s
    D = Du * (1e-6)**2    # m^2 / s
    box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)
    psf = pbm.NumericPSF()
    P = pbm.Particles(num_particles=100, D=D, box=box, rs=rs)
    t_step = 0.5e-6
    t_max = 0.1
    S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max,
                                particles=P, box=box, psf=psf)
    S.simulate_diffusion(save_pos=True, total_emission=False, radial=True,
                         rs=rs)
    S.store.close()
    return S.hash()[:6]

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
    saved_rs = S.traj_group._v_attrs['init_random_state']
    assert randomstate_equal(saved_rs, rs_prediffusion)
    saved_rs = S.traj_group._v_attrs['last_random_state']
    assert randomstate_equal(saved_rs, rs_postdiffusion)

def test_diffusion_sim_core():
    # Initialize the random state
    rs = np.random.RandomState(_SEED)
    Du = 12.0            # um^2 / s
    D = Du * (1e-6)**2    # m^2 / s
    box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)
    psf = pbm.NumericPSF()
    P = pbm.Particles(num_particles=100, D=D, box=box, rs=rs)
    t_step = 0.5e-6
    t_max = 0.001
    time_size = t_max / t_step
    assert t_max < 1e4
    S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max,
                                particles=P, box=box, psf=psf)

    start_pos = [p.r0 for p in S.particles]
    start_pos = np.vstack(start_pos).reshape(S.num_particles, 3, 1)

    for wrap_func in [pbm.diffusion.wrap_mirror, pbm.diffusion.wrap_periodic]:
        for total_emission in [True, False]:
            sim = S._sim_trajectories(time_size, start_pos, rs=rs,
                                      total_emission=total_emission,
                                      save_pos=True, wrap_func=wrap_func)

    POS, em = sim
    POS = np.concatenate(POS, axis=0)
    #x, y, z = POS[:, :, 0], POS[:, :, 1], POS[:, :, 2]
    #r_squared = x**2 + y**2 + z**2

    DR = np.diff(POS, axis=2)
    dx, dy, dz = DR[:, :, 0], DR[:, :, 1], DR[:, :, 2]
    dr_squared = dx**2 + dy**2 + dz**2

    D_fitted = dr_squared.mean() / (6 * t_max)  # Fitted diffusion coefficient
    assert np.abs(D - D_fitted) < 0.01


def test_simulate_timestamps():
    hash_ = create_diffusion_sim()
    S = pbm.ParticlesSimulation.from_datafile(hash_, mode='w')

    rs = np.random.RandomState(_SEED)
    kw = dict(max_rates=(400e3,), populations=(slice(0, 35),), bg_rate=1000,
              rs=rs)
    S.simulate_timestamps_mix(**kw)

    # The following two cases should not throw an error
    kw.update(overwrite=True, skip_existing=True,
              rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)
    kw.update(overwrite=True, skip_existing=False,
              rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)

    # This should still pass
    kw.update(overwrite=False, skip_existing=True,
              rs=np.random.RandomState(_SEED))
    S.simulate_timestamps_mix(**kw)

    # This shoudl throw an ExistingArrayError
    kw.update(overwrite=False, skip_existing=False,
              rs=np.random.RandomState(_SEED))
    with pytest.raises(pbm.storage.ExistingArrayError):
        S.simulate_timestamps_mix(**kw)

    # But with a different initial random state shoud succeede
    kw.pop('rs')
    S.simulate_timestamps_mix(**kw)
    S.store.close()

def test_TimestampSimulation():
    hash_ = create_diffusion_sim()
    S = pbm.ParticlesSimulation.from_datafile(hash_, mode='a')

    params = dict(
        em_rates = (400e3,),    # Peak emission rates (cps) for each population (D+A)
        E_values = (0.75,),     # FRET efficiency for each population
        num_particles = (35,),  # Number of particles in each population
        bg_rate_d = 1400,       # Poisson background rate (cps) Donor channel
        bg_rate_a = 800,        # Poisson background rate (cps) Acceptor channel
        )

    mix_sim = pbm.TimestapSimulation(S, **params)
    mix_sim.summarize()

    rs = np.random.RandomState(_SEED)
    mix_sim.run(rs=rs, overwrite=False)
    mix_sim.save_photon_hdf5()