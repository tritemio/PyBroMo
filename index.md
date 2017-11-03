# PyBroMo Overview

[![DOI](https://zenodo.org/badge/5991/tritemio/PyBroMo.svg)](https://zenodo.org/badge/latestdoi/5991/tritemio/PyBroMo)

<div>
<img title="Numerical PSF" src="https://cloud.githubusercontent.com/assets/4156237/11383966/b5781438-92c0-11e5-982c-0499b95dac43.png" height="110" />
<img title="Particles Trajectories" src="https://cloud.githubusercontent.com/assets/4156237/11383974/c3020bae-92c0-11e5-86d7-0f41055e2095.png" height="110" />
<img title="Simulated smFRET timetrace, bursts and FRET histogram" src="https://cloud.githubusercontent.com/assets/4156237/11384620/11051666-92c6-11e5-871e-041e71839f22.png" height="110" />
</div>

**[PyBroMo](http://tritemio.github.io/PyBroMo/)** is an open-source simulator
for Brownian-motion diffusion and photon emission of fluorescent particles
excited by a diffraction limited laser spot.
PyBroMo allows to simulate timestamps of photons emitted during
[smFRET](http://en.wikipedia.org/wiki/Single-molecule_FRET) experiments,
including sample background and detectors dark counts and to save the results in
in [Photon-HDF5](http://photon-hdf5.org) format. The smFRET data files can
be analyzed with any smFRET burst analysis software.

> For an opensource smFRET burst analysis software supporting Photon-HDF5 see [FRETBursts](https://github.com/tritemio/FRETBursts).

PyBromo simulates 3-D Brownian motion trajectories and emission of an
arbitrary number of particles freely diffusing in a simulation volume (a box).
Inside the simulation box a laser excitation volume (the
[PSF](http://en.wikipedia.org/wiki/Point_spread_function) of the objective lens)
is defined numerically or analytically (Gaussian shape). Particles diffusing
through the excitation volume emit photons at a rate proportional to the
local excitation intensity.

A precomputed numerical [PSF](http://en.wikipedia.org/wiki/Point_spread_function)
is included and used by default.
The included numerical PSF is computed through
rigorous vectorial electromagnetic computations ([Nasse, Woehl 2010]
(http://dx.doi.org/10.1364/JOSAA.27.000295)) using the
[PSFLab](http://onemolecule.chem.uwm.edu/software) software.
The user can provide a different numerical PSF or,
alternatively, use an analytical Gaussian-shaped PSF.

An overview of the architecture of the simulator can be found
[below](#architecture).

The user documentation is provided in a series of [Jupyter](http://jupyter.org) notebooks
(see **[Usage examples](#usage-examples)**).

# Feedback

If you have a question or find a bug in PyBroMo please open a GitHub Issue.
Bug fixes and/or enhancements are welcome, just send a [pull request (PR)](https://help.github.com/articles/using-pull-requests).

For more info contact me at tritemio@gmail.com.

# Environment

PyBroMo is written in the [python programming language](http://www.python.org/)
using the standard scientific stack of libraries (numpy, scipy, pytables,
matplotlib).

Usage examples are given as
Jupyter/IPython notebooks.
[Jupyter Notebook](http://jupyter.org/) is an interactive web-based environment that allows to mix rich text, math and graphics with (live) code.
You can find a static HTML version of the notebooks below in section **[Usage examples](#usage-examples)**.

Moreover the [IPython environment](http://ipython.org/) allows to easily setup a cluster for parallel computing. Therefore simulation time can be
greatly reduced using a single multi-core PC, multiple PC or a cloud-computing service.

If you are new to Jupyter Notebook, refer to this guide for installation and first steps:

- [Jupyter/IPython Notebook Quick Start Guide](http://jupyter-notebook-beginner-guide.readthedocs.org)

# Architecture

The simulation domain is defined as 3-D box, centered around the origin. 
As boundary conditions particles can be either reflected at the interface ("mirror" condition) 
or reinjected from the opposite face ("periodic" condition).

A particle is described by its initial position. A list of particles with random initial position 
is generated before running the diffusion simulation.

The excitation PSF is a function of the position and is centered with maximum on the origin. 
A realistic PSF obtained by vectorial electromagnetic simulation is precomputed using 
[PSFLab](http://onemolecule.chem.uwm.edu/software). The PSF is computed for a
water immersion objective (NA = 1.2) at 532 nm
and includes effects such as refractive index mismatch and mismatch between the objective lens 
correction and the cover-glass thickness. The user can  generate a different PSF using 
[PSFLab](http://onemolecule.chem.uwm.edu/software) or equivalent software. The PSF is generated 
using circular polarized light so it has cylindrical symmetry and it is precomputed only on the x-z plane.
Alternatively, a simple Gaussian PSF can also be used.

The Brownian motion parameters are: the diffusion coefficient, the simulation box, 
the list of particles, the simulation time step and the simulation duration.

The Brownian motion simulation uses constant time-steps (typically 0.5 μs).
This allows a straightforward and efficient implementation.
The total simulation time is divided in chunks so that trajectories for a single chunk
can easily fit in RAM. For each chunk, trajectories are computed by 
cumulative sum ([`cumsum`](http://docs.scipy.org/doc/numpy/reference/generated/numpy.cumsum.html)) 
of the array of Gaussian displacement.

The instantaneous emission rate of each particle is computed during the Brownian motion simulation
by evaluating the PSF intensity at each position. After the diffusion simulation, for each particle, 
photons are generated from a [Poisson process](http://en.wikipedia.org/wiki/Poisson_process) using the 
previously computed emission rates. An additional constant Poisson process models sample background 
and detectors' dark counts. The time bin in which a "count" (photon or background) is extracted
becomes the timestamp.

PyBroMo provides functions to simulate one or multiple FRET populations,
saving the results in regular smFRET data files in Photon-HDF5 format. For each timestamp,
the particle ID is also saved, allowing to separate the contribution of each particle.
Photo-physics effects, such as blinking and bleaching, are not explicily 
modeled but they can be easily included "modulating" the emission
rates before generating the photons.
Two-states systems (each state with a different FRET efficiency) can be also 
simulated. In this case, the user needs to generate a static smFRET data file for each state
(from the same diffusion trajectories). Next, transition times (switch-points) can be
computed (e.g. drawing exponetial random variables) for each particle until the simulation 
duration is covered. Finally, the user can create a new  
smFRET data file by selecting timestamps from each static-state file
according to the generated transitions.

As a final note, PyBroMo computations can be performed on a single core
or distributed on the nodes of a cluster (IPython cluster). 
Thanks to the IPython infrastructure the simulation can be seamless run on a single machine, 
on a cluster of machines or on a cloud computing server.

# Usage examples

PyBroMo includes a collection of notebooks which serves both as usage examples
and user guide. The notebooks can be read online at:

- http://nbviewer.ipython.org/github/tritemio/PyBroMo/tree/master/notebooks/

You may be also interested in a few notebooks on the theory of Brownian motion
simulation (they don't require PyBroMo):

* [Theory - Introduction to Brownian Motion simulation](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/Theory%2520-%2520Introduction%2520to%2520Brownian%2520Motion%2520simulation.ipynb)
* [Theory - On Browniam motion and Diffusion coefficient](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/Theory%2520-%2520On%2520Browniam%2520motion%2520and%2520Diffusion%2520coefficient.ipynb)

# Dependencies

- python 3.3 (or higher)
- numpy 1.8 (or higher)
- matplotlib 1.4.3 (or higher) (with QT backend for the trajectory explorer GUI)
- pytables 3.1 (or later)
- Jupyter notebook 3 (or later)
- [phconvert](http://photon-hdf5.github.io/phconvert/) (0.6.6 or later) to save smFRET Photon-HDF5 files

# Acknowledgements

I wish to thank Xavier Michalet for useful discussions.

This work was supported by NIH grants R01 GM069709 and R01 GM095904.

# License

PyBroMo - A single molecule FRET diffusion simulator in confocal geometry.

Copyright (C) 2013-2015  Antonino Ingargiola - <tritemio@gmail.com>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    version 2, as published by the Free Software Foundation.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You can find a full copy of the license in the file LICENSE.txt
