Overview
=======

PyBroMo is a simulator for confocal single-molecule fluorescence experiments.

The program simulates 3-D Brownian motion trajectories and fluorescent
emission of an arbitrary number of molecules diffusing in a simulation volume. 
The excitation volume is defined numerically or analytically (3-D Gaussian 
shape). Molecules diffusing through the excitation volume will emit at a rate
proportional to the excitation intensity.

PyBroMo allows to simulate smFRET experiments with a desired FRET efficiency.
Timestamps for donor and acceptor channel can be generated.

The PSF is numerically precomputed on a regular grid using the 
[PSFLab](http://onemolecule.chem.uwm.edu/software) software and interpolated 
during the Brownian motion simulation in order to compute the emission 
intensity. Alternatively a simple analytical Gaussian PSF can be also used.

The user documentation is in the form of a series of IPython Notebooks.

For more information contact me at tritemio@gmail.com.

Environment
==========

PyBroMo is written in the python programming language using the standard 
scientific stack of libraries (numpy, scipy, matplotlib).

Usage examples are given as 
[IPython Notebook](http://ipython.org/notebook.html) files. This is an 
interactive environment that allows to mix rich text, math and graphics with 
(executable) code, similarly to the Mathematica environment. An static version
of the notebooks can be read following the link provided at the end of this
 page.

Moreover the IPython environment allows to easily build and use a cluster 
for parallel computing. This feature allows to leverage the computing power
of multiple computers (different desktop in a lab) greatly enhancing
the length and span of simulation that can be performed. Examples on how to
perform parallel simulation are provided as well.

For more information for python and science:

* [Python Scientific Lecture Notes](http://scipy-lectures.github.io/)


#Installation


##MS Windows

In order to use the software you need to install a scientific python
distribution like [Anaconda](https://store.continuum.io/cshop/anaconda/).
The free version of Anaconda includes all the needed dependencies.
 
After that, you can start using the simulator
launching an IPython Notebook server in the PyBroMo notebook folder
(see the following paragraph) and executing the examples.

###Configuring IPython Notebook

If you put PyBroMo in "C:\PyBroMo" then the notebooks folder will be 
"C:\PyBroMo\notebooks".

Just right click on the IPython Notebook icon -> Properties and paste 
this folder in "Start in". Apply and close.

Now double click on the icon and a browser should pop up showing the list
of notebooks. Chrome browser is suggested.

##Linux and Mac OS X

On Linux or Mac OS X you can use the Anaconda distribution.

Otherwise this are the requirements you have to make sure are properly 
installed:

 - python 2.7.x
 - IPython 1.x
 - matplotlib 1.3.x or greater
 - numpy/scipy (any recent version)
 - modern browser (Chrome suggested)
 
#Usage examples

The following links will open (a static version of) the notebooks provided
with PyBroMo. These collection serves both as usage examples and user guide:

* [1.1 Run simulation - Single host](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/PyBroMo%2520-%25201.1%2520Run%2520simulation%2520-%2520Single%2520host.ipynb)
* [1.2 Run simulation - Parallel](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/PyBroMo%2520-%25201.2%2520Run%2520simulation%2520-%2520Parallel.ipynb)
* [2. Generate timestamps - Parallel](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/PyBroMo%2520-%25202.%2520Generate%2520timestamps%2520-%2520Parallel.ipynb)
* [3. Burst analisys](http://nbviewer.ipython.org/urls/raw.github.com/tritemio/PyBroMo/master/notebooks/PyBroMo%2520-%25203.%2520Burst%2520analisys.ipynb)


#Licence and Copyrights

PyBroMo - A single molecule diffusion simulator in confocal geometry.

Copyright (C) 2013  Antonino Ingargiola <tritemio@gmail.com>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You can find a full copy of the license in the file LICENSE.txt

