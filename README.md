PyBroMo is a simulator for confocal single-molecule fluorescence experiments.

The simulator allow to generate 3-D brownian motion trajectories and fluorescent
emission of an arbitrary number of molecules diffusing in a simulation volume. 
The excitation volume is defined numerically or analitically (3-D gaussian 
shape). Molecules diffusing through the excitation volume will emit at a rate
proportional to the excitation intensity.

PyBroMo allows to simulate smFRET experiments for and a desired FRET efficiency.
Timestamps for donor and acceptor channel are generated.

The PSF is numerically precomputed on a recular grid using the 
[PSFLab](http://onemolecule.chem.uwm.edu/software) software and interpolated 
during the brownian motion simulation in order to compute the emission 
intensity.

Several IPython Notebooks are provided as usage example and user guide.


