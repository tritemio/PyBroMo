"""
This module defines some default paths for saving and loading simulations.

For each folder you can define two values: one for Linux/Mac OS X ('posix')
and one for Windows ('nt').

SIM_DATA_DIR: is the folder where the brownian motion simulations are saved
            (they contain the emission trace). When runnning simulations on a
            cluster every node (ipengine) must use this folder to save data.

SIM_PH_DATA: is the folder where to save the timestamps arrays generated from
            the (saved) emission traces. Even when performing a simulation
            on a cluster the timestamps are collected and saved on the
            controlling host (ipcontroller). The remote nodes don't need
            to save in this folder.

GIT_PATH: path to the git executable (system dependent)

NB: In path definition preferably use '/' to separate the folders. If you 
    use '\' prepend the string with r (raw string). For example these
    paths are equivalent:

    'C:/Data/Simulations/'

    r'C:\Data\Simulations\'
"""

import os

if os.name == 'posix':
    SIM_DATA_DIR = "/home/anto/Documents/ucla/src/data/sim/brownian/objects/"
    SIM_PH_DIR = "/home/anto/Documents/ucla/src/data/sim/brownian/ph_times/"
    GIT_PATH = 'git'   # On *nix assumes that git is in the PATH
elif os.name == 'nt':
    SIM_DATA_DIR = "C:/Data/Antonio/data/sim/brownian/objects/"
    SIM_PH_DIR = "C:/Data/Antonio/data/sim/brownian/ph_times/"
    # On WIndows try to use SourceTree embedded git
    GIT_PATH = (os.environ['homepath'] + \
                r'\AppData\Local\Atlassian\SourceTree\git_local\bin\git.exe')
else:
    raise OSError ("Operating system not recognized (%s)." % os.name)
