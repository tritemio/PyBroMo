import os

if os.name == 'posix':
    #BROWN_DIR = "/home/anto/Documents/ucla/src/brownian/"
    SIM_DATA_DIR = "/home/anto/Documents/ucla/src/data/sim/brownian/objects/"
    SIM_PH_DIR = "/home/anto/Documents/ucla/src/data/sim/brownian/ph_times/"
elif os.name == 'nt':
    #BROWN_DIR = r"C:/Data/Antonio/software/Dropbox/brownian/"
    SIM_DATA_DIR = "C:/Data/Antonio/data/sim/brownian/objects/"
    SIM_PH_DIR = "C:/Data/Antonio/data/sim/brownian/ph_times/"
