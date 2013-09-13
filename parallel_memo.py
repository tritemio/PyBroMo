# -*- coding: utf-8 -*-

def gen_ph_times(dview, max_em_rate, bg_rate, dump=False, Id=0):
    dview.execute("S.sim_timetrace(max_em_rate=%f, bg_rate=%f)" % \
            (max_em_rate, bg_rate))
    dview.execute("S.gen_ph_times()")
    dview.execute("ph_times = S.ph_times")
    PH = dview['ph_times']
    ph_times = merge_ph_times(PH, time_block=t_max)
    if dump:
        EM = ("%.g" % max_em_rate).replace('+0','')
        BG = ("%.g" % bg_rate).replace('+0','')
        ph_times.dump("ph_times_EM"+EM+"_BG"+BG+"_80s_05us_40p_86pM_%d.dat"%Id)
    return ph_times


from IPython.parallel import Client
rc = Client() # get the running engines
dview = rc[:] # use all of them

# Push a variable with the engine ID to each engine
dview.scatter('eid', rc.ids, flatten=True)

# Change dir and load common libs
dview.execute(r"%cd C:\Data\Antonio\software\brownian")
dview.execute("from pylab import *")

#with dview.sync_imports():
#    import numpy as NP # imports numpy on the engines

# Setup the simulation
dview.execute("run -i brownian2.py")

# Run the simulation on all the engines
dview.block = False
ar = dview.execute("S.sim_motion_em(N_samples)")

# Save the emission array
dview.execute("S.dump_emission(ID=eid)")

# Load the emission
%px run -i brownian2 # NB use same setting as the saved file
dview.execute('em = NP.load("trace_em_40P_10s_0.5us_Du12_ID%d-0.npy" % eid)')
dview.execute('S.em = em') # now S methods for ph_times can be called

## 1. Merge the emission in a local variable Sm
run -i brownian2 # remeber to run it also locally!!
SS = dview['S'] # Return a list of variables S, one for each engine
Sm = merge_simulations(SS)

## 2. Pull all the ph_times and stack them to obtain a longer stream
ph_times = gen_ph_times(dview, 8e5, 8e3, dump=1)
#hist(ph_times, bins=arange(0,80,0.001), histtype='step')
