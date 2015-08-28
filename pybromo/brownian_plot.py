#
# PyBroMo - A single molecule diffusion simulator in confocal geometry.
#
# Copyright (C) 2013-2015 Antonino Ingargiola tritemio@gmail.com
#

"""
This module defines functions to plot simulation results.

File part of PyBroMo: a single molecule diffusion simulator.
Copyright (C) 2013-2014 Antonino Ingargiola tritemio@gmail.com
"""

import matplotlib.pyplot as plt
#from scroll_gui import ScrollingToolQT


def plot_tracks(S):
    fig, AX = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.05,
            hspace=0.05)
    plt.suptitle("%.1f ms diffusion" % (S.t_step*S.n_samples*1e3))

    for ip in range(S.np):
        x, y, z = S.pos[ip]
        x0, y0, z0 = S.particles[ip].r0
        plot_kwargs = dict(ls='', marker='o', mew=0, ms=1, alpha=0.2)
        l, = AX[1].plot(x*1e6, y*1e6, **plot_kwargs)
        AX[1].plot([x0*1e6], [y0*1e6], 'o', color=l.get_color())
        AX[0].plot(x*1e6, z*1e6, color=l.get_color(), **plot_kwargs)
        AX[0].plot([x0*1e6], [z0*1e6], 'o', color=l.get_color())

    AX[1].set_ylabel("y (um)")
    AX[1].set_xlabel("x (um)")
    AX[0].set_ylabel("z (um)")
    AX[0].grid(True); AX[1].grid(True)

    if S.psf.name == 'gauss':
        sig = S.psf.s
    else:
        sig = array([0.1, 0.1, 0.3])*1e-6
    ## Draw an outline of the PSF
    a = arange(360)/360.*2*np.pi
    rx, ry, rz = (sig)  # draw radius at 3 sigma
    AX[1].plot((rx*np.cos(a))*1e6, (ry*np.sin(a))*1e6, lw=2, color='k')
    AX[0].plot((rx*np.cos(a))*1e6, (rz*np.sin(a))*1e6, lw=2, color='k')

def plot_emission(S, dec=1, scroll_gui=False, multi=False, ms=False):
    fig = plt.figure()
    plt.title("%d Particles, %.1f s diffusion, %d pM" % (S.np,
            S.t_step*S.n_samples, S.concentration()*1e12))
    plt.xlabel("Time (s)"); plt.ylabel("Emission rate [A.U.]"); plt.grid(1)
    if ms: plt.xlabel("Time (ms)")
    if multi:
        for em in S.em:
            time = S.time(dec=dec) if not ms else S.time(dec=dec)*1e3
            plt.plot(time, em[::dec], alpha=0.5)
    else:
        time = S.time(dec=dec) if not ms else S.time(dec=dec)*1e3
        plt.plot(time, S.em.sum(axis=0)[::dec], alpha=0.5)
    s = None
    if scroll_gui: s = ScrollingToolQT(fig)
    return s

def plot_timetrace(S, rebin=2e3, scroll_gui=False):
    fig = plt.figure()
    plt.title("%d Particles, %.1f s diffusion, %d pM, bin=%.1fms" % (S.np,
            S.t_step*S.n_samples, S.concentration()*1e12, S.t_step*rebin*1e3))
    plt.xlabel("Time (s)"); plt.ylabel("Emission rate [A.U.]"); plt.grid(1)
    trace = S.tt.reshape(-1, rebin).sum(axis=1)
    t_trace = (arange(trace.size)+1)*(S.t_step*rebin)
    plt.plot(S.time(), S.em.sum(axis=0)*100, alpha=0.4)
    plt.plot(t_trace, trace, color='k', alpha=0.7, drawstyle='steps')
    if scroll_gui: return ScrollingToolQT(fig)
