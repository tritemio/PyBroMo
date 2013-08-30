import os
import scipy.stats as SS
import numpy.random as NR
import numpy as np
from numpy import array, arange, sqrt

from PSF import GaussianPSF, NumericPSF
from scroll_gui import ScrollingToolQT

## Avogadro constant
NA = 6.022141e23    # [mol^-1]

class Box:
    """The simulation box"""
    def __init__(self, x1, x2, y1, y2, z1, z2):
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.z1, self.z2 = z1, z2
        self.b = array([[x1,x2],[y1,y2],[z1,z2]])
    def volume(self):
        """Box volume in m^3."""
        return (self.x2-self.x1)*(self.y2-self.y1)*(self.z2-self.z1)
    def volume_L(self):
        """Box volume in liters."""
        return self.volume()*1e3
class Particle:
    """Class to describe a single particle"""
    def __init__(self, x0=0, y0=0, z0=0):
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.r0 = array([x0, y0, z0])

def wrap_periodic(a, a1, a2):
    """Folds all the values of `a` outside [a1..a2] inside that intervall.
    This function is used to apply periodic boundary contitions.
    """
    a -= a1
    wrapped = mod(a, a2-a1) + a1
    return wrapped

class Particles_in_a_box(object):
    """Class that performs the Brownian motion simulation of N particles.
    """
    def __init__(self, D, t_step, particles, box, psf):
        """Initialize the simulation parameters:
        `D`: diffusion coefficient (m/s^2)
        `t_step`: time step (s)
        `particles`: list of `Particle` objects
        `box`: a `Box` object defining the simulation boundaries
        `psf`: a "PSF" object (`GaussianPSF` or `NumericPSF`) defining the PSF
        """
        self.particles = particles
        self.box = box
        self.psf = psf
        self.np = len(particles)
        self.D = D
        self.t_step = t_step
        self.sigma = sqrt(2*D*3*t_step)
    def concentration(self):
        """Return the volumetric concentration of the particles in the box.
        """
        return (self.np/NA)/self.box.volume_L()
    def sim_motion_em(self, N_samples, delete_pos=True):
        """Simulate Brownian motion and emission rates in one step.
        This method simulates only one particle a time (to use less RAM).  
        `delete_pos` allows to discard the trajectories and save only the
        total emission rates (i.e. the sum of emissions of all the particles.)
        """
        self.N_samples = N_samples
        self.em = np.zeros((1, N_samples), dtype=float64)
        POS = []
        pid = os.getpid()
        for i,p in enumerate(self.particles):
            print "[%4d] Starting particle %d..." % (pid, i)
            delta_pos = NR.normal(loc=0, scale=self.sigma, size=3*N_samples)
            delta_pos = delta_pos.reshape(3,N_samples)
            pos = np.cumsum(delta_pos, axis=-1, out=delta_pos)
            pos += p.r0.reshape(3,1)
            # Coordinates wrapping using periodic boundary conditions
            for coord in (0,1,2):
                pos[coord,:] = wrap_periodic(pos[coord,:], *self.box.b[coord])
            Ro = sqrt(pos[0,:]**2+pos[1,:]**2) # radial position on x-y plane
            Z = pos[2,:]                       
            self.em += (self.psf.eval_xz(Ro,Z)**2)
            if not delete_pos: POS.append(pos.reshape(1,3,N_samples))
        if not delete_pos: self.pos = concatenate(POS)
    def sim_timetrace(self, max_em_rate=1, bg_rate=0):
        """Draw random emitted photons from Poisson(emission rates)."""
        self.bg_rate = bg_rate
        em_rates = (self.em.sum(axis=0)*max_em_rate + bg_rate)*self.t_step
        self.tt = NR.poisson(lam=em_rates).astype(uint8)
    def gen_ph_times(self):
        iph = arange(self.tt.size, dtype=uint32)
        PH = []
        for v in range(1,self.tt.max()+1):
            PH.append(iph[self.tt >= v])
        # Index of 1st time with V photons per bins, for each V >= 2
        I = cumsum(array([ph.size for ph in PH]))
        #print I
        ph_times = hstack(PH).astype(float64)
        fraction = 0.5
        for iph1,iph2 in zip(I[:-1],I[1:]):
            ph_times[iph1:iph2] += fraction
            fraction /= 2.
            #print iph1, iph2
        ph_times.sort()
        ph_times *= self.t_step
        self.ph_times = ph_times
    def time(self):
        if not hasattr(self, "_time"):
            self._time = arange(self.N_samples)*self.t_step
        return self._time
    def dump_emission(self, prefix="trace_em", EID=0, ID=0):
        s = "%s_%dP_%ds_%.1fus_Du%d_ID%d-%d.npy" % \
            (prefix,S.np, S.N_samples*S.t_step, S.t_step*1e6, S.D*1e12, EID,ID)
        self.em.dump(s)

def merge_particle_emission(SS):
    # Merge all the particles
    P = reduce(lambda x,y: x+y, [Si.particles for Si in SS])
    s = SS[0]
    S = Particles_in_a_box(D=s.D, t_step=s.t_step, particles=P, box=s.box,
                           psf=s.psf)
    S.N_samples = s.N_samples
    S.em = zeros(s.em.shape, dtype=float)
    for Si in SS:
        S.em += Si.em
    return S

def merge_ph_times(PH, time_block):
    DELAY = arange(len(PH))*time_block
    SIZES = cumsum([ph.size for ph in PH])
    ph_times = zeros(SIZES[-1])
    i1 = 0
    for i2,ph,delay in zip(SIZES[:-1],PH[:-1],DELAY[:-1]):
        ph_times[i1:i2] = ph+delay
        i1 = i2
    ph_times[i1:] = PH[-1]+DELAY[-1]
    return ph_times

def forge_ph_times(ph_times_d, ph_times_a):
    """Returns the merged timestamps of D and A and a bool mask for A
    """
    ph_times = np.hstack([ph_times_d, ph_times_a])
    a_em = np.hstack([np.zeros(ph_times_d.size, dtype=np.bool),
            np.ones(ph_times_a.size, dtype=np.bool)])
    index_sort = ph_times.argsort()
    return ph_times[index_sort], a_em[index_sort]

def gen_particles(N, box):
    X0 = NR.rand(N)*(box.x2-box.x1) + box.x1
    Y0 = NR.rand(N)*(box.y2-box.y1) + box.y1
    Z0 = NR.rand(N)*(box.z2-box.z1) + box.z1
    return [Particle(x0=x0,y0=y0,z0=z0) for x0,y0,z0 in zip(X0,Y0,Z0)]

##
# PLOT FUNCTIONS
#
def plot_tracks(S):
    fig, AX = subplots(2,1, figsize=(6,9), sharex=True)
    subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.05, hspace=0.05)
    suptitle("%.1f ms diffusion" % (S.t_step*S.N_samples*1e3))
    
    for ip in range(S.np):
        x, y, z = S.pos[ip]
        x0, y0, z0 = S.particles[ip].r0
        l, = AX[1].plot(x*1e6, y*1e6, '-', ms=0.5, lw=1, alpha=0.3)
        AX[1].plot([x0*1e6], [y0*1e6], 'o', color=l.get_color())
        AX[0].plot(x*1e6, z*1e6, '-', ms=0.5, lw=1, alpha=0.3, 
                color=l.get_color())
        AX[0].plot([x0*1e6], [z0*1e6], 'o', color=l.get_color())

    AX[1].set_ylabel("y (um)")
    AX[1].set_xlabel("x (um)")
    AX[0].set_ylabel("z (um)")
    AX[0].grid(True); AX[1].grid(True)

    if S.psf.name == 'gauss':
        sig = S.psf.s
    else:
        sig = array([0.1,0.1,0.3])*1e-6
    ## Draw an outline of the PSF
    a = arange(360)/360.*2*pi
    rx, ry, rz = (sig) # draw radius at 3 sigma
    AX[1].plot((rx*cos(a))*1e6, (ry*sin(a))*1e6, lw=2, color='k')
    AX[0].plot((rx*cos(a))*1e6, (rz*sin(a))*1e6, lw=2, color='k')

def plot_emission(S, dec=1):
    fig = figure()
    title("%d Particles, %.1f s diffusion, %d pM" % (S.np,
            S.t_step*S.N_samples, S.concentration()*1e12))
    xlabel("Time (s)"); ylabel("Emission rate [A.U.]"); grid(1)
    #plot_timetrace(S, rebin=rebin)
    plot(S.time(), S.em.sum(axis=0)[::dec], alpha=0.5)
    s = ScrollingToolQT(fig)
    return s
def plot_timetrace(S, rebin=2e3):
    fig = figure()
    title("%d Particles, %.1f s diffusion, %d pM, bin=%.1fms" % (S.np,
            S.t_step*S.N_samples, S.concentration()*1e12, S.t_step*rebin*1e3))
    xlabel("Time (s)"); ylabel("Emission rate [A.U.]"); grid(1)
    trace = S.tt.reshape(-1,rebin).sum(axis=1)
    t_trace = (arange(trace.size)+1)*(S.t_step*rebin)
    plot(S.time(), S.em.sum(axis=0)*100, alpha=0.4)
    plot(t_trace, trace, color='k', alpha=0.7, drawstyle='steps')
    s = ScrollingToolQT(fig)
    return s
def fun(S):
    S.sim_brownian_motion(N_samples)
    S.cal_emission(delete_pos=True)
    #S.sim_emission(max_em_rate=1e6, bg_rate=10e3)

if __name__ == '__main__':
    # Simulation time step
    t_step = 0.5e-6   # seconds

    # Diffusion coefficient
    Du = 12.0        # um^2 / s
    D = Du*(1e-6)**2

    # Number of samples to simulate
    t_max = 0.1 # seconds
    N_samples = int(t_max/t_step)

    # PSF definition
    #ss = 0.2*1e-6 # lateral dimension (sigma)
    #psf = GaussianPSF(xc=0,yc=0,zc=0,sx=ss,sy=ss,sz=3*ss)
    psf = NumericPSF()

    # Box definition
    box = Box(x1=-4.e-6,x2=4.e-6,y1=-4.e-6,y2=4.e-6,z1=-6e-6,z2=6e-6)

    # Particles definition
    #p1 = Particle(x0=-3e-6)
    #p2 = Particle(x0=3e-6)
    #p3 = Particle(y0=-3e-6)
    #p4 = Particle(y0=3e-6)
    #P = [p1,p2,p3,p4]
    P = gen_particles(1, box)

    # Particle simulation
    S = Particles_in_a_box(D=D, t_step=t_step, particles=P, box=box, psf=psf)
    #S.sim_motion_em(N_samples)
    #S.sim_timetrace(max_em_rate=3e5, bg_rate=10e3)
    #S.gen_ph_times()
    
    #S.sim_brownian_motion(N_samples)
    #S.cal_emission(delete_pos=True)
    #S.sim_timetrace(max_em_rate=1e6, bg_rate=10e3)

    #plot_tracks(S)
    #plot_emission(S)
    #plot(S.time()*1e3, S.tt, 'k')
    #hist(S._ph*1e3, bins=r_[:S.t_step*S.N_samples*1e3:0.1], histtype="step", normed=1, color='k')

