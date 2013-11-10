"""
This module provides classes to compute PSF functions either starting from
an analytical formula (i.e. Gaussian) or by interpolation of a precomputed PSF.
"""

from scipy.io import loadmat
import scipy.interpolate as SI
from scipy.ndimage import map_coordinates
import numexpr as NE
import numpy as np


class GaussianPSF:
    """This class implements a Gaussian-shaped PSF function."""

    def __init__(self, xc=0, yc=0, zc=0, sx=1, sy=1, sz=1):
        """Create a Gaussian PSF object with given center and sigmas.
        `xc`, `yc`, `zc`: position of hte center of the gaussian
        `sx`, `sy`, `sz`: sigmas of the gaussian function.
        """
        self.xc, self.yc, self.zc = xc, yc, zc
        self.rc = array([xc, yc, zc])
        self.sx, self.sy, self.sz = sx, sy, sz
        self.s = array([sx, sy, sz])
        self.name = "gauss"

    def eval(self, x, y, z):
        """Evaluate the function in (x, y, z)."""
        xc, yc, zc = self.rc
        sx, sy, sz = self.s

        ## Method1: direct evaluation
        #return exp(-(((x-xc)**2)/(2*sx**2) + ((y-yc)**2)/(2*sy**2) +\
        #        ((z-zc)**2)/(2*sz**2)))

        ## Method2: evaluation using numexpr
        arg = lambda s: "((%s-%sc)**2)/(2*s%s**2)" % (s, s, s)
        return NE.evaluate( "exp(-(%s + %s + %s))" %\
                (arg("x"), arg("y"), arg("z")) )

        ## Method3: evaluation with partial function
        #g_arg = lambda t, mu, sig: -((t-mu)**2)/(2*sig**2)
        #return exp(g_arg(x, xc, sx) + g_arg(y, yc, sy) + g_arg(z, zc, sz))


class NumericPSF:

    def __init__(self, fname='xz_realistic_z50_150_160_580nm_n1335_HR2'):
        self.fname = fname
        subdir = 'psf_data/'
        xi, zi, hdata, zm = load_PSFLab_xz(subdir+self.fname,
                x_step=0.5/8, z_step=0.5/8)
        hdata /= hdata.max() # normalize to 1 at peak
        self._fun_um = SI.RectBivariateSpline(xi, zi, hdata.T, kx=1, ky=1)
        self.xi, self.zi, self.hdata, self.zm = xi, zi, hdata, zm
        self.x_step, self.z_step = xi[1] - xi[0], zi[1] - zi[0]
        self.name = 'numeric'

    def eval_xz(self,x,z):
        """Evaluate the function in (x, z).
        The function is rotationally symmetric around z.
        """
        return self._fun_um.ev(x*1e6, z*1e6)

    def eval(self, x, y, z):
        """Evaluate the function in (x, y, z).
        The function is rotationally symmetric around z.
        """
        ro = np.sqrt(x**2 + y**2)
        zs, xs = ro.shape
        v = self.eval_xz(ro.ravel(), z.ravel())
        return v.reshape(zs, xs)


def load_PSFLab_xz(fname, x_step=0.5, z_step=0.5, normalize=0):
    """Load a PSFLab .mat file containing a x-z slice of a PSF."""
    data = loadmat(fname)['data']
    z_len, x_len = data.shape
    hdata = data[:,(x_len-1)/2:]
    x = np.arange(hdata.shape[1])*x_step
    z = np.arange(-(z_len-1)/2, (z_len-1)/2+1)*z_step
    if normalize: hdata /= hdata.max()
    return x, z, hdata, hdata[:,0].argmax()
