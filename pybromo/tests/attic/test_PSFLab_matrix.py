# -*- coding: utf-8 -*-
from scipy.io import loadmat
import numpy as NP

def load_PSFLab_xz(fname, x_step=0.5, z_step=0.5, normalize=0):
    data = loadmat(fname)['data']
    z_len, x_len = data.shape
    hdata = data[:,(x_len-1)/2:]
    x = NP.arange(hdata.shape[1])*x_step
    z = NP.arange(-(z_len-1)/2,(z_len-1)/2+1)*z_step
    if normalize: hdata /= hdata.max()
    return x,z,hdata, hdata[:,0].argmax()


x1,z1,hdata1,zm1 = load_PSFLab_xz(
        "psf_data/xz_realistic_z50_150_160_580nm_n1335_HR2",
        x_step=0.5/8, z_step=0.5/8, normalize=1)

#x1,z1,hdata1,zm1 = load_PSFLab_xz("psf_data/xz_532_HR2",
#        x_step=0.5/8, z_step=0.5/8, normalize=1)

semilogy(z1, hdata1**2); grid(1)
xlabel('z (um)')

figure()
semilogy(x1, hdata1.T**2); grid(1)
xlabel('x (um)')

