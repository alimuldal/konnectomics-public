# Copyright 2014 Alistair Muldal <alistair.muldal@pharm.ox.ac.uk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import warnings
from scipy.stats import poisson, binom

from cymodules import _fast_kernels

def fast_ca_from_spikes(S, dt=(1./50), A=50E-6, tau=1.):

    S = S.astype(np.float64)

    if S.ndim == 1:
        C = np.array(_fast_kernels.ca_from_spikes(S, dt, A, tau))

    elif S.ndim == 2:
        C = np.array(_fast_kernels.ca_from_spikes_parallel(S, dt, A, tau))

    else:
        raise ValueError('S must be either (nt,) or (nc,nt)')

    return C

def fluor_from_ca(C, kappa=300E-6):
    return C / (C + kappa)

def noisy_from_no_noise(F, sigma=0.0535, inplace=False):
    noise = np.random.normal(loc=0., scale=sigma, size=F.shape)
    noise = noise.astype(F.dtype)
    if inplace:
        F += noise
        return F
    else:
        return F + noise

def ca_from_spikes(S, dt=(1./50), A=50E-6, tau=1., dtype=np.float64):

    nc, nt = S.shape
    C = np.empty((nc, nt), dtype=dtype)
    C[..., 0] = A * S[..., 0]

    for tt in xrange(1, nt):
        dCa = -(dt / tau) * C[..., tt - 1] + (A * S[..., tt])
        C[..., tt] = C[..., tt-1] + dCa

    return C

def fake_fluor(ncells, nframes, dt=(1. / 50), rate=1., A=50E-6, tau=1.,
               kappa=300E-6, sigma=0.03):
    """
    Generate fake fluorescence traces based on the model described in Stetter
    et al., 2012

    Arguments:
    -----------
        ncells:     number of traces to generate
        nframes:    number of timebins to simulate
        dt:         timestep (s)
        rate:       spike rate (Hz)
        A:          amplitude of calcium influx for one spike (M)
        tau:        time constant of decay in calcium concentration (s)
        kappa:      saturating calcium concentration (M)
        sigma:      SD of additive noise on fluorescence (A.U.)

    Returns:
    -----------
        S:          spike counts
        C:          calcium concentration
        F:          simulated fluorescence

    Each of the outputs are (ncells, nframes) arrays
    """

    # poisson spikes
    S = poisson.rvs(rate * dt, size=(ncells, nframes)).squeeze()

    # # binomial spikes
    # S = binom.rvs(1, rate * dt, size=(ncells, nframes)).astype(np.bool)

    # internal calcium dynamics
    # C = ca_from_spikes(S, dt, A, tau, np.float32)
    C = fast_ca_from_spikes(S, dt, A, tau)

    # saturation & noise
    if kappa is not None:
        F = fluor_from_ca(C, kappa)
    else:
        F = C.copy()
    # F += np.random.normal(loc=0., scale=sigma, size=C.shape).astype(np.float32)
    F = noisy_from_no_noise(F, sigma, inplace=True)

    return S, C, F
