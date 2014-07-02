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


def apply_gaussian_blur(network_pos, fluor, A=0.15, lamb=0.0315):
    """
    Simulate optical blurring of fluorescence signal as a Gaussian function of
    distance (as described in Stetter et al., 2012)

    Arguments:
    ------------

        network_pos: (2, ncells) float array
            the x, y positions of each cell (nominally in mm)

        fluor: (ncells, ntimebins) float array
            the fluorescence traces for each cell

        A: float, optional*
            the amplitude of the Gaussian function

        lamb: float, optional*
            the length constant of the Gaussian function


    Returns:
    ------------

        blurred: (ncells, ntimebins)
            the blurred fluorescence traces

    * The default values of A and lamb were obtained by fitting the normal1
      competition dataset, using theano_unblur.fit_blur()
    """

    # handle HDF5 nodes
    network_pos = network_pos[:]
    fluor = fluor[:]


    blurmat = get_blurring_matrix(network_pos, A, lamb)
    crosstalk = np.dot((np.eye(blurmat.shape[0]) + blurmat), fluor)
    blurred_fluor = fluor + crosstalk

    return blurred_fluor

def fake_positions(ncells, x_lim=(0, 1), y_lim=(0, 1)):
    """
    Generate fake x, y coordinates for each cell, drawn from a uniform
    distribution bounded on x_lim and y_lim
    """

    x = np.random.uniform(low=x_lim[0], high=x_lim[1], size=ncells)
    y = np.random.uniform(low=y_lim[0], high=y_lim[1], size=ncells)

    return np.vstack((x, y)).T

def gauss(A, lamb, d):
    # we set the diagonal terms to zero
    return A * (np.exp(- (d / lamb) ** 2) - np.eye(d.shape[0]))

def all_distances(pos):
    x, y = pos.T
    dx = (x[:, None] - x[None, :])
    dy = (y[:, None] - y[None, :])
    dist = np.sqrt((dx * dx) + (dy * dy))
    return dist

def get_blurring_matrix(pos, A, lamb):

    dist =  all_distances(pos)

    # the amplitude still isn't quite right...
    blurmat = gauss(A, lamb, dist)

    return blurmat
