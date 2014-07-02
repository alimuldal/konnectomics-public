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
from scipy import optimize
from matplotlib import pyplot as plt
from matplotlib import colorbar

from utils import submission
reload(submission)


def all_distances(network_pos):
    network_pos = network_pos[:]
    x, y = network_pos.T
    dx = (x[:, None] - x[None, :])
    dy = (y[:, None] - y[None, :])
    return np.sqrt((dx * dx) + (dy * dy))


def ecdf(x, xnorm=False):
    xi = np.array(x, dtype=np.float32, copy=True)
    xi.sort(0)
    if xnorm:
        xi /= xi.max(0)
    y = np.linspace(1. / xi.shape[0], 1, xi.shape[0])
    return xi, y


def deconvolve(fluor, pos, prctile=10, A0=0.15, lamb0=0.15, do_plot=True):

    nc, nt = fluor.shape

    # euclidean distances
    dist = all_distances(pos)
    ij, distvec = submission.adjacency2vec(dist)

    # Pearson correlation coefficients for small fluorescence values
    corr = threshold_corr(fluor, prctile)
    ij, corrvec = submission.adjacency2vec(corr)

    # from Stetter et al 2012
    # A = 0.15
    # lamb = 0.15
    A, lamb = fit_gauss_blur(distvec, corrvec, A0, lamb0)

    # convolution matrix (nc x nc)
    C = gauss((A / 2., lamb), dist)   # why divide by 2?

    # # we set the diagonal to zero, since we don't consider a cell's own
    # # fluorescence
    # C[np.diag_indices(nc)] = 0

    # F + CF    = F_sc
    # (I + C)F  = F_sc
    deconv = np.linalg.solve((np.eye(nc) + C), fluor)

    if do_plot:

        corr2 = threshold_corr(deconv, prctile)
        ij, corrvec2 = submission.adjacency2vec(corr2)
        A2, lamb2 = fit_gauss_blur(distvec, corrvec2, A0, lamb0)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True,
                                       figsize=(8, 8))

        plot_hist_fit(distvec, corrvec, (A, lamb), ax=ax1)
        plot_hist_fit(distvec, corrvec2, (A2, lamb2), ax=ax2)

        ax1.set_title('Original', fontsize=18)
        ax2.set_title(      'Deconvolved', fontsize=18)

        ax2.set_xlabel('Distance (mm)', fontsize=14)
        ax1.set_ylabel('Correlation coefficient', fontsize=14)
        ax2.set_ylabel('Correlation coefficient', fontsize=14)

        cax, kw = colorbar.make_axes((ax1, ax2))
        ax2.images[0].set_clim(ax1.images[0].get_clim())
        cb = plt.colorbar(ax1.images[0], cax=cax, **kw)
        cb.set_label('Density')

        plt.show()

    return deconv


def threshold_corr(fluor, prctile):
    meanf = fluor.mean(0)
    thresh = np.percentile(meanf, prctile)
    baseline_fluor = fluor[:, meanf < thresh]
    return np.corrcoef(baseline_fluor)


def gauss(P, x):
    A, lamb = P
    return A * np.exp(- (x / lamb) ** 2)


def gauss_loss(P, x, y):
    diff = y - gauss(P, x)
    return np.sum(diff * diff)


def fit_gauss_blur(distvec, corrvec, A0, lamb0):

    P0 = np.hstack((A0, lamb0))
    res = optimize.minimize(gauss_loss, P0, args=(distvec, corrvec),
                            method='Nelder-Mead')

    return res.x


def plot_hist_fit(distvec, corrvec, P=None, ax=None):

    if ax is None:
        ax = plt.gca()

    # xbins = np.linspace(0, np.sqrt(2), 1000)
    # ybins = np.linspace(-1., 1., 500)

    xbins = np.linspace(0, 0.20, 250)
    ybins = np.linspace(-0.2, 0.5, 250)

    h, xe, ye = np.histogram2d(distvec, corrvec, bins=(xbins, ybins),
                               normed=True)

    ax.hold(True)

    im = ax.imshow(h.T, cmap=plt.cm.jet, aspect='auto',
                   origin='bottom', interpolation='bicubic',
                   extent=(xe[0], xe[-1], ye[0], ye[-1]))

    # cb = plt.colorbar(im, use_gridspec=True)
    # cb.set_label('Density')

    ax.axhline(0, ls='--', color='w', alpha=0.5)

    if P is not None:
        gfit = gauss(P, xbins)
        ax.plot(xbins, gfit, '-w', lw=2, scalex=False, scaley=False,
                label='$\^A$ = %.3g\n$\^\lambda$ = %.3g' % tuple(P))

        leg = ax.legend(loc=0, frameon=False, fontsize=18)
        for text in leg.get_texts():
            text.set_color('w')

    # ax.set_xlim(0, 0.15)
    # ax.set_ylim(-0.1, 0.4)

    return ax
