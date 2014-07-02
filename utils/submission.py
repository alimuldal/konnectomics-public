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
import datetime
import subprocess

from matplotlib import pyplot as plt
from itertools import izip
from sklearn import metrics


def make_submission(valid_weights, test_weights, fname=None, norm=True,
                    compress=True):

    if fname is None:
        fname = 'data/submission_%s.csv' % datetime.date.today()

    if norm:
        valid_weights = valid_weights.astype(np.float64)
        valid_weights = (valid_weights - valid_weights.min()
                            ) / valid_weights.ptp()

        test_weights = test_weights.astype(np.float64)
        test_weights = (test_weights - test_weights.min()) / test_weights.ptp()

    with open(fname, 'w') as f:

        f.write('NET_neuronI_neuronJ,Strength\n')

        for ii in xrange(valid_weights.shape[0]):
            for jj in xrange(valid_weights.shape[1]):
                f.write('valid_%i_%i,%.8f\n' %(
                    ii + 1, jj + 1, valid_weights[ii, jj]))

        for ii in xrange(test_weights.shape[0]):
            for jj in xrange(test_weights.shape[1]):
                f.write('test_%i_%i,%.8f\n' %(
                    ii + 1, jj + 1, test_weights[ii, jj]))

        print 'Connection weights written to "%s"' % fname

    if compress:

        state = subprocess.call(['gzip', '-9', fname])
        if state == 0:
            print 'Compressed to "%s"' % (fname + '.gz')
        else:
            print 'Compression failed, code=%i' % state

    pass

def run_auc(M, real_network, nsteps=1000, do_plot=False):

    n = M.shape[0]

    # convert the real network to dense vector format
    ij, ground_truth = real2dense(real_network, n)

    # convert the adjacency matrix to a vector of weights for each possible
    # connection
    ij, weights = adjacency2vec(M)

    # compute the ROC curve, and the AUC
    thresh, fpr, tpr, pl10, auc = roc(weights, ground_truth, do_plot=do_plot,
                                      nsteps=nsteps)

    return fpr, tpr, auc


def adjacency2vec(M):
    """
    Unpack an n-by-n directed adjacency matrix to a 1D vector of connection
    weights

    Arguments
    ----------
        M: 2D float array
            adjacency matrix, where:    M[i, j] corresponds to w(i->j)

    Returns
    ----------
        ij: 2D int array
            2-by-npairs array of row/column indices

        w_ij: 1D int array
            corresponding weights, i.e. w(i->j)

    """

    ncells = M.shape[0]
    ij = all_directed_connections(ncells)

    # sanity check
    # npairs = ncells * (ncells - 1)
    # assert ij.shape[1] == npairs

    i, j = ij
    w_ij = M[i, j]

    return ij, w_ij

def vec2adjacency(ij, connected):
    """
    Pack a 1D vector of connection weights into an n-by-n directed adjacency
    matrix

    Arguments
    ----------

        ij: 2D int array
            2-by-npairs array of row/column indices

        w_ij: 1D int array
            corresponding weights, i.e. w(i->j)

    Returns
    ----------

        M: 2D float array
            adjacency matrix, where:    M[i, j] corresponds to w(i->j)
                                        M[j, i] corresponds to w(j->i)

    """

    npairs = connected.size

    # 0 = ncells**2 - ncells -npairs
    roots = np.roots((1, -1, -npairs))
    ncells = int(roots[roots > 0])

    M = np.zeros((ncells, ncells), dtype=connected.dtype)

    for (ii, jj), cc in izip(ij.T, connected):
        M[ii, jj] = cc

    return M


def real2dense(real_connections, n=None, adj=False):
    """
    The network data provided for the challenge lists connections weighted '-1'
    (which aren't actually present in the simulation), and does not list any
    weights for pairs of nodes that are not connected.

    This function converts the provided data into a more convenient dense vector
    format compatible with adjacency2vec and roc, where every possible directed
    pair of nodes has a True/False weight.

    Arguments:
    -----------

        real_connections: 2D np.ndarray or tables.(C)Array
            npairs-by-3 array, whose columns represent (i, j, connected(i->j)).
            i, j are assumed to follow MATLAB indexing convenions (i.e. they
            start at 1).

        n: positive int, optional
            the total number of nodes (cells). if unspecified, this is taken to
            be the maximum index in the first two columns of real_connections
            plus 1.

    Returns:
    ----------

        ij: 2D int array
            2-by-npairs array of row/column indices

        connected:
            boolean vector, True where i->j is connected

    """

    if n is None:
        n = int(real_connections[:, :2].max())

    if np.any(real_connections[:, :2] > n):
        raise ValueError('real_connections contains indices > n')

    # handle CArrays
    real_connections = real_connections[:]

    # cast to integers
    real_connections = real_connections.astype(np.int)

    # find the indices of the cells that are genuinely connected ('1' means
    # connection, either '-1', '0' or omission means no connection).
    ij_con = real_connections[(real_connections[:, 2] == 1), :2].T

    # we subtract 1 from the indices because MATLAB-style indexing starts at 1,
    # whereas Python indexing starts at 0
    ij_con -= 1

    # we'll do this the lazy way - construct an adjacency matrix from the
    # connected indices ...
    M = np.zeros((n, n), dtype=np.bool)
    M[ij_con[0, :], ij_con[1, :]] = True

    if adj:
        return M

    else:
        # ... then convert this directly to the desired format
        ij, connected = adjacency2vec(M)

        return ij, connected


def all_directed_connections(n):
    """
    For an n-by-n adjacency matrix, return the indices of the nodes for every
    possible directed connection, i.e. (i->j) and (j->i), but not (i->i) or
    (j->j)

    Arguments:

        n: int
            number of nodes

    Returns:

        idx: 2D int array
            [2, n * (n - 1)] array of i, j indices
    """

    # all possible pairs of indices (including repeated indices)
    all_idx = np.indices((n, n)).T.reshape(-1, 2).T

    # remove repeated indices
    repeats = (all_idx[0, :] == all_idx[1, :])
    idx = all_idx[:, ~repeats]

    return idx


def roc(weights, ground_truth, nsteps=None, do_plot=False, show_progress=True):
    """
    Compute ROC curve and performance metrics for a given set of posterior
    connection probabilities and the set of ground-truth connections

    Arguments:
    ----------

        weights: 1D float array
            vector of posterior probabilities for each possible pairwise
            connection

        ground_truth: 1D bool array
            vector of ground-truth connections

        nsteps: int, optional
            number of linear steps between the minimum and maximum values of
            weights at which to compute the FPR and TPR. if unspecified, every
            unique value of weights is used, so that the ROC curve is computed
            exactly

        do_plot: bool, optional
            make a pretty plot

        show_progress: bool, optional
            show a pretty progress bar

    Returns:
    ---------

        thresh: 1D float array
            vector of threshold values used for computing the ROC curve

        fpr: 1D float array
            false-positive rate at each threshold value

        tpr: 1D float array
            true-positive rate at each threshold value

        pl10: float
            10% performance level (tpr at the threshold value that gives 10%
            false-positives)

        auc: float
            area under the ROC curve

    """

    # make sure we're dealing with 1D arrays
    weights = weights.ravel()
    ground_truth = ground_truth.ravel()

    if weights.size != ground_truth.size:
        raise ValueError('Input vectors must have the same number of elements')

    fpr, tpr, thresh = metrics.roc_curve(ground_truth, weights, pos_label=True)
    auc = metrics.roc_auc_score(ground_truth, weights)

    # 'performance level' is defined as the fraction of true positives at 10%
    # false-positives
    pl10 = tpr[fpr.searchsorted(0.1, side='left')]

    if do_plot:

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.hold(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.plot(fpr, tpr, '-b', lw=2)
        ax.set_xlabel('False-positive rate')
        ax.set_ylabel('True-positive rate')
        ax.set_title('ROC')
        bbox_props = dict(boxstyle='round', fc='w', ec='0.5')
        arrow_props = dict(arrowstyle='->', color='k', linewidth=2)
        ax.annotate('AUC = %.4g' % auc, xy=(0.9, 0.1),
                    xycoords='axes fraction', ha='right', va='bottom',
                    bbox=bbox_props, fontsize=16)
        plt.show()

    return thresh, fpr, tpr, pl10, auc
