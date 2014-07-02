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
import tables
from scipy import signal

from optical import deconvolution
from spikedetect import fnndeconv, test_spikedetect
from causality import transfer_entropy
from utils import submission
from utils.waitbar import ElapsedTimer


def run(f, g, cond_level=20, dt=0.02, compute_roc=True, do_roc_plot=True):
    """
    A skeleton script that takes 'raw' simulated data to the point of
    inferring connection probabilities and compares these to the ground-
    truth connections

    Arguments
    ---------

        f: tables.file.File
            handle to the PyTables HDF5 file containing the data

        g: tables.group.Group
            node consituting a dataset, should have child nodes containing
            'fluorescence', 'network_pos' and 'network' CArrays

        cond_level: int in the range [1, 100]
            the conditioning level used to exclude bursty time bins. only
            frames where the mean fluorescence across cells is less than
            cond_level % of the maximum are used when computing the GTE.

        dt: float
            the timestep between frames (in seconds)

        compute_roc, do_roc_plot: bool
            whether to do the AUROC analysis

    Returns:
    --------
        weight_mat: [ncells, ncells] np.ndarray
            the matrix of inferred connection weights. weight_mat[i, j]
            corresponds to the weight of the connection from neuron i -> j.

    (if compute_roc)

        fpr: [1000,] np.ndarray
            vector of false-positive rates for 1000 threshold values

        tpr: [1000,] np.ndarray
            vector of true-positive rates for 1000 threshold values

        auc: float
            area under the ROC curve

    """

    # read the data from the HDF5 file
    # ------------------------------------------------------------------------
    timer = ElapsedTimer("* Reading data from HDF5 file")

    fluor = g.fluorescence[:]
    xypos = g.network_pos[:]

    timer.done()

    # remove optical scattering artifact
    # ------------------------------------------------------------------------
    timer = ElapsedTimer("* Removing optical scattering artifact")

    # flatten the relationship between distance and Pearson correlation
    fluor = deconvolution.deconvolve(fluor, xypos, prctile=10, do_plot=False)

    timer.done()

    # segment traces into bursty and nonbursty time bins
    # ------------------------------------------------------------------------
    timer = ElapsedTimer("* Segmenting by mean activity")

    meanf = fluor.mean(0)

    # segment into 10 linearly spaced bins from min(meanf) to max(meanf)
    _, states = transfer_entropy.discretize(meanf, 100)

    # which frames fall below the cutoff?
    valid = states < cond_level

    timer.done()

    # preprocessing step (detect spikes, or just diff)
    # ------------------------------------------------------------------------
    timer = ElapsedTimer("* Extracting spikes")

    n = fnndeconv.apply_all_cells(fluor, disp=0, verbosity=0, spikes_tol=1E-4,
                                  spikes_maxiter=200)[0]

    expected_rate = 0.38    # obtained from simulation
    spikethresh, _ = test_spikedetect.get_thresh_from_rate(
        n[0], dt=dt, expected_rate=expected_rate, tstep=1E-2)
    spikes = (n // spikethresh)

    max_spikes = 2
    spikes[spikes > max_spikes] = max_spikes
    pre = spikes.astype(np.float64)
    post = pre
    # del spikes

    timer.done()

    # correlation / causality measure
    # ------------------------------------------------------------------------
    timer = ElapsedTimer("* Estimating connection probabilities")

    # weight_mat = correlation.get_best_xcorrs(pre, post, maxlag=3,
    #                                          normed=True)

    weight_mat = transfer_entropy.fast_gte_allpairs(post, pre,
                                                    markov_order=1,
                                                    valid=valid,
                                                    y_offset=1,
                                                    nlevels=post.max() + 1,
                                                    norm=True).T

    timer.done()

    # compute roc
    # ------------------------------------------------------------------------
    if compute_roc:
        network = g.network[:]
        fpr, tpr, auc = submission.run_auc(weight_mat, network,
                                           do_plot=do_roc_plot)
        return weight_mat, fpr, tpr, auc

    else:

        return weight_mat

def make_submission(f, g_valid, g_test, fname=None):

    """
    Make a new submission for the Kaggle Connectomics competition


    Arguments:
    -----------

        f: tables.file.File
            handle to the PyTables HDF5 file containing the data

        g_valid: tables.group.Group
            node containing the validation dataset, should have child nodes
            containing 'fluorescence', 'network_pos' and 'network' CArrays

        g_test: tables.group.Group
            node containing the test dataset, should have child nodes
            containing 'fluorescence', 'network_pos' and 'network' CArrays

        fname: string
            file name for saving the submission. by default it will be saved
            under 'submission_<date>.csv.gz'.

    """


    print "processing validation dataset"
    print "#" * 60
    weight_mat_valid = run(f, g_valid, compute_roc=False)

    print "processing test dataset"
    print "#" * 60
    weight_mat_test = run(f, g_test, compute_roc=False)

    print "saving results"
    print "#" * 60

    try:
        submission.make_submission(weight_mat_valid, weight_mat_test,
                                   fname=fname, norm=True, compress=True)
    except:
        pass
    finally:
        return weight_mat_valid, weight_mat_test