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
from scipy import ndimage
from matplotlib import pyplot as plt
import time

from utils.waitbar import s2h
from spikedetect import tridiag_solvers

from tempfile import mkstemp
from joblib import Parallel, delayed
from itertools import izip

DTYPE = np.float64
FMT = 'csc'
EPS = np.finfo(DTYPE).eps


def apply_all_cells(F, dt=0.02, n_jobs=-1, disp=1, *fnn_args, **fnn_kwargs):
    """
    Run FNN deconvolution on all rows of F in parallel
    """

    F = np.atleast_2d(F)
    nc, nt = F.shape

    pool = Parallel(n_jobs=n_jobs, verbose=disp, pre_dispatch='n_jobs * 2')

    results = pool(delayed(fnn_deconvolution)
                   (rr, *fnn_args, **fnn_kwargs) for rr in F)
    n, C, LL, theta = (np.vstack(a) for a in izip(*results))

    return n, C, LL, theta


def fnn_deconvolution(F, C0=None, theta0=None, dt=0.02,
                      learn_theta=(0, 0, 0, 0), params_tol=1E-3,
                      spikes_tol=1E-3, params_maxiter=10, spikes_maxiter=100,
                      verbosity=1, plot=False):
    """
    Infer spike trains from fluorescence using Fast Non-Negative Deconvolution
    ---------------------------------------------------------------------------

    This function uses an interior point method to solve the following
    optimization problem:

        n_best = argmax_{n >= 0} P(n | F)

    where n_best is a maximum a posteriori estimate for the most likely spike
    train, given the fluorescence signal F, and the model:

    C_{t} = gamma * C_{t-1} + n_t,          n_t ~ Poisson(lambda * dt)
    F_{t} = C_t + beta + epsilon,           epsilon ~ N(0, sigma)

    It is also possible to estimate the model parameters sigma, beta and lambda
    from the data using pseudo-EM updates.

    Arguments:
    ---------------------------------------------------------------------------
    F: ndarray, [nt]
        measured fluorescence values

    C0: ndarray, [nt]
        initial estimate of the calcium concentration for each time bin.

    theta0: len(4) sequence
        initial estimates of the model parameters (sigma, beta, lambda, gamma).

    dt: float scalar
        duration of each time bin (s)

    learn_theta: len(4) bool sequence
        specifies which of the model parameters to attempt learn via pseudo-EM
        iterations. currently gamma cannot be learned, and will raise an error.

    spikes_tol: float scalar
        termination condition for interior point spike train estimation:
            params_tol > abs((LL_prev - LL) / LL)

    params_tol: float scalar
        as above, but for the model parameter pseudo-EM estimation

    spikes_maxiter: int scalar
        maximum number of interior point iterations to estimate MAP spike train

    params_maxiter: int scalar
        maximum number of pseudo-EM iterations to estimate model parameters

    verbosity: int scalar
        0: no convergence messages (default)
        1: convergence messages for model parameters
        2: convergence messages for model parameters & MAP spike train

    plot: bool scalar
        live plot of n and (C + beta), updated during parameter estimation

    Returns:
    ---------------------------------------------------------------------------
    n_best: ndarray, [nt]
        MAP estimate of the most likely spike train

    C_best: ndarray, [nt]
        estimated intracellular calcium concentration (A.U.)

    LL_best: float scalar
        posterior log-likelihood of F given n_best and theta_best

    theta_best: len(4) tuple
        model parameters, updated according to learn_theta

    Reference:
    ---------------------------------------------------------------------------
    Vogelstein, J. T., Packer, A. M., Machado, T. a, Sippy, T., Babadi, B.,
    Yuste, R., & Paninski, L. (2010). Fast nonnegative deconvolution for spike
    train inference from population calcium imaging. Journal of
    Neurophysiology, 104(6), 3691-704. doi:10.1152/jn.01073.2009

    """

    tstart = time.time()

    nt = F.shape[0]

    if theta0 is None:
        theta_best = _init_theta(F, dt, hz=0.3, tau=0.5)
    else:
        theta_best = theta0

    # scale F to be between 0 and 1
    offset = F.min()
    scale = F.max() - offset
    F = (F - offset) / scale

    sigma, beta, lamb, gamma = theta_best

    # apply scale and offset to beta and sigma
    beta = (beta - offset) / scale
    sigma = sigma / scale

    theta_best = np.vstack((sigma, beta, lamb, gamma))

    if C0 is None:
        C = _init_C(F, dt)
    else:
        C = C0

    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        xmax = min(2000, nt)
        fr_t = np.arange(xmax) * dt
        ax[0].set_ylim(F.min(), F.max())
        ax[0].set_xlim(0, fr_t[-1])
        ax[0].hold(True)
        ax[1].set_ylim(0, 1)
        fig.canvas.draw()
        C_bg = fig.canvas.copy_from_bbox(ax[0].bbox)
        nt_bg = fig.canvas.copy_from_bbox(ax[1].bbox)
        F_line, = ax[0].plot(fr_t, F[:xmax], '-b')
        C_line, = ax[0].plot(fr_t, C[:xmax] + beta[0], '-r',
                             scalex=False, scaley=False)
        nt_line, = ax[1].plot(fr_t[1:xmax], np.ones(xmax - 1) * lamb * dt,
                              '-r', scalex=False, scaley=False)
        fig.canvas.draw()
        time.sleep(0.1)

    # if we're not learning the parameters, this step is all we need to do
    n_best, C_best, LL_best = estimate_MAP_spikes(
        F, C, theta_best, dt, spikes_tol, spikes_maxiter,
        verbosity
    )

    if plot:
        F_line.set_data(fr_t, F[:xmax])
        C_line.set_data(fr_t, C_best[:xmax] + theta_best[1])
        nt_line.set_data(fr_t[1:], n_best[:xmax - 1])
        fig.canvas.restore_region(C_bg)
        fig.canvas.restore_region(nt_bg)
        ax[0].draw_artist(C_line)
        ax[0].draw_artist(F_line)
        ax[1].draw_artist(nt_line)
        fig.canvas.blit(ax[0].bbox)
        fig.canvas.blit(ax[1].bbox)

    # pseudo-EM iterations to optimize the model parameters
    if np.any(learn_theta):

        if verbosity >= 1:
            sigma, beta, lamb, gamma = theta_best
            print('Params: iter=%3i; sigma=%6.4f, beta=%6.4f, '
                  'lambda=%6.4f, gamma=%6.4f; LL=%12.2f; delta_LL= N/A'
                  % (0, sigma, beta, lamb, gamma, LL_best))

        n = n_best
        C = C_best
        LL = LL_best
        theta = theta_best
        nloop_params = 1
        done = False

        while not done:

            # update the parameter estimates
            theta1 = _update_theta(n, C, F, theta, dt, learn_theta)

            # get the new n, C, and LL
            n1, C1, LL1 = estimate_MAP_spikes(
                F, C, theta1, dt, spikes_tol,
                spikes_maxiter, verbosity
            )

            # test for convergence
            delta_LL = -((LL1 - LL) / LL)

            if verbosity >= 1:
                sigma, beta, lamb, gamma = theta1

                print('Params: iter=%3i; sigma=%6.4f, beta=%6.4f, '
                      'lambda=%6.4f, gamma=%6.4f; LL=%12.2f; delta_LL= %8.4g'
                      % (nloop_params, sigma, beta, lamb, gamma, LL1,
                          delta_LL))

            if plot:
                C_line.set_data(fr_t, C1[:xmax] + theta1[1])
                nt_line.set_data(fr_t[1:xmax], n1[:xmax - 1])
                fig.canvas.restore_region(C_bg)
                fig.canvas.restore_region(nt_bg)
                ax[0].draw_artist(F_line)
                ax[0].draw_artist(C_line)
                ax[1].draw_artist(nt_line)
                fig.canvas.blit(ax[0].bbox)
                fig.canvas.blit(ax[1].bbox)
                time.sleep(0.1)

            # if the LL improved, keep these parameters
            if LL1 > LL_best:
                n_best, C_best, LL_best, theta_best = (
                    n1, C1, LL1, theta1)

            if (np.abs(delta_LL) < params_tol):
                if verbosity >= 1:
                    print("Parameters converged after %i iterations"
                          % (nloop_params))
                    print "Last delta log-likelihood:\t%8.4g" % delta_LL
                    print "Best posterior log-likelihood:\t%11.4f" % (
                        LL_best)
                done = True

            elif delta_LL < 0:
                if verbosity >= 1:
                    print 'Terminating because solution is diverging'
                done = True

            elif nloop_params > params_maxiter:
                if verbosity >= 1:
                    print 'Solution failed to converge before maxiter'
                done = True

            n, C, LL, theta = n1, C1, LL1, theta1
            nloop_params += 1

    if verbosity >= 1:
        time_taken = time.time() - tstart
        print "Completed: %s" % s2h(time_taken)

    sigma, beta, lamb, gamma = theta_best

    # correct for the offset and scaling we originally applied to F
    C_best *= scale
    beta *= scale
    beta += offset
    sigma *= scale

    # since we can't use FNND to estimate the spike probabilities in the 0th
    # timebin, for convenience we just concatenate (lamb * dt) to the start of
    # n so that it has the same shape as F and C
    n_best = np.concatenate((lamb * dt, n_best), axis=0)

    theta_best = np.hstack((sigma, beta, lamb, gamma))

    return n_best, C_best, LL_best, theta_best


def estimate_MAP_spikes(F, C, theta, dt, tol=1E-6, maxiter=100, verbosity=0):
    """
    Used internally by fnn_deconvolution to compute the maximum a posteriori
    spike train for a given set of fluorescence traces and model parameters.

    See the documentation for fnn_deconvolution for the meaning of the
    arguments

    Returns:    n_best, C_best, LL_best

    """

    sigma, beta, lamb, gamma = theta
    nt = F.shape[0]

    # used for computing the LL and gradient
    scale_var = 1. / (2 * sigma ** 2)
    lD = lamb * dt

    # used for computing the gradient
    grad_lnprior = np.zeros(nt, dtype=DTYPE)
    grad_lnprior[1:] = lD
    grad_lnprior[:-1] -= gamma * lD

    # initial estimate of spike probabilities (should be strictly non-negative)
    n = C[1:] - gamma * C[:-1]
    # assert not np.any(n < 0), "negative spike probabilities"

    # (predicted - actual) fluorescence
    res = F - (C + beta)

    # initialize the weight of the barrier term to 1
    z = 1.

    # compute initial posterior log-likelihood of the fluorescence
    LL = _post_LL(n, res, scale_var, lD, z)

    nloop1 = 0
    LL_prev = LL
    C_prev = C
    terminate_interior = False

    # in the outer loop we'll progressively reduce the weight of the barrier
    # term and check the interior point termination criteria
    while not terminate_interior:

        s = 1.
        d = 1.
        nloop2 = 0

        # converge for this barrier weight
        while (np.linalg.norm(d) > 1E-1) and (s > 1E-3):

            # compute direction of newton step
            d = _direction(n, res, sigma, gamma, scale_var, grad_lnprior, z)

            # ensure that s starts sufficiently small to guarantee that n
            # stays positive
            hit = n / (d[1:] - gamma * d[:-1])
            s = min(1., 0.99 * np.min(hit[hit >= EPS]))

            nloop3 = 0
            terminate_linesearch = False

            # backtracking line search for the largest step size that increases
            # the posterior log-likelihood of the fluorescence
            while not terminate_linesearch:

                # update estimated calcium
                C_new = C + (s * d)

                # update spike probabilities
                n = C_new[1:] - gamma * C_new[:-1]

                # (predicted - actual) fluorescence
                res = F - (C_new + beta)

                # compute the new posterior log-likelihood
                LL_new = _post_LL(n, res, scale_var, lD, z)
                # assert not np.any(np.isnan(LL_new)), "nan LL"

                # only update C & LL if LL improved
                if LL_new > LL:
                    LL = LL_new
                    C = C_new
                    terminate_linesearch = True

                # terminate when the step size is essentially zero but we're
                # still not improving (almost never happens in practice)
                elif s < EPS:
                    if verbosity >= 2:
                        print('terminated linesearch: s < EPS on %i iterations'
                              % nloop3)
                    terminate_linesearch = True

                if verbosity >= 2:
                    print('spikes: iter=%3i, %3i, %3i; z=%6.4f; s=%6.4f;'
                          ' LL=%13.4f' % (nloop1, nloop2, nloop3, z, s, LL))

                # reduce the step size
                s /= 5.
                nloop3 += 1

            nloop2 += 1

        # test for convergence
        delta_LL = np.abs((LL - LL_prev) / LL_prev)

        if (delta_LL < tol):
            terminate_interior = True

        elif z < EPS:
            if verbosity >= 2:
                print 'MAP spike train failed to converge before z -> 0'
            terminate_interior = True

        elif nloop1 > maxiter:
            if verbosity >= 2:
                print 'MAP spike train failed to converge within maxiter'
            terminate_interior = True

        LL_prev, C_prev = LL, C

        # increment the outer loop counter, reduce the barrier weight
        nloop1 += 1
        z /= 10.

    return n, C, LL


def _post_LL(n, res, scale_var, lD, z):

    # barrier term
    barrier = np.log(n).sum()       # this is a bottleneck

    # sum of squared (predicted - actual) fluorescence
    # res_ss = np.sum(res ** 2)
    res_ss = np.dot(res, res)       # faster sum of squares

    # weighted posterior log-likelihood of the fluorescence
    LL = -scale_var * res_ss - n.sum() * lD + z * barrier

    return LL


def _direction(n, res, sigma, gamma, scale_var, grad_lnprior, z):

    # gradient
    n_term = np.zeros_like(res)
    n_term[:n.shape[0]] = -gamma / n
    n_term[-n.shape[0]:] += 1. / n
    g = 2 * scale_var * res - grad_lnprior + z * n_term

    # main diagonal of the hessian
    n2 = n ** 2
    Hd0 = np.zeros_like(g)
    Hd0[:n.shape[0]] = gamma ** 2 / n2
    Hd0[-n.shape[0]:] += 1 / n2
    Hd0 *= -z
    Hd0 += -1. / sigma ** 2

    # upper/lower diagonals of the hessian
    Hd1 = z * gamma / n2

    # solve the tridiagonal system Hd = -g
    d = tridiag_solvers.trisolve(Hd1, Hd0, Hd1.copy(), -g, inplace=True)

    return d


def _update_theta(n, C, F, theta, dt, learn_theta):

    sigma, beta, lamb, gamma = theta
    learn_sigma, learn_beta, learn_lamb, learn_gamma = learn_theta

    nt = F.shape[0]

    if learn_sigma:
        res = F - (C + beta)
        # res_ss = np.sum(res ** 2)
        res_ss = np.dot(res, res)       # faster sum of squares
        sigma = np.sqrt(res_ss / (nt * dt))  # RMS error

    if learn_beta:
        # this converges very slowly!
        beta = np.sum(F - C) / nt

    if learn_lamb:
        lamb = np.sum(n) / (nt * dt)

    if learn_gamma:
        raise NotImplementedError('not sure how to learn gamma yet!')

    return np.vstack((sigma, beta, lamb, gamma)).reshape(4, 1)


def _init_theta(F, dt=0.02, hz=0.3, tau=0.5):

    orig_shape = F.shape
    F = np.atleast_2d(F)
    nc, nt = F.shape

    # K is the correction factor when using the median absolute deviation as a
    # robust estimator of the standard deviation of a normal distribution
    # http://en.wikipedia.org/wiki/Median_absolute_deviation
    K = 1.4785

    # noise parameter
    abs_dev = np.abs(F - np.median(F, axis=1, keepdims=True))
    sigma = np.median(abs_dev, axis=1) / K     # vector

    # baseline parameter
    beta = _hist_mode(F, row_wise=True)     # vector

    # rate parameter
    lamb = hz * np.ones(nc)

    # decay parameter (fraction of remaining fluorescence after one time step)
    gamma = (1. - (dt / tau)) * np.ones(nc)  # vector

    return np.vstack((sigma, beta, lamb, gamma)).reshape(4, 1)


def _init_C(F, dt=0.02, avg_win=1.0):

    orig_shape = F.shape
    F = np.atleast_2d(F)

    nc, nt = F.shape

    # boxcar smoothing
    win_len = max(1, avg_win / dt)
    win = np.ones(win_len) / win_len
    C0 = ndimage.convolve1d(F, win, axis=1, mode='reflect')

    return C0.reshape(orig_shape)


def _hist_mode(F, row_wise=False, nbins=None):
    """
    estimate the baseline fluorescence value from the histogram mode
    """
    if row_wise:
        F = np.atleast_2d(F)

        if nbins is None:
            nbins = int(np.ceil(np.sqrt(F.shape[1])))

        mode = []

        for row in F:
            counts, edges = np.histogram(row, bins=nbins)
            idx = np.argmax(counts)
            mode.append((edges[idx] + edges[idx + 1]) / 2.)

        return np.hstack(mode)

    else:
        if nbins is None:
            nbins = int(np.ceil(np.sqrt(F.shape[0])))

        counts, edges = np.histogram(F, bins=nbins)
        idx = np.argmax(counts)

        return (edges[idx] + edges[idx + 1]) / 2.

