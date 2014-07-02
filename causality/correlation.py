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

from utils.waitbar import Waitbar


def get_best_xcorrs(A, B=None, maxlag=3, normed=True):

    lags, XC = allpairs_xcorr(A, B, maxlag=maxlag)

    bestlags, bestxcorrs = best_nonnegative_lags(lags, XC)

    if normed:
        # we normalize these to between 0 and 1 for the ROC analysis
        bestxcorrs = (bestxcorrs + 1) / 2.

    return bestxcorrs


def best_nonnegative_lags(lags, XC, nonnegative=True):
    """
    Find the maximum cross-correlation values for each pair of cells across lags

    Arguments
    ----------

        lags: int sequence
            list of relative lags (in timebins)

        XC: 3D float array
            [lag, i, j] array of cross-correlation values (see allpairs_xcorr)

        nonnegative:
            if True (default), only consider correlations at positive timelags

    Returns:
    ----------

        bestlags: 2D int array
            [i, j] array containing the lag values corresponding to the peaks
            in cross-correlation for each pair

        bestxcorrs: 2D float array
            [i, j] array containing the corresponding xcorr values

    """

    if nonnegative:
        # filter out negative lags
        XC = XC[lags >= 0, :, :]

    bestlags = np.argmax(XC, axis=0)

    # need to construct row, col indices into XC explicitly
    ii, jj = np.indices(XC.shape[1:])

    bestxcorrs = XC[bestlags, ii, jj]

    return bestlags, bestxcorrs


def allpairs_xcorr(A, B=None, maxlag=None, out=None, bias=False,
                   showprogress=True):
    """
    Compute the normalized cross-correlation for every possible pair of signals.

    Arguments:
    -----------

        A: 2D float array
            [m, n] array of 'leading' (causal) signals, where m corresponds to
            signals and n corresponds to timesteps

        B: 2D float array (optional)
            [m, n] array of 'lagging' (dependent) signals. If B is unspecified,
            we assume that B = A.

        maxlag: int
            the maximum absolute lag (in time bins) over which to compute the
            cross-correlation. by default, the cross-correlation correlation is
            computed over all possible lags, i.e. maxlag = (n - 1).

        out: 2D array, optional
            if specified, the cross-correlation result is written to this array

        bias: bool, optional
            if False (default), the normalization is by (N - 1), where N is the
            number of overlapping timebins for a given lag (unbiased estimate).
            if bias is True, the normalization is by N.

        showprogress: bool, optional
            show a progress bar

    Returns:
    -----------

        lags: 1D float array
            the lag values, for plotting convenience

        XC: 3D float array
            an [(2*maxlag + 1), m, m] array containing the cross-correlations
            for every pair of at each lag value.

            * XC[:, i, j] corresponds to r_(A[i]->B[j]) for positive lags
            * XC[:, j, i] corresponds to r_(A[j]->B[i]) for positive lags
            * XC[:, i, j] == XC[::-1, j, i] if A == B
    """

    # reshape an (n,) vector to a (1, n) matrix
    A = np.atleast_2d(A)

    # use cumsum to speed up finding the local sums for each lag
    csum_A = np.cumsum(A, axis=1)

    if B is None:
        B = A
        csum_B = csum_A
    else:
        if B.shape != A.shape:
            raise ValueError('A & B must have the same shape')
        csum_B = np.cumsum(B, axis=1)

    m, n = A.shape

    if maxlag is None:
        maxlag = n - 1

    # lags = np.arange(-maxlag, maxlag + 1, 1)
    lags = range(-maxlag, maxlag + 1, 1)    # indexing by integers is faster

    nlags = 2 * maxlag + 1

    if out is not None:

        # check we can fill the output array in place
        if not (out.shape == (nlags, m, m)
                and out.dtype == np.result_type(A, B)
                and out.flags.c_contiguous
                and out.flags.writeable):
            raise ValueError('invalid output array')

    else:

        # allocate a new output array
        out = np.empty((nlags, m, m), dtype=A.dtype, order='C')

    di, dj = np.diag_indices(m)

    if showprogress:
        wb = Waitbar(title='computing xcorr (%i cells, %i timesteps, %i lags)'
                     % (m, n, nlags), showETA=True)

    for ii, lag in enumerate(lags):

        if bias:
            # biased estimate of sample standard deviation
            n_overlap = float(n - abs(lag))

        else:
            # unbiased estimate of sample standard deviation
            n_overlap = float(n - abs(lag) - 1)

        if lag < 0:
            xmean = (csum_A[:, (lag - 1)]) / n_overlap
            ymean = (csum_B[:, -1] - csum_B[:, -(lag + 1)]) / n_overlap
            xdiff = A[:, :lag] - xmean[..., None]
            ydiff = B[:, -lag:] - ymean[..., None]

        elif lag > 0:
            xmean = (csum_A[:, -1] - csum_A[:, (lag - 1)]) / n_overlap
            ymean = (csum_B[:, -(lag + 1)]) / n_overlap
            xdiff = A[:,  lag:] - xmean[..., None]
            ydiff = B[:, :-lag] - ymean[..., None]

        else:
            xmean = (csum_A[:, -1]) / n_overlap
            ymean = (csum_B[:, -1]) / n_overlap
            xdiff = A - xmean[..., None]
            ydiff = B - ymean[..., None]

        # compute the covariance matrix
        # sum((x - mean(x)) * (y - mean(y)))
        np.dot(xdiff, ydiff.T, out=out[maxlag + lag])

        # divide by the product of the signal standard deviations
        # sqrt( sum((x - mean(x))^2) * sum((y - mean(y))^2) )
        # lsum_x2 = np.sum(xdiff * xdiff, axis=1)
        # lsum_y2 = np.sum(ydiff * ydiff, axis=1)

        # einsum is faster than sum
        lsum_x2 = np.einsum('ij,ij->i', xdiff, xdiff)
        lsum_y2 = np.einsum('ij,ij->i', ydiff, ydiff)
        fac = np.sqrt(lsum_x2[..., None] * lsum_y2[None, ...])

        out[maxlag + lag] /= fac

        if showprogress:
            wb.update((ii + 1.) / nlags)

    lags = np.hstack(lags)

    return lags, out


def xcorr1d(x, y, maxlag=None, bias=False):
    """
    Compute the normalized cross-correlation of two vectors, given by:

        r_xy(d) =           sum[(x_i - mu_x) * (y_(i-d) - mu_y)]
                    -----------------------------------------------------
                    sqrt( sum[(x_i - mu_x)^2] * sum[(y_(i-d) - mu_y)^2] )

    for timelag d from x->y. This is identical to the Pearson correlation
    coefficient for each lag value.

    Arguments:
    ----------

        x, y: 1D vectors
            input signals of length n

        maxlag: int, optional
            the maximum absolute lag (in time bins) over which to compute the
            cross-correlation. by default, the cross-correlation correlation is
            computed over all possible lags, i.e. maxlag = (n - 1).

        bias: bool, optional
            if False (default), the normalization is by (N - 1), where N is the
            number of overlapping timebins for a given lag (unbiased estimate).
            if bias is True, the normalization is by N.

    Returns:
    ----------

        lags: 1D vector
            lag values for each cross-correlation bin

        r: 1D vector
            corresponding cross-correlation values

    """

    if x.shape[0] != y.shape[0]:
        raise ValueError('Inputs must have same length')

    n = x.shape[0]

    if maxlag is None:
        maxlag = n - 1

    lags = np.arange(-maxlag, maxlag + 1, 1)
    nlags = 2 * maxlag + 1

    # allocate a new output array
    out = np.empty((nlags,), dtype=np.result_type(x, y), order='C')

    # use cumsum to speed up finding the local sums for each lag
    csum_x = np.cumsum(x)
    csum_y = np.cumsum(y)

    for lag in lags:

        if bias:
            n_overlap = float(n - abs(lag))

        else:
            n_overlap = float(n - abs(lag) - 1)

        if lag < 0:
            xmean = (csum_x[(lag - 1)]) / n_overlap
            ymean = (csum_y[-1] - csum_y[-(lag + 1)]) / n_overlap
            xdiff = x[:lag] - xmean
            ydiff = y[-lag:] - ymean

        elif lag > 0:
            xmean = (csum_x[-1] - csum_x[(lag - 1)]) / n_overlap
            ymean = (csum_y[-(lag + 1)]) / n_overlap
            xdiff = x[lag:] - xmean
            ydiff = y[:-lag] - ymean

        else:
            xmean = (csum_x[-1]) / n_overlap
            ymean = (csum_y[-1]) / n_overlap
            xdiff = x - xmean
            ydiff = y - ymean

        # sum((x - mean(x)) * (y - mean(y)))
        out[maxlag + lag] = np.dot(xdiff, ydiff)

        # divide by the product of the signal standard deviations
        # sqrt( sum((x - mean(x))^2) * sum((y - mean(y))^2) )
        lsum_x2 = np.sum(xdiff * xdiff)
        lsum_y2 = np.sum(ydiff * ydiff)
        out[maxlag + lag] /= np.sqrt(lsum_x2 * lsum_y2)

    return lags, out
