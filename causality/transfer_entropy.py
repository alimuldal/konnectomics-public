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

from cymodules import _fast_gte


def fast_gte(x, y, markov_order=2, nlevels=None, valid=None, y_offset=1,
             norm=True, return_pdf=False):
    """
    Compute the Generalized Transfer Entropy for a pair of signals

    Arguments:
    -----------
        x, y: uint8 arrays
            The dependent and causal signals. These are assumed to be insigned
            integers corresponding to bin indices.

        markov_order: uint8 scalar, either 1 or 2
            The Markov order for the two processes (i.e. the maximum number of
            bins 'back in time' to look when computing the PDF). At the moment
            only 1 and 2 are implemented in the fast version

        nlevels: uint8 scalar > 1
            The number of discrete states of the input signals. If unspecified,
            (x.max() + 1) is used, based on the assumption that x and y are
            bin indices.

        valid: bool array
            If specified, this array is used as a mask: only timebins where
            valid == True will be used when computing the joint PDF.

        y_offset: int scalar
            If y_offset > 0, signal y is shifted forward y_offset timebins
            relative to signal x, such that for a Markov order of N, the
            dimensions of the joint PDF will correspond to:

            x[t], x[t-1], ..., x[t-N], y[t-1+offset], ..., y[t-N+y_offset]

            The default y_offset of 1 is equivalent to allowing an
            'instantaneous' (same-timebin) causal influence from y -> x.
            Negative values would correspond to a delayed causal
            influence of y upon x, whereas values > 1 would correspond to
            'future' values of y influencing 'past' values of x.

        norm: bool scalar
            If True, the GTE is normalized by the entropy of the 'dependent'
            signal, i.e. it is a measure of the fraction of the total entropy
            of the 'dependent' signal that can be explained by the 'causal'
            signal. If False, the raw GTE (in bits) is returned.

        return_pdf: bool scalar
            If True, the joint PDF for the two signals is also returned

    Returns
    -----------
        gte: double scalar
            Generalized Transfer Entropy from y -> x

        pdf: double array (optional)
            The joint PDF for the two signals, returned if return_pdf == True
            (see above for dimensionality)

    Reference:
    -----------
    Stetter, O., Battaglia, D., Soriano, J., & Geisel, T. (2012). Model-Free
    Reconstruction of Excitatory Neuronal Connectivity from Calcium Imaging
    Signals. PLoS Computational Biology, 8(8), e1002653.
    doi:10.1371/journal.pcbi.1002653
    """

    if x.size != y.size:
        raise ValueError('Input arrays must have the same size')

    if x.dtype != np.uint8:
        warnings.warn('Casting x to uint8')
        x = np.uint8(x)
    if y.dtype != np.uint8:
        warnings.warn('Casting y to uint8')
        y = np.uint8(y)

    if nlevels is None:
        nlevels = np.uint(np.maximum(x.max(), y.max()) + 1)
    elif nlevels < 1 or np.maximum(x.max(), y.max()) + 1 > nlevels:
        raise ValueError('1 < nlevels <= max(max(x),max(y)) + 1')

    if valid is None:

        if markov_order == 1:
            pdf = _fast_gte.pdf_markov1(x.ravel(), y.ravel(), nlevels,
                                        y_offset)
            gte = _fast_gte.gte_from_pdf_markov1(pdf, nlevels, norm)

        elif markov_order == 2:
            pdf = _fast_gte.pdf_markov2(x.ravel(), y.ravel(), nlevels,
                                        y_offset)
            gte = _fast_gte.gte_from_pdf_markov2(pdf, nlevels, norm)

        else:
            raise NotImplementedError(
                'Only Markov orders of 1 & 2 are currently implemented')
    else:

        if valid.size != x.size:
            raise ValueError('valid must have the same size as x and y')

        valid = valid.astype(np.uint8)

        if markov_order == 1:
            pdf = _fast_gte.pdf_markov1_valid(x.ravel(), y.ravel(),
                                              valid.ravel(), nlevels,
                                              y_offset)
            gte = _fast_gte.gte_from_pdf_markov1(pdf, nlevels, norm)

        elif markov_order == 2:
            pdf = _fast_gte.pdf_markov2_valid(x.ravel(), y.ravel(),
                                              valid.ravel(), nlevels,
                                              y_offset)
            gte = _fast_gte.gte_from_pdf_markov2(pdf, nlevels, norm)

        else:
            raise NotImplementedError(
                'Only Markov orders of 1 & 2 are currently implemented')

    if return_pdf:
        return gte, pdf

    else:
        return gte


def fast_gte_allpairs(A, B=None, valid=None, markov_order=2, nlevels=None,
                      y_offset=1, norm=True, return_pdf=False):
    """
    Compute the Generalized Transfer Entropy in parallel for every possible
    pair of rows in an array of signals

    Arguments:
    -----------
        A: [nsignals, nt] uint8 array
            The dependent signals

        B: [nsignals, nt] uint8 array (optional)
            The (preprocessed?) causal signals. If unspecified, A = B.

        The values in A are assumed to be unsigned integers corresponding to
        bin indices

        (See the docstring for fast_gte for details of the other arguments)

    Returns
    -----------
        GTE: [nsignals, nsignals] double array
            GTE[i, j] contains the Generalized Transfer Entropy from
            B[j, :] --> A[i, :]

        PDF: [nsignals, nsignals, ...] double array (optional)
            GTE[i, j, ...] contains the joint PDF for B[j, :] --> A[i, :]

    """

    if B is None:
        B = A

    elif A.shape != B.shape:
            raise ValueError('A & B must have the same shape')

    if A.dtype != np.uint8:
        warnings.warn('Casting A to uint8')
        A = np.uint8(A)
    if B.dtype != np.uint8:
        warnings.warn('Casting B to uint8')
        B = np.uint8(B)
    if nlevels is None:
        nlevels = np.uint(np.maximum(A.max(), B.max()) + 1)
    elif nlevels < 1 or np.maximum(A.max(), B.max()) + 1 > nlevels:
        raise ValueError('1 < nlevels <= max(max(A),max(B)) + 1')

    if valid is None:

        if markov_order == 1:
            PDF = _fast_gte.pdf_markov1_parallel_allpairs(A, B, nlevels,
                                                          y_offset)
            GTE = _fast_gte.gte_from_pdf_markov1_allpairs(PDF, nlevels, norm)

        elif markov_order == 2:
            PDF = _fast_gte.pdf_markov2_parallel_allpairs(A, B, nlevels,
                                                          y_offset)
            GTE = _fast_gte.gte_from_pdf_markov2_allpairs(PDF, nlevels, norm)

        else:
            raise NotImplementedError(
                'Only Markov orders of 1 & 2 are currently implemented')
    else:

        if valid.shape != A.shape[1:]:
            raise ValueError(
                'valid must have the same number of timebins as A & B')
        valid = valid.astype(np.uint8)

        if markov_order == 1:
            PDF = _fast_gte.pdf_markov1_parallel_allpairs_valid(
                A, B, valid, nlevels, y_offset)
            GTE = _fast_gte.gte_from_pdf_markov1_allpairs(PDF, nlevels, norm)

        elif markov_order == 2:
            PDF = _fast_gte.pdf_markov2_parallel_allpairs_valid(
                A, B, valid, nlevels, y_offset)
            GTE = _fast_gte.gte_from_pdf_markov2_allpairs(PDF, nlevels, norm)

        else:
            raise NotImplementedError(
                'Only Markov orders of 1 & 2 are currently implemented')

    if return_pdf:
        return np.array(GTE), np.array(PDF)

    else:
        return np.array(GTE)


def gte(x, y, markov_order=2, nlevels=None, y_offset=1, norm=True):
    """
    Compute the Generalized Transfer Entropy for a pair of signals

    Arguments:
    -----------

        x, y: uint8 sequences
            The two sequences are assumed to be insigned integers corresponding
            to bin indices

        markov_order: uint8 scalar
            The Markov order for the two processes (i.e. the maximum number of
            bins 'back in time' to look when computing the PDF).

        nlevels: uint8 scalar > 1
            The number of discrete states of the input signals. If unspecified,
            (x.max() + 1) is used, based on the assumption that x and y are
            unsigned integer indices.

        y_offset: bool scalar
            If y_offset == True (default), sequence y is shifted forward one
            timebin relative to x, such that for a Markov order of N, the
            dimensions of the joint PDF will correspond to:

                (x[t], x[t-1], ..., x[t - N], y[t], ..., y[t-N+1])

            If y_offset == False:

                (x[t], x[t-1], ..., x[t - N], y[t-1], ..., y[t-N])

        norm: bool scalar
            If True, the GTE is normalized by the entropy of the 'dependent'
            signal, i.e. it is a measure of the fraction of the total entropy
            of the 'dependent' signal that can be explained by the 'causal'
            signal. If False, the raw GTE (in bits) is returned.

    Returns
    -----------

        gte: double scalar
            Generalized Transfer Entropy from y --> x

    Reference:
    -----------
    Stetter, O., Battaglia, D., Soriano, J., & Geisel, T. (2012). Model-Free
    Reconstruction of Excitatory Neuronal Connectivity from Calcium Imaging
    Signals. PLoS Computational Biology, 8(8), e1002653.
    doi:10.1371/journal.pcbi.1002653
    """

    for a in (x, y):
        if a.dtype != np.uint8:
            warnings.warn('Casting input array to uint8')
            a = a.astype(np.uint8)
    if nlevels is None:
        nlevels = x.max() + 1

    return gte_from_pdf(pdf(x, y, markov_order, nlevels, y_offset), norm)


def pdf(x, y, markov_order=2, nlevels=None, y_offset=1):
    """
    Compute the empirical joint PDF for two Markov processes

    Arguments:
    -----------

        x, y: uint sequences
            The two sequences are assumed to be insigned integers corresponding
            to bin indices

        markov_order: uint scalar > 0
            The Markov order for the two processes (i.e. the maximum number of
            bins 'back in time' to look when computing the PDF)

        nlevels: uint scalar > 1
            The number of discrete states of the input signals. If unspecified,
            (x.max() + 1) is used, based on the assumption that x and y are
            unsigned integer indices.

        y_offset: bool scalar
            If y_offset == True (default), sequence y is shifted forward one
            timebin relative to x, such that for a Markov order of N, the
            dimensions of the joint PDF will correspond to:

                (x[t], x[t-1], ..., x[t - N], y[t], ..., y[t-N+1])

            If y_offset == False:

                (x[t], x[t-1], ..., x[t - N], y[t-1], ..., y[t-N])

    Returns
    -----------

        pdf: double array
            The (2 * N) + 1 dimensional joint PDF, normalized to sum to 1

    """

    y_offset = np.bool(y_offset)

    if nlevels is None:
        nlevels = x.max() + 1

    ndims = (2 * markov_order) + 1
    outdims = (nlevels,) * ndims

    out = np.zeros(outdims, np.float64)

    # do the first step (handle numpy's annoying slice indexing behavior)
    xidx = slice(markov_order, None, -1)
    if y_offset:
        yidx = slice(markov_order, 0, -1)
    else:
        yidx = slice(markov_order - 1, None, -1)
    out[tuple(x[xidx]) + tuple(y[yidx])] += 1

    # do the rest
    for tt in xrange(markov_order + 1, x.size):

        xidx = slice(tt, tt - (markov_order + 1), -1)
        yidx = slice(tt - 1 + y_offset,
                     tt - (markov_order + 1) + y_offset,
                     -1)

        out[tuple(x[xidx]) + tuple(y[yidx])] += 1

    return out.astype(np.float64) / (x.size - markov_order)


def gte_from_pdf(pdf, norm=True):
    """
    Compute the Generalized Transfer Entropy from the empirical joint PDF for
    two Markov processes

    See the docstrings for gte() and pdf() for more details
    """

    markov_order = (pdf.ndim - 1) / 2

    tmpsum = pdf.sum(0, keepdims=True)
    numer = np.where(tmpsum > 0, pdf / tmpsum, 0)

    cj = pdf.sum(tuple(xrange(markov_order + 1, pdf.ndim)), keepdims=True)
    # cj = pdf.sum((3, 4, 5), keepdims=True)
    tmpsum = cj.sum(0, keepdims=True)
    denom = np.where(tmpsum > 0, cj / tmpsum, 0)

    ratio = numer / denom
    logratio = np.where(ratio > 0, np.log2(ratio), 0)

    gte = (pdf * logratio).sum()

    # normalize by the entropy of the putative 'dependent' signal
    if norm:
        gte /= entropy(pdf)

    return gte


def entropy(pdf):
    """
    Compute the entropy of a signal from its empirical PDF

    Arguments:
    -----------

        pdf: double array
            The empirical PDF. This can have any number of dimensions, but the
            first dimension must correspond to unique states of the signal at
            time t, and the whole thing must sum to 1

    Returns:
    -----------

        entropy: double scalar
            The entropy of the signal in bits

    """

    conditional = pdf.sum(tuple(xrange(1, pdf.ndim)))
    # conditional = pdf.sum((1, 2, 3, 4, 5), keepdims=True)
    log_conditional = np.where(conditional > 0, np.log2(conditional), 0)
    return -1 * (conditional * log_conditional).sum()


def discretize(A, nbins=3):
    """
    Discretize a set of continuous signals into linearly spaced bins

    Arguments:
    ------------
        A: float ndarray
            Array of continuous values

        nbins: uint scalar, 1 < nbins < 255
            The number of discrete bins to use. The bins will be evenly spaced
            between A.min() and A.max(), such that every value in A falls into
            a bin.

    Returns:
    ------------
        bin_edges: float array
            (nbins + 1) array of bin edge values

        B: uint8 ndarray
            The values in B correspond to indices into bin_edges, such
            that value ii in B corresponds to where:
                bin_edges[ii - 1] <= A < bin_edges[ii]

    """

    if not (1 < nbins < 255):
        raise ValueError('nbins must be > 1 and < 2**8')

    # a filthy hack to make sure that the maximum value in A falls into the bin
    # with index range(nbins)[-1]. I'm sure there must be a better way to do
    # this...
    epsilon = np.finfo(A.dtype).eps
    fmin, fmax = A.min(), A.max() + epsilon

    bin_edges = np.linspace(fmin, fmax, nbins + 1, endpoint=True)
    discretized = np.digitize(A.ravel(), bin_edges) - 1
    return bin_edges, discretized.reshape(A.shape).astype(np.uint8)


def pdf_markov2(x, y, y_offset=1, nlevels=3):
    """
    Compute the empirical joint PDF for two processes of Markov order 2. This
    version is a bit quicker than the more general pdf() function.

    See the docstring for pdf for more info.
    """

    y_offset = np.bool(y_offset)

    # out = np.ones((nlevels,)*6, np.uint32)
    out = np.zeros((nlevels,) * 5, np.float64)
    n = x.size

    for tt in xrange(2, x.size):

        # out[x[tt], x[tt - 1], x[tt - 2], y[tt], y[tt - 1], y[tt - 2]] += 1

        # offset signal y by +1 if we want to allow same-timebin interactions
        out[x[tt], x[tt - 1], x[tt - 2],
            y[tt - 1 + y_offset], y[tt - 2 + y_offset]] += 1

    return out / (n - 2.)

# for debugging purposes / satisfying Tim's for loop fetish
# def gte_markov2_forloop(pdf, nlevels=3):

# cast to floats and normalize
# pdf = pdf.astype(np.float32) / pdf.sum()

#     numer = np.zeros((nlevels,) * 5)

#     for jj in xrange(nlevels):
#         for kk in xrange(nlevels):
#             for ll in xrange(nlevels):
#                 for mm in xrange(nlevels):
#                     numer[:, jj, kk, ll, mm] = (
#                         pdf[:, jj, kk, ll, mm] /
#                         pdf[:, jj, kk, ll, mm].sum())

#     denom = np.zeros((nlevels,) * 3)

#     for ii in xrange(nlevels):
#         for jj in xrange(nlevels):
#             for kk in xrange(nlevels):
#                 denom[ii, jj, kk] = pdf[ii, jj, kk, ...].sum()

#     for jj in xrange(nlevels):
#         for kk in xrange(nlevels):
#             denom[:, jj, kk] = denom[:, jj, kk] / denom[:, jj, kk].sum()

#     gte = 0.

#     for ii in xrange(nlevels):
#         for jj in xrange(nlevels):
#             for kk in xrange(nlevels):
#                 for ll in xrange(nlevels):
#                     for mm in xrange(nlevels):
#                         gte = (gte + pdf[ii, jj, kk, ll, mm] *
#                                np.log2(numer[ii, jj, kk, ll, mm] /
#                                        denom[ii, jj, kk]))
#     return gte
