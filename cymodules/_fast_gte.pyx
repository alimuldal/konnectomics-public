#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

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
cimport cython
cimport numpy as np

from cython.parallel cimport prange
# from cython.view cimport array as cvarray

from libc.math cimport log2
from libc.stdlib cimport malloc, free

# -----------------------------------------------------------------------------
# Compute empirical PDF
#------------------------------------------------------------------------------

# for a single pair
# -----------------------------------------------------------------------------

cpdef double[:, :, :] pdf_markov1(unsigned char[:] x, unsigned char[:] y,
                                  unsigned int nlevels, int y_offset):

    cdef:
        unsigned int nt = x.shape[0]
        double step = 1. / (nt - 1)
        double[:, :, :] pdf = np.zeros((nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt

    for tt in range(1, nt):
        # y -> x
        pdf[x[tt], x[tt - 1], y[tt - 1 + y_offset]] += step

    return pdf

cpdef double[:, :, :, :, :] pdf_markov2(unsigned char[:] x, unsigned char[:] y,
                                        unsigned int nlevels, int y_offset):

    cdef:
        unsigned int nt = x.shape[0]
        double step = 1. / (nt - 2)
        double[:, :, :, :, :] pdf = np.zeros(
            (nlevels, nlevels, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt

    for tt in range(2, nt):

        # y -> x
        pdf[x[tt], x[tt - 1], x[tt - 2],
            y[tt - 1 + y_offset], y[tt - 2 + y_offset]] += step

    return pdf

cpdef double[:, :, :] pdf_markov1_valid(unsigned char[:] x,
                                        unsigned char[:] y,
                                        unsigned char[:] valid,
                                        unsigned int nlevels,
                                        int y_offset):
    cdef:
        unsigned int nt = x.shape[0]
        double nvalid = 0
        double[:, :, :] pdf = np.zeros((nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t tt, ii, jj, kk

    for tt in range(1, nt):
        if valid[tt] and valid[tt - 1] and valid[tt - 1 + y_offset]:

            # y -> x
            pdf[x[tt], x[tt - 1], y[tt - 1 + y_offset]] += 1
            nvalid += 1

    # normalize to sum to 1
    for ii in range(nlevels):
        for jj in range(nlevels):
            for kk in range(nlevels):
                pdf[ii, jj, kk] /= nvalid

    return pdf

cpdef double[:, :, :, :, :] pdf_markov2_valid(unsigned char[:] x,
                                              unsigned char[:] y,
                                              unsigned char[:] valid,
                                              unsigned int nlevels,
                                              int y_offset):
    cdef:
        unsigned int nt = x.shape[0]
        double nvalid = 0
        double[:, :, :, :, :] pdf = np.zeros(
            (nlevels, nlevels, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t tt, ii, jj, kk, ll, mm

    for tt in range(2, nt):
        if (valid[tt] and valid[tt - 1] and valid[tt - 2]
                and valid[tt - 1 + y_offset] and valid[tt - 2 + y_offset]):

            # y -> x
            pdf[x[tt], x[tt - 1], x[tt - 2],
                y[tt - 1 + y_offset], y[tt - 2 + y_offset]] += 1
            nvalid += 1

    # normalize to sum to 1
    for ii in range(nlevels):
        for jj in range(nlevels):
            for kk in range(nlevels):
                for ll in range(nlevels):
                    for mm in range(nlevels):
                        pdf[ii, jj, kk, ll, mm] /= nvalid

    return pdf

# parallelized over all pairs
# -----------------------------------------------------------------------------

cpdef double[:, :, :, :, :] pdf_markov1_parallel_allpairs(
        unsigned char[:, :] A,
        unsigned char[:, :] B,
        unsigned int nlevels,
        int y_offset):

    cdef:
        unsigned int nc = A.shape[0]
        unsigned int nt = A.shape[1]
        double step = 1. / (nt - 1)
        double[:, :, :, :, :] pdf = np.zeros(
            (nc, nc, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt

    # parallelize over the outer loop
    for aa in prange(nc, nogil=True, schedule='guided'):
        for bb in range(nc):
            if aa != bb:
                for tt in range(1, nt):

                    # bb -> aa
                    pdf[aa, bb, A[aa, tt], A[aa, tt - 1],
                        B[bb, tt - 1 + y_offset]] += step

    return pdf

cpdef double[:, :, :, :, :, :, :] pdf_markov2_parallel_allpairs(
        unsigned char[:, :] A,
        unsigned char[:, :] B,
        unsigned int nlevels,
        int y_offset):

    cdef:
        unsigned int nc = A.shape[0]
        unsigned int nt = A.shape[1]
        double step = 1. / (nt - 2)
        double[:, :, :, :, :, :, :] pdf = np.zeros(
            (nc, nc, nlevels, nlevels, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt

    # parallelize over the outer loop
    for aa in prange(nc, nogil=True, schedule='guided'):
        for bb in range(nc):
            if aa != bb:
                for tt in range(2, nt):

                    # bb -> aa
                    pdf[aa, bb,
                        A[aa, tt], A[aa, tt - 1], A[aa, tt - 2],
                        B[bb, tt - 1 + y_offset], B[bb, tt - 2 + y_offset]
                        ] += step

    return pdf

cpdef double[:, :, :, :, :] pdf_markov1_parallel_allpairs_valid(
        unsigned char[:, :] A,
        unsigned char[:, :] B,
        unsigned char[:] valid,
        unsigned int nlevels,
        int y_offset):

    cdef:
        unsigned int nc = A.shape[0]
        unsigned int nt = A.shape[1]
        double step, nvalid = 0
        double[:, :, :, :, :] pdf = np.zeros(
            (nc, nc, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt, ii, jj, kk

    # get the total # of valid frames
    for tt in range(2, nt):
        if (valid[tt] and valid[tt - 1] and valid[tt - 1 + y_offset]):
            nvalid += 1

    step = 1. / nvalid

    # parallelize over the outer loop
    for aa in prange(nc, nogil=True, schedule='guided'):
        for bb in range(nc):
            if aa != bb:
                for tt in range(1, nt):
                    if (valid[tt] and valid[tt - 1]
                            and valid[tt - 1 + y_offset]):

                        # bb -> aa
                        pdf[aa, bb, A[aa, tt], A[aa, tt - 1],
                            B[bb, tt - 1 + y_offset]] += step

    return pdf

cpdef double[:, :, :, :, :, :, :] pdf_markov2_parallel_allpairs_valid(
        unsigned char[:, :] A,
        unsigned char[:, :] B,
        unsigned char[:] valid,
        unsigned int nlevels,
        int y_offset):

    cdef:
        unsigned int nc = A.shape[0]
        unsigned int nt = A.shape[1]
        double step, nvalid = 0
        double[:, :, :, :, :, :, :] pdf = np.zeros(
            (nc, nc, nlevels, nlevels, nlevels, nlevels, nlevels), np.float64)
        Py_ssize_t aa, bb, tt

    # get the total # of valid frames
    for tt in range(2, nt):
        if (valid[tt] and valid[tt - 1] and valid[tt - 2]
                and valid[tt - 1 + y_offset] and valid[tt - 2 + y_offset]):
            nvalid += 1
    step = 1. / nvalid

    # parallelize over the outer loop
    for aa in prange(nc, nogil=True, schedule='guided'):
        for bb in range(nc):
            if aa != bb:
                for tt in range(2, nt):
                    if (valid[tt] and valid[tt - 1] and valid[tt - 2]
                        and valid[tt - 1 + y_offset]
                            and valid[tt - 2 + y_offset]):

                        # bb -> aa
                        pdf[aa, bb,
                            A[aa, tt], A[aa, tt - 1], A[aa, tt - 2],
                            B[bb, tt - 1 + y_offset], B[bb, tt - 2 + y_offset]
                            ] += step

    return pdf

# -----------------------------------------------------------------------------
# Compute the GTE from the empirical PDF
# -----------------------------------------------------------------------------

# for a single pair
# -----------------------------------------------------------------------------

cpdef double gte_from_pdf_markov1(double[:, :, :] pdf, unsigned int nlevels,
                                  unsigned char norm):

    cdef:
        Py_ssize_t ii, jj, kk, ll, mm
        double tmpsum, tmpratio, step, p_xi, entropy, gte

        double[:, :] denom
        double[:, :, :] numer

    denom_carr = <double * > malloc((nlevels ** 2) * sizeof(double))
    denom = < double[:nlevels, :nlevels] > denom_carr
    numer_carr = <double * > malloc((nlevels ** 3) * sizeof(double))
    numer = < double[:nlevels, :nlevels, :nlevels] > numer_carr

    for jj in range(nlevels):
        for kk in range(nlevels):
            tmpsum = 0
            for ii in range(nlevels):
                tmpsum += pdf[ii, jj, kk]
            if tmpsum > 0:
                for ii in range(nlevels):
                    numer[ii, jj, kk] = pdf[ii, jj, kk] / tmpsum

    entropy = 0
    for ii in range(nlevels):
        p_xi = 0
        for jj in range(nlevels):
            tmpsum = 0
            for kk in range(nlevels):
                tmpsum += pdf[ii, jj, kk]
            denom[ii, jj] = tmpsum
            p_xi += tmpsum
        if p_xi > 0:
            entropy -= p_xi * log2(p_xi)

    for jj in range(nlevels):
        tmpsum = 0
        for ii in range(nlevels):
            tmpsum += denom[ii, jj]
        if tmpsum > 0:
            for ii in range(nlevels):
                    denom[ii, jj] /= tmpsum

    gte = 0
    for ii in range(nlevels):
        for jj in range(nlevels):
            for kk in range(nlevels):
                if denom[ii, jj] > 0:
                    tmpratio = numer[ii, jj, kk] / denom[ii, jj]
                    if tmpratio > 0:
                        gte += pdf[ii, jj, kk] * log2(tmpratio)

    # normalize by entropy of dependent signal
    if norm == 1:
        if entropy > 0:
            gte /= entropy

    # NB: remember to free the temporary C arrays, or we'll have a memory leak!
    free(denom_carr)
    free(numer_carr)

    return gte


cpdef double gte_from_pdf_markov2(double[:, :, :, :, :] pdf,
                                  unsigned int nlevels,
                                  unsigned char norm):

    cdef:
        Py_ssize_t ii, jj, kk, ll, mm
        double tmpsum, tmpratio, step, p_xi, entropy, gte

        double[:, :, :] denom
        double[:, :, :, :, :] numer

    # since these arrays are only used internally, it's better to avoid the
    # overhead of creating them as Python objects. However this does mean we
    # have to be careful to manually free them afterwards, since they won't be
    # automatically garbage-collected.
    denom_carr = <double * > malloc((nlevels ** 3) * sizeof(double))
    denom = < double[:nlevels, :nlevels, :nlevels] > denom_carr
    numer_carr = <double * > malloc((nlevels ** 5) * sizeof(double))
    numer  = (< double[:nlevels, :nlevels, :nlevels, :nlevels, :nlevels] >
               numer_carr)

    # equivalent to:
    #   numer = pdf / pdf.sum(0, keepdims=True)
    for jj in range(nlevels):
        for kk in range(nlevels):
            for ll in range(nlevels):
                for mm in range(nlevels):

                    # sum over the first dimension
                    tmpsum = 0

                    for ii in range(nlevels):
                        tmpsum += pdf[ii, jj, kk, ll, mm]

                    # avoid / 0
                    if tmpsum > 0:
                        for ii in range(nlevels):
                            numer[ii, jj, kk, ll, mm] = (
                                pdf[ii, jj, kk, ll, mm] / tmpsum)

    # equivalent to:
    #   cj = pdf.sum((3, 4, 5), keepdims=True)
    #   denom = cj / cj.sum(0, keepdims=True)
    entropy = 0

    for ii in range(nlevels):

        # sum over all but the first dimension
        p_xi = 0

        for jj in range(nlevels):
            for kk in range(nlevels):

                # sum over the last 3 dimensions
                tmpsum = 0

                for ll in range(nlevels):
                    for mm in range(nlevels):
                        tmpsum += pdf[ii, jj, kk, ll, mm]

                denom[ii, jj, kk] = tmpsum
                p_xi += tmpsum

        # avoid log2(0)
        if p_xi > 0:

            # H(X) = - sum(P{xi} * log2(P{xi}))
            entropy -= p_xi * log2(p_xi)

    for jj in range(nlevels):
        for kk in range(nlevels):

            # sum over first dimension
            tmpsum = 0

            for ii in range(nlevels):
                tmpsum += denom[ii, jj, kk]

            # avoid / 0
            if tmpsum > 0:
                for ii in range(nlevels):
                        denom[ii, jj, kk] /= tmpsum

    # equivalent to:
    #   gte = (pdf * np.log2(numer / denom)).sum()
    gte = 0

    for ii in range(nlevels):
        for jj in range(nlevels):
            for kk in range(nlevels):
                for ll in range(nlevels):
                    for mm in range(nlevels):

                            if denom[ii, jj, kk] > 0:   # avoid / 0

                                tmpratio = (numer[ii, jj, kk, ll, mm] /
                                            denom[ii, jj, kk])

                                # avoid log2(0)
                                if tmpratio > 0:

                                    # using the log2 function from libc.math
                                    # was critical for getting good performance
                                    # here
                                    gte += (pdf[ii, jj, kk, ll, mm] *
                                            log2(tmpratio))

    # normalize by entropy of dependent signal
    if norm == 1:
        if entropy > 0:
            gte /= entropy

    # NB: remember to free the temporary C arrays, or we'll have a memory leak!
    free(denom_carr)
    free(numer_carr)

    return gte

# over all pairs
# -----------------------------------------------------------------------------

cpdef double[:, :] gte_from_pdf_markov1_allpairs(
        double[:, :, :, :, :] pdf,
        unsigned int nlevels,
        unsigned char norm):

    cdef:
        int nc = pdf.shape[0]
        Py_ssize_t aa, bb
        double[:, :] gte = np.zeros((nc, nc), np.float64)

    for aa in range(nc):
        for bb in range(nc):
            gte[aa, bb] = gte_from_pdf_markov1(pdf[aa, bb, :, :, :],
                                               nlevels, norm)

    return gte

cpdef double[:, :] gte_from_pdf_markov2_allpairs(
        double[:, :, :, :, :, :, :] pdf,
        unsigned int nlevels,
        unsigned char norm):

    cdef:
        int nc = pdf.shape[0]
        Py_ssize_t aa, bb
        double[:, :] gte = np.zeros((nc, nc), np.float64)

    # sadly, there are fundamental reasons why we can't prange this! It's still
    # a lot quicker to loop in C, though...
    for aa in range(nc):
        for bb in range(nc):
            gte[aa, bb] = gte_from_pdf_markov2(pdf[aa, bb, :, :, :, :, :],
                                               nlevels, norm)

    return gte
