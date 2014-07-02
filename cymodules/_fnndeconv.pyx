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

cimport cython
from lapacke cimport *

# Standard Thomas algorithm for solving Ax = d where A is a tridiagonal
# matrix:
#
# a[i]x[i-1] + b[i]x[i] + c[i]x[i+1] = d[i]
#
# this is just a thin wrapper for LAPACK's {s|d}gtsv function, see:
# http://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html
# -----------------------------------------------------------------------------

cpdef TDMAs_lapacke(float[::1] DL, float[::1] D, float[::1] DU,
                    float[:, ::1] B):
    cdef:
        lapack_int n = D.shape[0]
        lapack_int nrhs = B.shape[1]
        lapack_int ldb = B.shape[0]
        float * dl = &DL[0]
        float * d = &D[0]
        float * du = &DU[0]
        float * b = &B[0, 0]
        lapack_int info

    info = LAPACKE_sgtsv(LAPACK_COL_MAJOR, n, nrhs, dl, d, du, b, ldb)

    return info

cpdef TDMAd_lapacke(double[::1] DL, double[::1] D, double[::1] DU,
                    double[:, ::1] B):
    cdef:
        lapack_int n = D.shape[0]
        lapack_int nrhs = B.shape[1]
        lapack_int ldb = B.shape[0]
        double * dl = &DL[0]
        double * d = &D[0]
        double * du = &DU[0]
        double * b = &B[0, 0]
        lapack_int info

    info = LAPACKE_dgtsv(LAPACK_COL_MAJOR, n, nrhs, dl, d, du, b, ldb)

    return info

# Generalized Thomas algorithm for solving Ax = d where A is a tridiagonal
# matrix with an arbitrary diagonal offset, i.e.:
#
# a[i]x[i-k] + b[i]x[i] + c[i]x[i+k] = d[i]
#
# -----------------------------------------------------------------------------
# http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

cpdef TDMAs_offset(float[:] a, float[:] b, float[:] c, float[:] d):

    cdef:
        Py_ssize_t m = b.shape[0]
        Py_ssize_t n = a.shape[0]
        Py_ssize_t k = m - n
        Py_ssize_t ii

    for ii in range(n):
        b[ii + k] -= c[ii] * a[ii] / b[ii]
        d[ii + k] -= d[ii] * a[ii] / b[ii]

    for ii in range(n - 1, -1, -1):
        d[ii] -= d[ii + k] * c[ii] / b[ii + k]

    for ii in range(m):
        d[ii] /= b[ii]

cpdef TDMAd_offset(double[:] a, double[:] b, double[:] c, double[:] d):

    cdef:
        Py_ssize_t m = b.shape[0]
        Py_ssize_t n = a.shape[0]
        Py_ssize_t k = m - n
        Py_ssize_t ii

    for ii in range(n):
        b[ii + k] -= c[ii] * a[ii] / b[ii]
        d[ii + k] -= d[ii] * a[ii] / b[ii]

    for ii in range(n - 1, -1, -1):
        d[ii] -= d[ii + k] * c[ii] / b[ii + k]

    for ii in range(m):
        d[ii] /= b[ii]