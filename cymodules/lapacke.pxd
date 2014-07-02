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

ctypedef int lapack_int

cdef extern from "lapacke.h" nogil:

    int LAPACK_ROW_MAJOR
    int LAPACK_COL_MAJOR

    lapack_int LAPACKE_sgtsv(int matrix_order,
                             lapack_int n,
                             lapack_int nrhs,
                             float * dl,
                             float * d,
                             float * du,
                             float * b,
                             lapack_int ldb)

    lapack_int LAPACKE_dgtsv(int matrix_order,
                             lapack_int n,
                             lapack_int nrhs,
                             double * dl,
                             double * d,
                             double * du,
                             double * b,
                             lapack_int ldb)