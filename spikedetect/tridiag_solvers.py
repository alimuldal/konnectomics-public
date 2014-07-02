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
from cymodules import _fnndeconv

def trisolve(a, b, c, y, inplace=False):
    """
    The tridiagonal matrix (Thomas) algorithm for solving tridiagonal systems
    of equations:

        a_{i}x_{i-1} + b_{i}x_{i} + c_{i}x_{i+1} = y_{i}

    in matrix form:
        Mx = y

    TDMA is O(n), whereas standard Gaussian elimination is O(n^3).

    Arguments:
    -----------
        a: (n - 1,) vector
            the lower diagonal of M
        b: (n,) vector
            the main diagonal of M
        c: (n - 1,) vector
            the upper diagonal of M
        y: (n,) vector
            the result of Mx
        inplace:
            if True, and if b and y are both float64 vectors, they will be
            modified in place (may be faster)

    Returns:
    -----------
        x: (n,) vector
            the solution to Mx = y

    References:
    -----------
    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    http://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html
    """

    if (a.shape[0] != c.shape[0] or a.shape[0] >= b.shape[0]
            or b.shape[0] != y.shape[0]):
        raise ValueError('Invalid diagonal shapes')

    yshape_in = y.shape

    if y.ndim == 1:
        # needs to be (ldb, nrhs)
        y = y[:, None]

    rtype = np.result_type(a, b, c, y)

    if not inplace:
        # force a copy
        a = np.array(a, dtype=rtype, copy=True, order='C')
        b = np.array(b, dtype=rtype, copy=True, order='C')
        c = np.array(c, dtype=rtype, copy=True, order='C')
        y = np.array(y, dtype=rtype, copy=True, order='C')

    # this may also force copies if arrays have inconsistent types / incorrect
    # order
    a, b, c, y = (np.array(v, dtype=rtype, copy=False, order='C')
                  for v in (a, b, c, y))

    # y will now be modified in place to give the result
    if rtype == np.float32:
        _fnndeconv.TDMAs_lapacke(a, b, c, y)
    elif rtype == np.float64:
        _fnndeconv.TDMAd_lapacke(a, b, c, y)
    else:
        raise ValueError('Unsupported result type: %s' %rtype)

    return y.reshape(yshape_in)

def trisolve_offset(a, b, c, y, inplace=False):
    """
    A variant of TDMA for solving tridiagonal systems of equations where the
    upper and lower diagonals are offset by k rows/columns from the main
    diagonal:

        a_{i}x_{i-k} + b_{i}x_{i} + c_{i}x_{i+k} = y_{i}

    in matrix form:
        Mx = y

    TDMA is O(n), whereas standard Gaussian elimination is O(n^3).

    Arguments:
    -----------
        a: (n - k,) vector
            the lower diagonal of M
        b: (n,) vector
            the main diagonal of M
        c: (n - k,) vector
            the upper diagonal of M
        y: (n,) vector
            the result of Mx
        inplace:
            if True, and if b and y are both float64 vectors, they will be
            modified in place (may be faster)

    Returns:
    -----------
        x: (n,) vector
            the solution to Mx = y

    Reference:
    -----------
    http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """

    if (a.shape[0] != c.shape[0] or a.shape[0] >= b.shape[0]
            or b.shape != y.shape):
        raise ValueError('Invalid diagonal shapes')

    rtype = np.result_type(a, b, c, y)

    if not inplace:
        # force a copy
        b = np.array(b, dtype=rtype, copy=True, order='C')
        y = np.array(y, dtype=rtype, copy=True, order='C')

    # this may also force copies if arrays have inconsistent types / incorrect
    # order
    a, b, c, y = (np.array(v, dtype=rtype, copy=False, order='C')
                  for v in (a, b, c, y))

    # y will now be modified in place to give the result
    if rtype == np.float32:
        _fnndeconv.TDMAs_offset(a, b, c, y)
    elif rtype == np.float64:
        _fnndeconv.TDMAd_offset(a, b, c, y)
    else:
        raise ValueError('Unsupported result type: %s' %rtype)

    return y


def test_trisolve(n=20000, k=10):

    from scipy import sparse
    import scipy.sparse.linalg
    from scikits.sparse import cholmod

    # Cholesky factorisation
    def cholsolve(A, b, fac=None):
        if fac is None:
            fac = cholmod.cholesky(A)
        else:
            # store a representation of the factorised matrix and update it in
            # place
            fac.cholesky_inplace(A)
        x = fac.solve_A(b)
        return x, fac

    d0 = np.random.randn(n)
    d1 = np.random.randn(n - k)

    H = sparse.diags((d1, d0, d1), (-k, 0, k), format='csc')
    x = np.random.randn(n)
    g = np.dot(H, x)

    # xhat1 = g.copy()
    # _fnndeconv.TDMA_offset(d1.copy(), d0.copy(), d1.copy(), xhat1)
    xhat1 = trisolve(d1, d0, d1, g, inplace=False)
    xhat2 = cholsolve(H, g)[0]
    xhat3 = sparse.linalg.spsolve(H, g)
    xhat4 = trisolve_offset(d1, d0, d1, g)

    print "LAPACKE Thomas: ", np.linalg.norm(x - xhat1)
    print "Cholesky: ", np.linalg.norm(x - xhat2)
    print "LU: ", np.linalg.norm(x - xhat3)
    print "Offset Thomas: ", np.linalg.norm(x - xhat4)
