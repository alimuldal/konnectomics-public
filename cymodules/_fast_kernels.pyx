#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport cython
cimport numpy as np

from cython.parallel cimport parallel, prange
# from libc.stdlib cimport abort, malloc, free


# -----------------------------------------------------------------------------
# Calcium dynamics
#------------------------------------------------------------------------------

# for a single trace
# -----------------------------------------------------------------------------
cpdef double[:] ca_from_spikes(double[:] S, double dt, double A,
                               double tau):

    cdef:
        unsigned int nt = S.shape[0]
        double[:] C = np.empty((nt,), np.float64)
        Py_ssize_t tt

    C[0] = A * S[0]

    for tt in range(1, nt):
        C[tt] = C[tt - 1] + (-(dt / tau) * C[tt - 1] + (A * S[tt]))

    return C

# parallelized over all traces
# -----------------------------------------------------------------------------
cpdef double[:, :] ca_from_spikes_parallel(double[:, :] S, double dt,
                                           double A, double tau):

    cdef:
        unsigned int nc = S.shape[0]
        unsigned int nt = S.shape[1]
        double[:, :] C = np.empty((nc, nt), np.float64)
        Py_ssize_t cc, tt

    # for cc in range(nc):
    for cc in prange(nc, nogil=True, schedule='guided'):

        # NB there may be spikes in time bin 0
        C[cc, 0] = A * S[cc, 0]

        # internal calcium dynamics
        for tt in range(1, nt):
            C[cc, tt] = C[cc, tt - 1] + (
                -(dt / tau) * C[cc, tt - 1] + (A * S[cc, tt]))

    return C

# -----------------------------------------------------------------------------
# Presynaptic kernels
#------------------------------------------------------------------------------

# for a single trace
#------------------------------------------------------------------------------

cpdef double[:] binned_tsodyks(double[:] S, double dt, double tau_rec,
                               double U):

    cdef:
        unsigned int nt = S.shape[0]

        # fraction of neurotransmitter in 'effective' state (i.e. strength)
        double[:] E = np.empty(nt, np.float64)

        # fraction of neurotransmitter in 'recovered' state (initally all)
        double R = 1.

        # scaled recovery time constant
        double tau = dt / tau_rec

        Py_ssize_t tt

    # deal with any spikes in the first time bin
    # E[0] = S[0] * U
    E[0] = R * (1 - (1 - U) ** S[0])
    R += tau * R - E[0]

    for tt in range(1, nt):

        # (remaining fraction) * (fraction released in this timebin)
        E[tt] = R * (1 - (1 - U) ** S[tt])

        # update the fraction of recovered neurotransmitter
        R += tau * (1 - R) - E[tt]

    return E

# parallelized over all traces
# -----------------------------------------------------------------------------
cpdef double[:, :] binned_tsodyks_parallel(double[:, :] S, double dt,
                                           double tau_rec, double U):

    cdef:
        unsigned int nc = S.shape[0]
        unsigned int nt = S.shape[1]

        double[:, :] E = np.empty((nc, nt), np.float64)

        # R needs to have thread-local values. ugly but it does the job.
        double[:] R = np.ones(nc, np.float64)

        double tau = dt / tau_rec

        Py_ssize_t cc, tt

    for cc in prange(nc, nogil=True, schedule='guided'):

        E[cc, 0] = R[cc] * (1 - (1 - U) ** S[cc, 0])
        R[cc] += tau * (1 - R[cc])

        for tt in range(1, nt):
            E[cc, tt] = R[cc] * (1 - (1 - U) ** S[cc, tt])
            R[cc] += tau * (1 - R[cc]) - E[cc, tt]

    return E


# -----------------------------------------------------------------------------
# Postsynaptic kernels
#------------------------------------------------------------------------------

# for a single trace
#------------------------------------------------------------------------------

cpdef double[:] binned_postsyn(double[:] S, double dt, double tau_m):

    cdef:
        unsigned int nt = S.shape[0]
        double[:] Q = np.empty(nt, np.float64)
        double tau = dt / tau_m
        Py_ssize_t tt

    Q[nt - 1] = S[nt - 1]

    for tt in range(nt - 2, -1, -1):

        if S[tt]:

            # # allow presynaptic spikes to contribute to multiple postsynaptic
            # # spikes within the same time bin
            # Q[tt] = S[tt]

            # truncate Q at 1 - we think that multiple presynaptic spikes will
            # probably contribute to at most one postsynaptic spike per time
            # bin because the membrane potential resets between spikes
            Q[tt] = 1.

        else:

            # we predict that the causal contribution of a presynaptic spike
            # decays exponentially with increasing positive pre -> post time
            # lags because of the membrane time constant
            Q[tt] = Q[tt + 1] - tau * Q[tt + 1]

    return Q


# parallelized over all traces
#------------------------------------------------------------------------------

cpdef double[:, :] binned_postsyn_parallel(double[:, :] S, double dt,
                                           double tau_m):

    cdef:
        unsigned int nc = S.shape[0]
        unsigned int nt = S.shape[1]
        double[:, :] Q = np.empty((nc, nt), np.float64)
        double tau = dt / tau_m
        Py_ssize_t cc, tt

    for cc in prange(nc, nogil=True, schedule='guided'):

        Q[cc, nt - 1] = S[cc, nt - 1]

        for tt in range(nt - 2, -1, -1):

            if S[cc, tt]:
                # Q[cc, tt] = S[cc, tt]
                Q[cc, tt] = 1.
            else:
                Q[cc, tt] = Q[cc, tt + 1] - tau * Q[cc, tt + 1]

    return Q
