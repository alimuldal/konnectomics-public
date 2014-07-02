#! /usr/bin/env python

# Copyright 2012, Olav Stetter
#
# This file is part of TE-Causality.
#
# TE-Causality is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TE-Causality is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TE-Causality.  If not, see <http://www.gnu.org/licenses/>.

# NEST simulator designed to iterate over a number of input topologies
# and to adjst the internal synaptic weight to always achieve an
# equal bursting rate across networks.

import numpy as np
import igraph       # igraph is sloooooow. better to use graph_tool.


# -----------------------------------------------------------------------------
# defaults are defined in the constants at the top of this file
#
# *** don't hard-code your changes anywhere else! ***
# -----------------------------------------------------------------------------

# default params for network construction

N_NODES = 100       # number of neurons to model
DENSITY = 0.3       # overall connection probability
GLOBAL_CC = 0.5     # global clustering coefficient

# -----------------------------------------------------------------------------

def construct_network(n_nodes=N_NODES, density=DENSITY, global_cc=GLOBAL_CC,
                      tol=1E-3, verbose=False):
    """
    Creates a network with the specified connection density. The initial
    network is then rewired to achieve a specified global clustering
    coefficient.

    Arguments:
    -----------
        n_nodes: int [0, inf]
            number of nodes
        density: float [0, 1]
            probability of a connection from node i to j
        global_cc: float [0, 1]
            target gcc; the target gcc > density, otherwise slow to converge
        tol: float
            tolerance for desired gcc value
    """

    g = igraph.Graph.Erdos_Renyi(
        n=n_nodes, p=density, directed=True, loops=False)

    # global clustering coefficients bounded by [0,1]
    diff_0 = 1.
    gcc_0 = np.nan

    rewiring_rate = len(g.es.count_multiple())  # n edges
    successful_rrs = 3 * [rewiring_rate]

    if verbose:
        print 'Rewiring network to achieve specified clustering coefficient...'

    while diff_0 > tol:

        g_0 = g.copy()
        n_trials = np.maximum(int(rewiring_rate * diff_0), 1)
        g.rewire(n=n_trials)
        gcc = g.transitivity_undirected()
        diff = np.abs(gcc - global_cc)

        if diff_0 > diff:
            diff_0 = diff
            gcc_0 = gcc
            successful_rrs.append(rewiring_rate)
            rewiring_rate = 1.1 * np.mean(successful_rrs[-3:])

            if verbose:
                print ("diff=%.4g\tCC got better\tnew rewiring rate=%.1f"
                       % (diff, rewiring_rate))

        else:
            g = g_0  # reset to old graph
            rewiring_rate *= 0.9  # decrease step size
            if verbose:
                print ("diff=%.4g\tCC got worse\tnew rewiring rate=%.1f"
                       % (diff, rewiring_rate))

    adjacency_matrix = np.array(g.get_adjacency().data)

    return adjacency_matrix
