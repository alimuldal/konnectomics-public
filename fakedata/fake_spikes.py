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

import os
import numpy as np
# import matplotlib.pyplot as pp
import nest
# import nest.raster_plot
import time
import datetime

from multiprocessing import cpu_count

from fakedata import fake_network
from utils.waitbar import s2h


# -----------------------------------------------------------------------------
# defaults are defined in the constants at the top of this file
#
# *** don't hard-code your changes anywhere else! ***
# -----------------------------------------------------------------------------

# default params for simulation
RESOLUTION = 0.1            # simulation timestep, ms
NOISE_WEIGHT = 4.           # external synaptic weight, pA
NOISE_RATE = 1.6            # rate of external inputs, Hz

# auto weight adjustment
BURSTRATE_TARGET = 0.1      # Hz
BURSTRATE_TOL = 0.01        # Hz
JE_WEIGHT_INIT = 5.         # internal synaptic weight, initial value, pA
SIMTIME_ADAPT = 200 * 1000.  # time: s * ms
MAXITER_ADAPT = 100

# actual simulation
SIMTIME_RUN = 1. * 60. * 60. * 1000.      # time: h * min * s * ms

# -----------------------------------------------------------------------------


def adjacency2netobj(adjacency_matrix):
    ncells = adjacency_matrix.shape[0]
    ncons = adjacency_matrix.sum()

    nodes = []
    for ii, row in enumerate(adjacency_matrix):
        presyn_index = ii
        postsyn_indices = np.where(row)[0]
        nodes.append(dict(id=presyn_index, connectedTo=postsyn_indices))

    net_object = dict(ncells=ncells, ncons=ncons, nodes=nodes)
    return net_object


def determine_burst_rate(xtimes, total_timeMS, ncells, avgwinMS=50,
                         burst_threshold=0.4):
    """
    Sliding window approach to determine intervals when population burst; count
    burst onsets
    """
    total_spikes_per_bin = np.zeros(total_timeMS)

    tmp = np.bincount(xtimes.astype(np.int))

    total_spikes_per_bin[:len(tmp)] = tmp
    window_summed_vec = np.convolve(
        total_spikes_per_bin, np.ones(avgwinMS), 'valid')

    burst_times = window_summed_vec >= burst_threshold * ncells

    # detect burst onsets
    n_bursts = np.sum(np.diff(burst_times.astype(np.int)) == 1)

    return n_bursts * 1000. / total_timeMS


def create_network(network_obj, weight, JENoise, noise_rate, resolution=0.1,
                   verbose=True, print_time=False):

    ncells = network_obj['ncells']
    ncons = network_obj['ncons']

    if verbose:
        print "Constructing NEST network of %i nodes and %i connections." % (
            ncells, ncons)

    nest.ResetKernel()

    nthreads = cpu_count()

    if verbose:
        nest.set_verbosity('M_INFO')
    else:
        nest.set_verbosity('M_ERROR')

    nest.SetKernelStatus(dict(local_num_threads=nthreads, resolution=0.1,
                              print_time=print_time, overwrite_files=True))

    neuron_params = dict(C_m=1.0, tau_m=20.0, t_ref=2.0, E_L=0.0, V_th=20.0)
    nest.SetDefaults("iaf_neuron", neuron_params)
    neuronsE = nest.Create("iaf_neuron", n=ncells)

    # save GID offset of first neuron - this has the advantage that the output
    # later will be independent of the point at which the neurons were created
    GIDoffset = neuronsE[0]

    espikes = nest.Create("spike_detector")
    nest.ConvergentConnect(neuronsE, espikes)

    noise = nest.Create("poisson_generator", n=1, params=dict(rate=noise_rate))

    # Warning: delay is overwritten later if weights are given!
    nest.SetDefaults("tsodyks_synapse",
                     dict(delay=1.5, tau_rec=500., tau_fac=0., U=0.3))
    nest.CopyModel("tsodyks_synapse", "exc", dict(weight=weight))
    nest.CopyModel("static_synapse", "poisson", dict(weight=JENoise))

    # every neuron gets the same noisy input???
    nest.DivergentConnect(noise, neuronsE, model="poisson")

    for node in network_obj['nodes']:

        presyn_index = node['id']
        postsyn_indices = node['connectedTo']

        nest.DivergentConnect(
            [neuronsE[presyn_index]],                   # from, list of len 1
            [neuronsE[ii] for ii in postsyn_indices],   # to, list
            model='exc',                                # synapse model
        )

    return ncells, ncons, neuronsE, espikes, noise, GIDoffset


def resample_spikes(spike_times, cell_indices, output_resolution=20,
                    simtime=None, ncells=None):

    if ncells is None:
        ncells = cell_indices.max()
    if simtime is None:
        simtime = spike_times.max()

    if np.max(cell_indices) > (ncells - 1):
        raise ValueError('cell index %i out of bounds for %i cells'
                         % (np.max(cell_indices), ncells))
    if np.max(spike_times) > simtime:
        raise ValueError('spike time %g out iof bounds for simtime %g'
                         % (np.max(spike_times), simtime))

    nt_out = int(simtime / output_resolution)

    # uint8 would almost certainly be enough, unless the cells are very spiky
    # and the output bins are very wide
    out = np.zeros((ncells, nt_out), dtype=np.uint32)

    for si in xrange(spike_times.size):
        cells = cell_indices[si]
        timebin = int(spike_times[si] / output_resolution)
        out[cells, timebin] += 1

    return out


def adjust_weight(adjacency_matrix=None,
                  burstrate_target=BURSTRATE_TARGET,
                  burstrate_tol=BURSTRATE_TOL,
                  weight=JE_WEIGHT_INIT,
                  noise_weight=NOISE_WEIGHT,
                  noise_rate=NOISE_RATE,
                  resolution=RESOLUTION,
                  simtime=SIMTIME_ADAPT,
                  maxiter_adapt=MAXITER_ADAPT,
                  verbose=True,
                  print_time=False,
                  ):

    if adjacency_matrix is None:
        # construct a default network
        adjacency_matrix = fake_network.construct_network()

    # convert into network_object
    network_obj = adjacency2netobj(adjacency_matrix)

    adaptParList = []
    burst_rate = -1
    adaptation_iteration = 1
    last_burst_rates = []
    last_JEs = []

    if verbose:
        print "Starting adaptation phase..."

    while abs(burst_rate - burstrate_target) > burstrate_tol:

        if (len(last_burst_rates) < 2
                or last_burst_rates[-1] == last_burst_rates[-2]):

            if len(last_burst_rates) > 0:
                if verbose:
                    print 'Auto-burst stage II. - changing weight by 10%'
                if burst_rate > burstrate_target:
                    weight *= 0.9
                else:
                    weight *= 1.1
            else:
                if verbose:
                    print 'Auto-burst stage I. - initial run'
        else:
            if verbose:
                print 'Auto-burst stage III. - linear extrapolation'
            weight = (((burstrate_target - last_burst_rates[-2])
                       * (last_JEs[-1] - last_JEs[-2])
                       / (last_burst_rates[-1] - last_burst_rates[-2]))
                      + last_JEs[-2])
        assert weight > 0.

        if verbose:
            print "adaptation %i, setting weight to %g ..." % (
                adaptation_iteration, weight)
            print 'Setting up network...'
        ncells, ncons, neuronsE, espikes, noise, GIDoffset = create_network(
            network_obj, weight, noise_weight, noise_rate,
            resolution=resolution, verbose=verbose, print_time=print_time,
        )
        if verbose:
            print 'Simulating...'
        nest.Simulate(simtime)
        if verbose:
            print 'Calculating the burst rate...'
        spike_times = nest.GetStatus(espikes, "events")[0]["times"]
        burst_rate = determine_burst_rate(spike_times, simtime, ncells)
        if verbose:
            print "-> the burst rate is %g Hz" % burst_rate
        adaptation_iteration += 1
        last_burst_rates.append(burst_rate)
        last_JEs.append(weight)
        assert adaptation_iteration < maxiter_adapt

    return weight, burst_rate


def run_simulation(adjacency_matrix=None,
                   weight=None,
                   noise_rate=NOISE_RATE,
                   noise_weight=NOISE_WEIGHT,
                   resolution=RESOLUTION,
                   simtime=SIMTIME_RUN,
                   save=False, output_path='data/', basename='nest_sim_',
                   overwrite=False, verbose=True, print_time=False):

    if adjacency_matrix is None:
        # construct a network according to defaults
        adjacency_matrix = fake_network.construct_network()

    if weight is None:
        # if unspecified, find the weight automatically according to defaults
        weight, _ = adjust_weight(adjacency_matrix, noise_weight=noise_weight,
                                  noise_rate=noise_rate, resolution=resolution,
                                  verbose=verbose, print_time=print_time)

    # convert into network_object
    network_obj = adjacency2netobj(adjacency_matrix)

    ncells, ncons, neuronsE, espikes, noise, GIDoffset = create_network(
        network_obj, weight, noise_weight, noise_rate, resolution=resolution,
        verbose=verbose, print_time=print_time,
    )

    if verbose:
        print 'Simulating %s of activity for %i neurons' % (
            s2h(simtime / 1000.), ncells)

    startsimulate = time.time()
    nest.Simulate(simtime)
    endsimulate = time.time()

    sim_elapsed = endsimulate - startsimulate

    totalspikes = nest.GetStatus(espikes, "n_events")[0]
    events = nest.GetStatus(espikes, "events")[0]

    # NEST increments the GID whenever a new node is created. we therefore
    # subtract the GID offset, so that the output cell indices are correct
    # regardless of when the corresponding nodes were created in NEST
    cell_indices = events["senders"] - GIDoffset
    spike_times = events["times"]

    burst_rate = determine_burst_rate(spike_times, simtime, ncells)

    if verbose:
        print "\n" + "-" * 60
        print "Number of neurons: ", ncells
        print "Number of spikes recorded: ", totalspikes
        print "Avg. spike rate of neurons: %.2f Hz" % (
            totalspikes / (ncells * simtime / 1000.))
        print "Network burst rate: %.2f Hz" % burst_rate
        print "Simulation time: %s" % s2h(sim_elapsed)
        print "-" * 60

    # resample at 50Hz to make an [ncells, ntimesteps] array of bin counts
    resampled = resample_spikes(spike_times, cell_indices,
                                output_resolution=20, simtime=simtime,
                                ncells=ncells)

    if save:

        today = str(datetime.date.today())
        fname = basename + today
        if os.path.exists(fname) & ~overwrite:
            suffix = 0
            while os.path.exists(fname):
                suffix += 1
                fname = '%s%s_%i.npz' % (basename, today, suffix)
        fullpath = os.path.join(output_path, fname)
        np.savez(fullpath, spike_times=spike_times, cell_indices=cell_indices,
                 resampled=resampled)

        if verbose:
            print "Saved output in '%s'" % fullpath

    return spike_times, cell_indices, resampled

if __name__ == "__main__":
    run_simulation()
