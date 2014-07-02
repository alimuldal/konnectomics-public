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


from fakedata import fake_network, fake_spikes, fake_ca, fake_blur
from utils import submission
import tables

for mod in (fake_network, fake_spikes, fake_ca, fake_blur, submission):
    reload(mod)

# HDF5 compression options
FILTERS = tables.Filters(complevel=5, complib='blosc')


def run(f, name, network=None, network_pos=None, ncells=None, sim_hours=1,
        overwrite=False):

    try:

        try:
            print ">> Creating new group: '/%s'" % name
            g = f.create_group(f.root, name)

        except tables.NodeError as e:
            if overwrite:
                # recursively remove existing group
                f.remove_node('/' + name, recursive=True)
                g = f.create_group(f.root, name)
            else:
                raise e

        if network is None:
            print ">> Creating new connectivity matrix"

            # create an adjacency matrix for ncells, with default params
            adjacency = fake_network.create_network(ncells)

            # reshape the adjacency matrix into a vector form that is
            # compatible with submission.run_auc()
            ij, connected = submission.adjacency2vec(adjacency)
            network = np.vstack((ij, connected)).T
            del ij, connected

            f.create_carray(g, 'network', obj=network, filters=FILTERS)
            f.flush()

        else:
            # convert existing network to an adjacency matrix
            ij, connected = submission.real2dense(network, ncells)
            adjacency = submission.vec2adjacency(ij, connected)
            f.create_carray(g, 'network', obj=network[:], filters=FILTERS)

        if network_pos is None:
            print ">> Generating fake cell positions"
            network_pos = fake_blur.fake_positions(ncells)
            f.create_carray(g, 'network_pos', obj=network_pos, filters=FILTERS)
            f.flush()

        else:
            f.create_carray(g, 'network_pos', obj=network_pos[:],
                            filters=FILTERS)

        print ">> Running spiking network simulation"
        # run a NEST simulation using this adjacency matrix. the synaptic
        # scaling parameter will be adjusted automatically to achieve a target
        # burst rate according to default params
        simtime = sim_hours * 60 * 60 * 1000    # time in ms
        (spike_times, cell_indices,
            resampled_spikes) = fake_spikes.run_simulation(adjacency,
                                                           simtime=simtime,
                                                           verbose=True)

        f.create_carray(g, 'spike_times', obj=spike_times, filters=FILTERS)
        f.create_carray(g, 'spike_cell_indices', obj=cell_indices,
                        filters=FILTERS)
        f.create_carray(g, 'resampled_spikes', obj=resampled_spikes,
                        filters=FILTERS)
        f.flush()
        del spike_times, cell_indices

        print ">> Modelling calcium dynamics"
        calcium = fake_ca.fast_ca_from_spikes(resampled_spikes)
        f.create_carray(g, 'calcium', obj=calcium, filters=FILTERS)
        f.flush()
        del resampled_spikes

        print ">> Modelling dye saturation"
        no_noise_fluor = fake_ca.fluor_from_ca(calcium)
        f.create_carray(g, 'no_noise_fluor', obj=no_noise_fluor,
                        filters=FILTERS)
        f.flush()
        del calcium

        print ">> Modelling fluorescence noise"
        noisy_fluor = fake_ca.noisy_from_no_noise(no_noise_fluor)
        f.create_carray(g, 'noisy_fluor', obj=noisy_fluor, filters=FILTERS)
        f.flush()
        del no_noise_fluor

        print ">> Modelling optical blurring"
        fluor = fake_blur.apply_gaussian_blur(network_pos, noisy_fluor)
        f.create_carray(g, 'fluorescence', obj=fluor, filters=FILTERS)
        f.flush()

        print ">> Done."

    except:

        import traceback
        # print the traceback for the exception
        traceback.print_exc()

        # make sure the file is closed, otherwise we won't have access
        # to the object and we won't be able to close it without
        # restarting ipython. close() calls flush() first, so no need to
        # call it explicitly.
        print 'Closing file "%s"' % f.filename
        f.close()
