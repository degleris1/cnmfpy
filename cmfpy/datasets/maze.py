import numpy as np
from scipy.io import loadmat
import h5py
import os

DATAPATH = os.path.expanduser(
    "~/cmf_data/NoveltySessInfoMatFiles/Achilles_10252013_sessInfo.mat")


class Maze:
    """
    Silicon-Probe neural recordings from rats before, during, and after
    a maze-running task.

    Reference
    ---------
    https://crcns.org/data-sets/hc/hc-11
    """

    def __init__(self,
                 normalize=True,
                 path=DATAPATH,
                 start_time=0,
                 end_time=200,
                 bin_time=1e-3):

        self.name = "maze"
        self.normalize = normalize
        self.path = path
        self.start_time = start_time
        self.end_time = end_time
        self.bin_time = bin_time

    def generate(self):
        f = h5py.File(self.path, 'r')
        spike_ids = f["sessInfo"]["Spikes"]["SpikeIDs"][0]
        spike_times = f["sessInfo"]["Spikes"]["SpikeTimes"][0]

        # Only a few neurons have spikes in our data, so we remove
        # unneeded neurons by forming a map from spike_id -> neuron
        id_map = {}
        for i, neuron in enumerate(np.unique(spike_ids)):
            id_map[neuron] = i
        neuron_assignments = [id_map[x] for x in spike_ids]
        num_neurons = int(max(neuron_assignments) + 1)

        # Reject spikes outside of our time window
        # An end time of -1 corresponds to using all data
        if (self.end_time == -1):
            self.end_time = spike_ids[-1]
        neuron_assignments = np.array(neuron_assignments)
        spike_times = np.array(spike_times)
        spike_idx = (spike_times >= self.start_time) & (spike_times <= self.end_time)
        neuron_assignments = neuron_assignments[spike_idx]
        spike_times = spike_times[spike_idx]

        num_bins = int((self.end_time - self.start_time) / self.bin_time)+1
        data = np.zeros((num_neurons, num_bins))
        spike_times_binned = np.round((spike_times - self.start_time) / (self.bin_time)).astype(int)

        # Create array of spike indices, and use np.unique to find
        # the number of times each index occurs. This corresponds to
        # the number of spikes in a given time bin.
        spike_times_binned = np.expand_dims(spike_times_binned, 1)
        neuron_assignments = np.expand_dims(neuron_assignments, 1)
        spike_indices = np.hstack((neuron_assignments, spike_times_binned))
        (unique_indices, counts) = np.unique(spike_indices, axis=0, return_counts=True)

        data[unique_indices[:, 0], unique_indices[:, 1]] = counts

        if self.normalize:
            data /= 1e-8 + np.linalg.norm(data, ord=1, axis=1, keepdims=True)
        return data
