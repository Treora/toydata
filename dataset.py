"""
Provides functions to generate toy datasets, wrapped by the class Dataset to
provide easy ways to obtain single layers from the dataset.
"""

import numpy as np

from toydata import toydata
from helpers import argrowsort


def generate_dataset(N):
    """Generate a toy data set with N samples."""
    samples = toydata(n_samples=N)
    return Dataset(samples)


def generate_uncorrelated_dataset(N):
    """Generate a toy dataset with equal co-occurence among the features"""
    features = ['000001', '000010', '000100', '001000', '010000', '100000']
    samples = []
    for _ in range(N):
        # Choose two or three features
        f1 = np.random.choice(features)
        f2 = np.random.choice(features)
        f3 = np.random.choice(features)
        if np.random.choice([True, False]):
            # Use only two features
            f3 = '0'
        # Perform binary OR on the strings
        f = format(int(f1, 2) | int(f2, 2) | int(f3, 2), "06b")
        samples += toydata(latent_vars=['0000', f], n_samples=1)
    return Dataset(samples)


def generate_single_feature_dataset(N):
    """Generate a toy dataset with only one active feature in each sample"""
    features = ['000001', '000010', '000100', '001000', '010000', '100000']
    samples = []
    for _ in range(N):
        # Choose one feature
        f = np.random.choice(features)
        samples += toydata(latent_vars=['0000', f], n_samples=1)
    return Dataset(samples)


class Dataset:
    """A container for data samples having layers of binary data.
       Basically, it can be regarded a 3d array (samples, layers, bits), but
       layers can have different amounts of bits, unlike numpy's ndarray.
    """
    def __init__(self, data):
        if type(data) is np.ndarray and data.ndim==2 and data.dtype is np.dtype('object'):
            # Assume data is (part of) another Dataset's data
            self.data = data
        elif type(data) in (list, tuple) and all([type(item) is np.ndarray for item in data]):
            # Assume data is list of 2d arrays, each representing one layer's activations
            n_samples = len(data[0])
            if not all([len(layer)==n_samples for layer in data]):
                raise ValueError, "Given layers have different amounts of samples"
            self.data = np.array([[layer[i] for layer in data] for i in range(n_samples)])
        else:
            # Assume data is a sequence of samples as generated by toydata
            # Turn list of lists of strings into 2d array of arrays.
            samples = np.array([
                [self._bitstring_to_array(bitstring) for bitstring in sample]
                for sample in data
            ])
            self.data = samples[:,::-1] # mirror horizontally, visible data first

        self.n_samples = self.data.shape[0]
        self.n_layers = self.data.shape[1]
        self.layer_sizes = [len(self.data[0,i]) for i in range(self.n_layers)]

    def get_layer(self, index):
        """Get the specified layer as a 2d array (samples, bits)."""
        return self._to_2d_array(self.data[:,index])

    def get_layers(self):
        """Equivalent to [get_layer(0), get_layer(1), ...]."""
        return [self.get_layer(i) for i in range(self.n_layers)]

    def sort_using_layer(self, index, reverse=False):
        """Get the dataset with samples sorted by the specfied layer."""
        layer = self.get_layer(index)
        ordering = argrowsort(layer)
        if reverse:
            ordering = ordering[::-1]
        sorted_data = self.data[ordering]
        return Dataset(sorted_data)

    @staticmethod
    def _to_2d_array(array_of_arrays):
        # Ugly workaround to turn an array of arrays into a 2d array.
        return np.array(list(array_of_arrays))

    @staticmethod
    def _bitstring_to_array(bitstring):
        """Turn a string like '10101' into an array([1,0,1,0,1])."""
        return np.array([int(bit) for bit in bitstring])


class TestDataset:
    def setup(self):
        self.n_samples = 10
        self.samples = toydata(n_samples=self.n_samples)
        self.n_layers = len(self.samples[0])

    def test_init_from_samples(self):
        d1 = Dataset(self.samples)
        assert d1.data.shape == (self.n_samples, self.n_layers)
        # Check contents of visible data of first sample
        assert all([str(bit1)==bit2 for bit1,bit2 in zip(d1.data[0][0], self.samples[0][-1])])

    def test_init_from_own_data(self):
        d1 = Dataset(self.samples)
        double_data = np.vstack((d1.data, d1.data))
        d2 = Dataset(double_data)
        assert self._equal_data(d2.data, double_data)

    def test_init_from_layers(self):
        d1 = Dataset(self.samples)
        layers = [d1.get_layer(i) for i in range(self.n_layers)]
        d2 = Dataset(layers)
        assert self._equal_data(d1.data, d2.data)

    def test_sort_using_layer(self):
        d1 = Dataset([['10','001'],['01','110'],['11','000']])
        sorted_by_h = Dataset([['01','110'],['10','001'],['11','000']])
        sorted_by_v = Dataset([['11','000'],['10','001'],['01','110']])
        sorted_by_h_reverse = Dataset([['11','000'],['10','001'],['01','110']])
        d2 = d1.sort_using_layer(-1) # sort by deepest hidden layer
        d3 = d2.sort_using_layer(0) # sort by visible layer
        d4 = d1.sort_using_layer(-1, reverse=True)

        assert self._equal_data(d2.data, sorted_by_h.data)
        assert self._equal_data(d3.data, sorted_by_v.data)
        assert self._equal_data(d2.sort_using_layer(-1).data, d2.data)
        assert self._equal_data(d4.data, sorted_by_h_reverse.data)

    def _equal_data(self, data1, data2):
        if data1 is data2 or len(data1) == len(data2) == 0:
            return True
        if not data1.shape == data2.shape:
            return False
        # Deep equality test
        from itertools import product
        index_pairs = product(range(len(data1)), range(len(data1[0])))
        if not all([np.array_equal(data1[i,j], data2[i,j]) for i,j in index_pairs]):
            return False
        return True
