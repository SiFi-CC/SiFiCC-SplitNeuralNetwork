import numpy as np
import tensorflow as tf

from spektral.data.loaders import Loader
from SiFiCCNN.data.utils import batch_generator


class DenseLoader:

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        """
        Returns lists (batches) of `Graph` objects.
        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        """
        Converts a list of graph objects to Tensors or np.arrays representing
        the batch.
        :param batch: a list of `Graph` objects.
        """
        return batch

    def load(self):
        """
        Returns an object that can be passed to a Keras model when using the
        `fit`,
        `predict` and `evaluate` functions.
        By default, returns the Loader itself, which is a generator.
        """
        return self

    @property
    def steps_per_epoch(self):
        """
        :return: the number of batches of size `self.batch_size` in the dataset
        (i.e., how many batches are in an epoch).
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))
