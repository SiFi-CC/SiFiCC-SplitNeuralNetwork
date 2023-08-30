import os
import tensorflow as tf
import numpy as np
import sklearn as sk
import scipy as scp
import spektral as spk


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self,
                 datasetPath,
                 slicing=(...,),
                 map_fnc=None,
                 batch_size=64,
                 shuffle=False,
                 mmap_mode="r",
                 seed=1234):
        r""" Generator class which provides batches of event images and corresponding true labels/vertices.

        Args:
            imagePath: Path of event images file with PMT data (numpy binary). Read as memmap
            labelPath: Path of output labels (numpy binary). Read as memmap
            slicing: tuple of slicing objects. For using only part of the dataset. e.g. only limited amount of data
            map_fnc: function which is applied to the event images
            batch_size: int. size of mini batch
            weights: array of weights for each sample
            shuffle: bool. if True dataset is shuffled after each epoch
            seed: random seed
        """

        # Load event images and true vertices as memory mapped files
        self.images = np.load(imagePath, mmap_mode=mmap_mode)[slicing]
        self.param = np.load(labelPath, mmap_mode=mmap_mode)[
            slicing[0] if isinstance(slicing, list) else slicing]

        if map_fnc is None:
            self.map_fnc = lambda x: x
        else:
            self.map_fnc = map_fnc

        self.NumberOfEvents = self.images.shape[0]
        print("Number of samples: {}".format(self.NumberOfEvents))

        self.batch_size = batch_size
        self.steps_per_epoch = int(np.ceil(self.NumberOfEvents / self.batch_size))
        self.shuffle = shuffle
        self.weights = weights

        np.random.seed(seed)
        self.on_epoch_end()

    def __len__(self):
        r"""Denotes the number of batches per epoch."""
        return self.steps_per_epoch

    def __getitem__(self, index):
        r"""Generate batch of data which corresponds to 'index'.
        Args:
            index: the index of a certain batch of the dataset
        Returns:
            mini batch with index 'index' of training data true lable
        """
        # Generate indices of the batch
        batch_indices = self.indices[index * self.batch_size: (index + 1) * self.batch_size]
        return self.__data_generation(batch_indices)

    def on_epoch_end(self):
        r"""Updates indices after each epoch"""
        self.indices = np.arange(self.NumberOfEvents)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_indices):
        r"""Generates data containing batch_size samples.

        Args:
            batch_indices: Indices of samples which shall be filled in the mini batch.
        Returns:
            X_batch: array with batch_size length first dimension of training samples.
            Y_batch: array with batch_size length first dimension of corresponding true labels
            weights_batch (optional): array of size batch_size of sample weights
        """
        X_batch = self.map_fnc(self.images[batch_indices])
        Y_batch = self.param[batch_indices]

        if not (self.weights is None):
            weights_batch = self.weights[batch_indices]
            return X_batch, Y_batch, weights_batch
        else:
            return X_batch, Y_batch
