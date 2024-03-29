import numpy as np
import os
import tensorflow as tf
from spektral.utils import io, sparse


class SiPMSample:

    def __init__(self, x, a, y):
        self.x = x
        self.y = y
        self.a = a


class ConvSiPM(tf.keras.utils.Sequence):
    def __init__(self,
                 name,
                 batch_size=64,
                 slicing="train",
                 shuffle=False,
                 p_only=False,
                 reg_type=None,
                 seed=42):

        # get current path, go two subdirectories higher
        self.name = name

        # determine train test split
        self.p_train = 0.6
        self.p_test = 0.2
        self.p_valid = 0.2
        self.slicing = slicing

        self.p_only = p_only
        self.reg_type = reg_type

        # read the dataset
        self.samples = self.__read()

        self.entries = len(self.samples)
        self.batch_size = batch_size
        self.steps_per_epoch = int(np.ceil(self.entries / self.batch_size))
        self.shuffle = shuffle

        # shuffling
        np.random.seed(seed)
        self.on_epoch_end()

        # class weighting
        self.class_weights = self.get_classweights()

        # normalization factors
        self.norm_qdc = 4105
        self.norm_tt = 3.125

    @property
    def path(self):
        # get current path, go two subdirectories higher
        path = os.getcwd()
        while True:
            if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
                break
            path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "datasets", "SiFiCCNN_ConvSiPM", self.name)

        return path

    def __read(self):
        # Batch index
        node_batch_index = np.load(self.path + "/" + self.name + "_graph_indicator.npy")
        # nodes == hit sipm
        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # define indexing for train-test splitting
        # needs to be done after one file was opened to determine the length
        entries = len(n_nodes)
        indices = np.arange(0, entries, 1.0, dtype=int)
        ntrain = int(entries * self.p_train)
        nvalid = int(entries * self.p_valid)
        ntest = int(entries * self.p_test)
        if self.slicing == "train":
            indices = indices[:ntrain]
        if self.slicing == "valid":
            indices = indices[ntrain:(ntrain + nvalid)]
        if self.slicing == "test":
            indices = indices[(ntrain + nvalid):]
        # TODO: optimize loading times by not fully processing dataset and then
        # TODO: slicing it

        # Read edge lists
        edges = np.load(self.path + "/" + self.name + "_A.npy")

        # Split edges into separate edge lists
        a_list = np.split(edges, n_nodes_cum[1:])

        # get node attributes (x_list)
        x_list = self._get_x_list(n_nodes_cum=n_nodes_cum)

        # set dataset target (classification / regression)
        y_list = self._get_y_list()
        labels = np.load(self.path + "/" + self.name + "_graph_labels.npy")

        x_list = np.array(x_list)[indices]
        a_list = np.array(a_list)[indices]
        labels = np.array(labels)[indices]

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))

        # limited to True positives only if needed
        if self.p_only:
            # Convert to Graph
            print("Successfully loaded {}.".format(self.name))
            return [
                SiPMSample(x=x, a=a, y=y)
                for x, a, y, label in zip(x_list, a_list, y_list, labels) if label
            ]

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            SiPMSample(x=x, a=a, y=y)
            for x, a, y in zip(x_list, a_list, labels)
        ]

    def _get_x_list(self, n_nodes_cum):
        # Node features
        x_attr = np.load(self.path + "/" + self.name + "_node_attributes.npy")
        x_list = np.split(x_attr, n_nodes_cum[1:])
        return x_list

    def _get_e_list(self, n_edges_cum):
        e_attr = np.load(self.path + "/" + self.name + "_edge_attributes.npy")  # ["arr_0"]
        e_list = np.split(e_attr, n_edges_cum)
        return e_list

    def _get_y_list(self):
        if self.reg_type is not None:
            graph_attributes = np.load(self.path + "/" + self.name + "_graph_attributes.npy")
            if self.reg_type == "Energy":
                y_list = graph_attributes[:, :2]
            elif self.reg_type == "Position":
                y_list = graph_attributes[:, 2:]
            else:
                print("Warning: Regression type not set correctly")
                return None

        else:
            # return class labels
            y_list = np.load(self.path + "/" + self.name + "_graph_labels.npy")
        return y_list

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        # Generate indices of the batch
        batch_indices = self.indices[
                        index * self.batch_size: (index + 1) * self.batch_size]
        return self.__data_generation(batch_indices)

    def on_epoch_end(self):
        self.indices = np.arange(self.entries)
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

        a_batch = [self.samples[i].a for i in batch_indices]
        x_batch = [self.samples[i].x for i in batch_indices]
        f_batch = self.__batch_generation(a_batch=a_batch,
                                          x_batch=x_batch)

        y_batch = np.zeros(shape=(self.batch_size,))
        weights = np.zeros(shape=(self.batch_size,))
        for i in range(len(batch_indices)):
            y_batch[i] = self.samples[batch_indices[i]].y
            weights[i] = self.class_weights[y_batch[i]]

        return f_batch, y_batch, weights

    def __batch_generation(self,
                           x_batch,
                           a_batch,
                           padding=2,
                           gap_padding=4):

        # hardcoded detector size
        dimx = 12
        dimy = 2
        dimz = 32

        norm_qdc = self.norm_qdc
        norm_tt = self.norm_tt

        f_batch = np.ones(shape=(self.batch_size,
                                 dimx + 2 * padding + gap_padding,
                                 dimy + 2 * padding,
                                 dimz + 2 * padding, 2), dtype=np.float32)
        f_batch *= -1.0

        for i in range(len(a_batch)):
            for j in range(len(a_batch[i])):
                x, y, z = a_batch[i][j]
                qdc, triggertime = x_batch[i][j]
                x_final = x + padding if x < 4 else x + padding + gap_padding
                y_final = y + padding
                z_final = z + padding

                f_batch[i, x_final, y_final, z_final, 0] = qdc / norm_qdc
                f_batch[i, x_final, y_final, z_final, 1] = triggertime / norm_tt

        return f_batch

    def get_classweights(self):
        y = [self.samples[i].y for i in range(self.entries)]
        _, counts = np.unique(y, return_counts=True)
        class_weights = {0: len(y) / (2 * counts[0]),
                         1: len(y) / (2 * counts[1])}

        return class_weights

    def y(self):
        return [self.samples[i].y for i in range(self.entries)]
