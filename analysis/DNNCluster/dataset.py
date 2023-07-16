# ##################################################################################################
# dataset.py
#
# Container class to hold the neural network input. The container class is kept simple as the raw
# dataset is small enough to load into RAM.
#
# TODO: data generator would be better
# ##################################################################################################

import numpy as np
import os


class DenseCluster:
    """
    Container class to hold neural network input in the cluster reconstruction configuration.
    Compatible with a dense neural network and with slight modification also for recurrent or
    convolutional networks.

    """

    def __init__(self,
                 name):

        self.name = name

        # load all .npy files defining the dataset
        self.features = np.load(self.path + "/" + self.name + "_sample_features.npy")
        self.labels = np.load(self.path + "/" + self.name + "_sample_labels.npy")
        self.attributes = np.load(self.path + "/" + self.name + "_sample_attributes.npy")
        self.pe = np.load(self.path + "/" + self.name + "_sample_pe.npy")
        self.sp = np.load(self.path + "/" + self.name + "_sample_sp.npy")

        # set primary target, initialized with classification targets. Switch to regression targets
        # is done by calling the "update_targets" methods.
        self.targets = self.labels

        # define train-test-split ratios. Test ratio is kept small on purpose as training and test
        # dataset are different in general.
        self.p_train = 0.7
        self.p_test = 0.1
        self.p_valid = 0.2
        self.entries = len(self.labels)

        # generate shuffled indices with a random generator. Ensures that so systematic is created
        # by a fix data order
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    @property
    def path(self):
        """
        Defines the expected dataset path build from the initialized name

        Returns:
            string: expected path to dataset
        """
        # get current path, go two subdirectories higher
        path = os.getcwd()
        while True:
            if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
                break
            path = os.path.abspath(os.path.join(path, os.pardir))

        path = os.path.join(path, "datasets", "SiFiCCNN_DenseCluster", self.name)

        return path

    def idx_train(self):
        """
        Build list of indices representing every entry belonging to training subset

        Returns:
            ndarray, array containing all indices for the training sample
        """
        return self.ary_idx[0: int(len(self.ary_idx) * self.p_train)]

    def idx_valid(self):
        """
        Build list of indices representing every entry belonging to validation subset

        Returns:
            ndarray, array containing all indices for the validation sample
        """
        return self.ary_idx[
               int(len(self.ary_idx) * self.p_train): int(
                   len(self.ary_idx) * (self.p_train + self.p_valid))]

    def idx_test(self):
        """
        Build list of indices representing every entry belonging to test subset

        Returns:
            ndarray, array containing all indices for the test sample
        """
        return self.ary_idx[
               int(len(self.ary_idx) * (self.p_train + self.p_valid)):]

    def x_train(self):
        """
        Get feature input for network of training subset.

        Returns:
            ndarray, returns features, training subset
        """
        return self.features[self.idx_train()]

    def y_train(self):
        """
        Get target input for network of training subset.

        Returns:
            ndarray, returns targets, training subset
        """
        return self.targets[self.idx_train()]

    def x_valid(self):
        """
        Get feature input for network of validation subset.

        Returns:
            ndarray, returns features, validation subset
        """
        return self.features[self.idx_valid()]

    def y_valid(self):
        """
        Get target input for network of validation subset.

        Returns:
            ndarray, returns targets, validation subset
        """
        return self.targets[self.idx_valid()]

    def x_test(self):
        """
        Get feature input for network of test subset.

        Returns:
            ndarray, returns features, test subset
        """
        return self.features[self.idx_test()]

    def y_test(self):
        """
        Get target input for network of test subset.

        Returns:
            ndarray, returns targets, test subset
        """
        return self.targets[self.idx_test()]

    def update_indexing_positives(self):
        """
        Updates the ary_idx attribute to only contain indices of true positive events. Used to
        filter for true positive events for training regression networks

        Returns:
            None
        """
        ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        # grab indices of all positives events
        self.ary_idx = ary_idx[self.labels == 1.0]

        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_indexing_all(self):
        """
        Updates the ary_idx attribute to contain all indices again. Used to reset indices list.
        Reshuffles indices list if called.

        Returns:
            None
        """
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_indexing_ordered(self):
        """
        Updates the ary_idx attribute to contain all indices in order.

        Returns:
            None
        """
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)

    def update_indexing_s1ax(self):
        """
        Updates the ary_idx attribute to contain all indices for events with only one scatterer
        cluster.

        Returns:
            None
        """
        list_idx = []
        for i in range(self.entries):
            # check if attribute of second timestamp is zero == second cluster is not filled
            if self.features[i, 1, 0] == 0.0:
                list_idx.append(i)
        self.ary_idx = np.array(list_idx)

    def update_indexing_s1ax_positive(self):
        """
        Updates the ary_idx attribute to contain all indices for events with only one scatterer
        cluster.

        Returns:
            None
        """
        list_idx = []
        for i in range(self.entries):
            # check if attribute of second timestamp is zero == second cluster is not filled
            if self.features[i, 1, 0] == 0.0 and self.labels[i] == 1:
                list_idx.append(i)
        self.ary_idx = np.array(list_idx)

    def update_targets_clas(self):
        """
        Sets targets attribute to classification targets

        Returns:
            None
        """
        # set legacy targets to be module energies
        self.targets = self.labels

    def update_targets_energy(self):
        """
        Sets targets attribute to energy regression targets

        Returns:
            None
        """
        # set legacy targets to be module energies
        self.targets = self.attributes[:, :2]

    def update_targets_position(self):
        """
        Sets targets attribute to position regression targets

        Returns:
            None
        """
        # set legacy targets to be module energies
        self.targets = self.attributes[:, 2:]

    def get_classweights(self):
        """
        Generate a dictionary containing class weights used to rebalanced classification training.

        Returns:
            dict, dictionary containing class weights
        """
        # set sample weights to class weights
        _, counts = np.unique(self.labels[self.ary_idx], return_counts=True)
        class_weights = {0: len(self.labels[self.ary_idx]) / (2 * counts[0]),
                         1: len(self.labels[self.ary_idx]) / (2 * counts[1])}

        return class_weights

    def num_features(self):
        """
        Grabs the number of features in the dataset. Used to build the input dimension of a
        neural network.

        Returns:
            int, number of features in dataset
        """
        return self.features.shape[1]

    def get_standardization(self, n_cluster=10, n_features=10):
        """
        Calculate feature wise standardization. Features are expected as (n_cluster, n_feature)
        dimension. Standardization normalizes the distribution of every feature to mean 0 and
        standard deviation 1. Formula used:

                x_i = (x_i - mean(x)) / std(x)

        Returns:
            ndarray, array containing mean and standard deviation of every feature
        """
        norm = np.zeros(shape=(n_features, 2))

        for i in range(n_features):
            con = np.reshape(self.features[:, :, i],
                             (self.features.shape[0] * n_cluster,))
            con = con[con != 0.0]
            norm[i, :] = [np.mean(con), np.std(con)]

        return norm

    def standardize(self, norm, n_features=10):
        """
        Standardizes the stored datasets with the given norm. Norm is created by the
        "get_standardization" method.

        Args:
            norm: ndarray, containing the mean and standard deviation of every feature
            n_features: number of features

        Returns:
            None
        """
        for i in range(n_features):
            self.features[:, :, i] = (self.features[:, :, i] - norm[i, 0]) / norm[i, 1]

    def normalize(self, n_features=10):

        ary_p0 = np.array([0, 0, 0, 210, 0, 0, 0, 0, 0, 0])
        ary_p1 = np.array([2, 5, 3, 60, 50, 50, 3, 3, 16, 3])

        for i in range(n_features):
            self.features[:, :, i] = (self.features[:, :, i] - ary_p0[i]) / ary_p1[i]
