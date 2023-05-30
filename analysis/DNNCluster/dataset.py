import numpy as np
import os


class DenseCluster:

    def __init__(self,
                 name):

        self.name = name

        self.features = np.load(self.path + "/" + self.name + "_sample_features.npy")
        self.labels = np.load(self.path + "/" + self.name + "_sample_labels.npy")
        self.attributes = np.load(self.path + "/" + self.name + "_sample_attributes.npy")
        self.pe = np.load(self.path + "/" + self.name + "_sample_pe.npy")
        self.sp = np.load(self.path + "/" + self.name + "_sample_sp.npy")

        # set primary target, initialized with classification targets
        self.targets = self.labels

        # Train-Test-Valid split
        self.p_train = 0.7
        self.p_test = 0.1
        self.p_valid = 0.2
        self.entries = len(self.labels)

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    @property
    def path(self):
        # get current path, go two subdirectories higher
        path = os.getcwd()
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "datasets", "SiFiCCNN_DenseCluster", self.name)

        return path

    def idx_train(self):
        return self.ary_idx[0: int(len(self.ary_idx) * self.p_train)]

    def idx_valid(self):
        return self.ary_idx[
               int(len(self.ary_idx) * self.p_train): int(
                   len(self.ary_idx) * (self.p_train + self.p_valid))]

    def idx_test(self):
        return self.ary_idx[
               int(len(self.ary_idx) * (self.p_train + self.p_valid)):]

    def x_train(self):
        return self.features[self.idx_train()]

    def y_train(self):
        return self.targets[self.idx_train()]

    def x_valid(self):
        return self.features[self.idx_valid()]

    def y_valid(self):
        return self.targets[self.idx_valid()]

    def x_test(self):
        return self.features[self.idx_test()]

    def y_test(self):
        return self.targets[self.idx_test()]

    def update_indexing_positives(self):
        ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        # grab indices of all positives events
        self.ary_idx = ary_idx[self.labels == 1.0]

        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_indexing_all(self):
        ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_targets_clas(self):
        # set legacy targets to be module energies
        self.targets = self.labels

    def update_targets_energy(self):
        # set legacy targets to be module energies
        self.targets = self.attributes[:, :2]

    def update_targets_position(self):
        # set legacy targets to be module energies
        self.targets = self.attributes[:, 2:]

    def get_classweights(self):
        # set sample weights to class weights
        _, counts = np.unique(self.labels, return_counts=True)
        class_weights = {0: len(self.labels) / (2 * counts[0]),
                         1: len(self.labels) / (2 * counts[1])}

        return class_weights

    def num_features(self):
        return self.features.shape[1]

    def get_standardization(self, n_cluster=10, n_features=10):
        """Calculate feature wise standardization.

        Features are expected as (n_cluster, n_feature) dimension.


        """
        norm = np.zeros(shape=(n_features, 2))

        for i in range(n_features):
            con = np.reshape(self.features[:, :, i],
                             (self.features.shape[0] * n_cluster,))
            con = con[con != 0.0]
            norm[i, :] = [np.mean(con), np.std(con)]

        return norm

    def standardize(self, norm, n_features=10):
        for i in range(n_features):
            self.features[:, :, i] = (self.features[:, :, i] - norm[i, 0]) / norm[i, 1]
