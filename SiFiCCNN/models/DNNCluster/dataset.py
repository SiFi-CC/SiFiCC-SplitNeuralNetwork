import numpy as np
import os


class DenseCluster:

    def __init__(self,
                 name):

        # get current path, go two subdirectories higher
        path = os.path.dirname(os.path.abspath(__file__))
        for i in range(3):
            path = os.path.dirname(path)
        path = os.path.join(path, "datasets", "SiFiCCNN", name)

        self.features = np.load(path + "/features.npz")["arr_0"]
        self.targets_clas = np.load(path + "/targets_clas.npz")["arr_0"]
        self.targets_energy = np.load(path + "/targets_energy.npz")["arr_0"]
        self.targets_position = np.load(path + "/targets_position.npz")["arr_0"]
        self.targets_theta = np.load(path + "/targets_theta.npz")["arr_0"]
        self.pe = np.load(path + "/primary_energy.npz")["arr_0"]
        self.sp = np.load(path + "/source_position_z.npz")["arr_0"]

        # set primary target, initialized with classification targets
        self.targets = self.targets_clas

        # Train-Test-Valid split
        self.p_train = 0.7
        self.p_test = 0.1
        self.p_valid = 0.2
        self.entries = len(self.targets_clas)

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

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
        self.ary_idx = ary_idx[self.targets_clas == 1.0]

        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_indexing_all(self):
        ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    def update_targets_clas(self):
        # set legacy targets to be module energies
        self.targets = self.targets_clas

    def update_targets_energy(self):
        # set legacy targets to be module energies
        self.targets = self.targets_energy

    def update_targets_position(self):
        # set legacy targets to be module energies
        self.targets = self.targets_position

    def update_targets_theta(self):
        # set legacy targets to be module energies
        self.targets = self.targets_theta

    def get_classweights(self):
        # set sample weights to class weights
        _, counts = np.unique(self.targets_clas, return_counts=True)
        class_weights = {0: len(self.targets_clas) / (2 * counts[0]),
                         1: len(self.targets_clas) / (2 * counts[1])}

        return class_weights

    def update_targets_regression(self):
        self.targets = np.concatenate(
            [self.targets_energy, self.targets_position], axis=1)

    def num_features(self):
        return self.features.shape[1]

    def get_standardization(self, n_cluster=10, n_features=10):
        """Calculate feature wise standardization.

        Features are expected as (n_cluster, n_feature) dimension.


        """
        list_mean = []
        list_std = []

        for i in range(n_features):
            ary_con = np.reshape(self.features[:, :, i],
                                 (self.features.shape[0] * n_cluster,))
            ary_con = ary_con[ary_con != 0.0]
            list_mean.append(np.mean(ary_con))
            list_std.append(np.std(ary_con))

        return list_mean, list_std

    def standardize(self, list_mean, list_std, n_features=10):
        for i in range(n_features):
            self.features[:, :, i] = (self.features[:, :, i] - list_mean[
                i]) / list_std[i]
