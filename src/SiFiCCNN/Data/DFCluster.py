import numpy as np


class DFCluster:

    def __init__(self,
                 ary_features,
                 ary_targets_clas,
                 ary_targets_reg1,
                 ary_targets_reg2,
                 ary_targets_reg3,
                 ary_weights,
                 ary_meta):

        # all core features of the DataCluster class
        self.features = ary_features
        self.targets_clas = ary_targets_clas
        self.targets_reg1 = ary_targets_reg1
        self.targets_reg2 = ary_targets_reg2
        self.targets_reg3 = ary_targets_reg3
        self.weights = ary_weights
        self.meta = ary_meta

        # legacy feature
        # can still be used as final targets for whatever
        self.targets = ary_targets_clas

        # Train-Test-Valid split
        self.p_train = 0.7
        self.p_test = 0.1
        self.p_valid = 0.2
        self.entries = len(ary_targets_clas)

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    # ------------------------------------------------------------------------------------------------------------------
    # Indexing methods for Train-Test-Validation splitting
    def idx_train(self):
        return self.ary_idx[0: int(len(self.ary_idx) * self.p_train)]

    def idx_valid(self):
        return self.ary_idx[
               int(len(self.ary_idx) * self.p_train): int(len(self.ary_idx) * (self.p_train + self.p_valid))]

    def idx_test(self):
        return self.ary_idx[int(len(self.ary_idx) * (self.p_train + self.p_valid)):]

    # classification
    def x_train(self):
        return self.features[self.idx_train()]

    def y_train(self):
        return self.targets[self.idx_train()]

    def w_train(self):
        return self.weights[self.idx_train()]

    def x_valid(self):
        return self.features[self.idx_valid()]

    def y_valid(self):
        return self.targets[self.idx_valid()]

    def x_test(self):
        return self.features[self.idx_test()]

    def y_test(self):
        return self.targets[self.idx_test()]

    # ------------------------------------------------------------------------------------------------------------------
    def num_features(self):
        return self.features.shape[1]

    def get_standardization(self, n_feat=10, n_cluster=10):
        # save mean and std of every feature in terms of physical feature
        list_mean = []
        list_std = []

        # check if feature dimension is 2D (DNNs) or 3D (RNNs / CNNs)
        # 2D: features are flatten for DNN input, account for feature sequence and special padding
        # 3D: features are 2D arrays, standardization is feature wise, padding is handled by mask layer
        if self.features.ndim == 2:
            # padding, standard values are pre-set
            # updating needs to be done manually later on
            # padding = [0.0, -1.0, -1.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0, 0.0]

            list_idx = np.arange(0, n_feat * n_cluster, n_feat)
            for i in range(len(list_idx)):
                ary_con = np.reshape(self.features[:, np.array(list_idx) + i],
                                     (self.features.shape[0] * n_feat * n_cluster,))
                ary_con = ary_con[ary_con != 0.0]
                list_mean.append(np.mean(ary_con))
                list_std.append(np.std(ary_con))

            list_mean = list_mean * 10
            list_std = list_std * 10

        if self.features.ndim == 3:
            for i in range(n_feat):
                ary_con = np.reshape(self.features[:, :, i], (self.features.shape[0] * n_cluster,))
                ary_con = ary_con[ary_con != 0.0]
                list_mean.append(np.mean(ary_con))
                list_std.append(np.std(ary_con))

        return list_mean, list_std

    def standardize(self, list_mean, list_std):
        # check if feature dimension is 2D (DNNs) or 3D (RNNs / CNNs)
        # 2D: features are flatten for DNN input, account for feature sequence and special padding
        # 3D: features are 2D arrays, standardization is feature wise, padding is handled by mask layer
        if self.features.ndim == 2:
            for i in range(self.features.shape[1]):
                if np.std(self.features[:, i]) == 0.0:
                    print("Zero Division in feature :", i)

                self.features[self.ary_idx, i] = (self.features[self.ary_idx, i] - list_mean[i]) / list_std[i]
        if self.features.ndim == 3:
            for i in range(10):
                self.features[:, :, i] = (self.features[:, :, i] - list_mean[i]) / list_std[i]

    def de_standardize(self, list_mean, list_std):
        for i in range(self.features.shape[1]):
            self.features[:, i] *= list_std[i]
            self.features[:, i] += list_mean[i]

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
        self.targets = self.targets_reg1

    def update_targets_position(self):
        # set legacy targets to be module energies
        self.targets = self.targets_reg2

    def update_targets_theta(self):
        # set legacy targets to be module energies
        self.targets = self.targets_reg3

    def get_classweights(self):
        # set sample weights to class weights
        _, counts = np.unique(self.targets_clas, return_counts=True)
        class_weights = [len(self.targets_clas) / (2 * counts[0]), len(self.targets_clas) / (2 * counts[1])]

        ary_weights = np.ones(shape=(self.targets_clas.shape[0],))
        for i in range(len(self.targets_clas)):
            ary_weights[i] = class_weights[int(self.targets_clas[i])]

        return ary_weights

    def get_energyweights(self):
        # grab energies from meta data
        ary_mc_energy_primary = self.meta[:, 1]

        # set manual energy bins and weights
        bins = np.array([0.0, 2.5, 4.3, 4.5, 8.0, int(max(ary_mc_energy_primary))])
        ary_energy_weights = np.array([0.1, 1.0, 5.0, 3.0, 5.0])

        ary_w = np.ones(shape=(self.entries,))
        for i in range(len(ary_w)):
            for j in range(len(bins) - 1):
                if bins[j] < ary_mc_energy_primary[i] < bins[j + 1]:
                    ary_w[i] = ary_energy_weights[j]
                    break

        return ary_w

    def reset_weights(self):
        self.weights = np.ones(shape=(self.entries,))