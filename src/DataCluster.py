import numpy as np


class DataCluster:

    def __init__(self, ary_meta, ary_features, ary_targets_clas, ary_targets_reg1, ary_targets_reg2, ary_weights):
        self.meta = ary_meta
        self.features = ary_features
        self.weights = ary_weights
        self.targets_clas = ary_targets_clas
        self.targets_reg1 = ary_targets_reg1
        self.targets_reg2 = ary_targets_reg2

        # legacy feature
        # can still be used as final targets for whatever
        self.targets = ary_targets_clas

        self.p_train = 0.7
        self.p_test = 0.2
        self.p_valid = 0.1

        self.entries = len(ary_targets_clas)

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    ####################################################################################################################

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

    def num_features(self):
        return self.features.shape[1]

    def standardize(self):
        # save mean and std of every feature
        self.list_mean = []
        self.list_std = []

        for i in range(self.features.shape[1]):
            self.list_mean.append(np.mean(self.features[:, i]))
            self.list_std.append(np.std(self.features[:, i]))

            if np.std(self.features[:, i]) == 0.0:
                print("Zero Division in feature :", i)

            self.features[:, i] = (self.features[:, i] - np.mean(self.features[:, i])) / np.std(self.features[:, i])

    def new_standardie(self):
        n_cluster = 8
        n_features = 9
        self.list_mean = []
        self.list_std = []
        list_idx = np.arange(0, n_cluster * n_features, n_features)

        for i in range(9):
            # print(np.array(list_idx) + i)
            ary_con = np.reshape(self.features[:, np.array(list_idx) + i], (self.features.shape[0] * n_cluster,))
            self.list_mean.append(np.mean(ary_con))
            self.list_std.append(np.std(ary_con))

        self.list_mean = self.list_mean * n_cluster
        self.list_std = self.list_std * n_cluster

        for i in range(len(self.list_mean)):
            mean = self.list_mean[i]
            std = self.list_std[i]
            self.features[:, i] -= mean
            self.features[:, i] /= std

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

    def get_classweights(self):
        # set sample weights to class weights
        _, counts = np.unique(self.targets, return_counts=True)
        class_weights = [len(self.targets) / (2 * counts[0]), len(self.targets) / (2 * counts[1])]

        ary_weights = np.ones(shape=(self.targets.shape[0],))
        for i in range(len(self.targets)):
            ary_weights[i] = class_weights[int(self.targets[i])]

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

    def get_peakweights(self):
        # grab MC source positions from meta data
        ary_mc_source_position = self.meta[:, 2]

        # get full range of source positions
        # take the last 10% as heavier weighted peak positions
        source_pos_range = abs(max(ary_mc_source_position) - min(ary_mc_source_position))
        source_pos_threshold = max(ary_mc_source_position) - 0.1 * source_pos_range

        ary_w = np.ones(shape=(self.entries,))
        for i in range(len(ary_w)):
            if ary_mc_source_position[i] > source_pos_threshold:
                ary_w[i] = 5
        return ary_w

    def reset_weights(self):
        self.weights = np.ones(shape=(self.entries,))

    def update_energy_range(self, e_min, e_max):
        """
        updates feature and target data based on energy range. Needs total cluster energy as meta[:,2] entry

        Args:
            e_min:
            e_max:

        return:
            None
        """

        # determine which rows need to be deleted
        list_idx = []
        for i in range(self.entries):
            if not e_min < self.meta[i, 1] < e_max:
                list_idx.append(i)

        self.entries -= len(list_idx)
        # update features, targets and weights
        self.features = np.delete(self.features, obj=np.array(list_idx), axis=0)
        self.targets = np.delete(self.targets, obj=np.array(list_idx), axis=0)
        self.targets_clas = np.delete(self.targets_clas, obj=np.array(list_idx), axis=0)
        self.targets_reg1 = np.delete(self.targets_reg1, obj=np.array(list_idx), axis=0)
        self.targets_reg2 = np.delete(self.targets_reg2, obj=np.array(list_idx), axis=0)
        self.weights = np.delete(self.weights, obj=np.array(list_idx), axis=0)
        self.meta = np.delete(self.meta, obj=np.array(list_idx), axis=0)

        # reshuffle the train-test indexing
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)
