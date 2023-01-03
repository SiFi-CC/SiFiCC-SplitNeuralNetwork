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

        # repeat with indices of positive events
        self.ary_idx_pos = [i for i in range(len(self.targets_clas)) if self.targets_clas[i] == 1]
        rng.shuffle(self.ary_idx_pos)

    ####################################################################################################################

    def idx_train(self):
        return self.ary_idx[0: int(len(self.ary_idx) * self.p_train)]

    def idx_valid(self):
        return self.ary_idx[
               int(len(self.ary_idx) * self.p_train): int(len(self.ary_idx) * (self.p_train + self.p_valid))]

    def idx_test(self):
        return self.ary_idx[int(len(self.ary_idx) * (self.p_train + self.p_valid)):]

    def idx_train_pos(self):
        return self.ary_idx_pos[0: int(len(self.ary_idx_pos) * self.p_train)]

    def idx_valid_pos(self):
        return self.ary_idx_pos[
               int(len(self.ary_idx_pos) * self.p_train): int(len(self.ary_idx_pos) * (self.p_train + self.p_valid))]

    def idx_test_pos(self):
        return self.ary_idx_pos[int(len(self.ary_idx_pos) * (self.p_train + self.p_valid)):]

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

    # regression
    def x_train_reg(self):
        return self.features[self.idx_train_pos()]

    def w_train_reg(self):
        return self.weights[self.idx_train_pos()]

    def x_valid_reg(self):
        return self.features[self.idx_valid_pos()]

    def x_test_reg(self):
        return self.features[self.idx_test_pos()]

    def y_train_reg1(self):
        return self.targets_reg1[self.idx_train_pos()]

    def y_train_reg2(self):
        return self.targets_reg2[self.idx_train_pos()]

    def y_valid_reg1(self):
        return self.targets_reg1[self.idx_valid_pos()]

    def y_valid_reg2(self):
        return self.targets_reg2[self.idx_valid_pos()]

    def y_test_reg1(self):
        return self.targets_reg1[self.idx_test_pos()]

    def y_test_reg2(self):
        return self.targets_reg2[self.idx_test_pos()]

    def num_features(self):
        return self.features.shape[1]

    def standardize(self):
        # save mean and std of every feature
        self.list_mean = []
        self.list_std = []

        for i in range(self.features.shape[1]):
            self.list_mean.append(np.mean(self.features[:, i]))
            self.list_std.append(np.std(self.features[:, i]))

            self.features[:, i] = (self.features[:, i] - np.mean(self.features[:, i])) / np.std(self.features[:, i])

    def remove_background_events(self):
        # grab indices of all positives events
        list_idx_positives = self.targets_clas == 1

        # remove all background events
        self.targets = self.targets[list_idx_positives, :]
        self.targets_clas = self.targets_clas[list_idx_positives, :]
        self.targets_reg1 = self.targets_reg1[list_idx_positives, :]
        self.targets_reg2 = self.targets_reg2[list_idx_positives, :]

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

    """
    # COMMENTED OUT AS NORMALIZING WITH MONTE CARLO FEATURES DOES NOT MAKE SENSE
    def normalize_by_eprimary(self):
        # TODO: UPDATE INDEXING
        ary_idx_e = [3, 8, 13, 18, 23, 28, 33, 38]

        for i in range(self.features.shape[1]):
            if i not in ary_idx_e:
                self.features[:, i] = (self.features[:, i] - np.mean(self.features[:, i])) / np.std(self.features[:, i])

        for i in range(len(self.targets)):
            self.features[i, ary_idx_e] = self.features[i, ary_idx_e] / self.meta[i, 1]
    """

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
        # update features, targets and weights
        self.features = np.delete(self.features, obj=np.array(list_idx), axis=0)
        self.targets = np.delete(self.targets, obj=np.array(list_idx), axis=0)
        self.weights = np.delete(self.weights, obj=np.array(list_idx), axis=0)

        # reshuffle the train-test indexing
        self.ary_idx = np.arange(0, self.entries, 1.0, dtype=int)
        rng = np.random.default_rng(42)
        rng.shuffle(self.ary_idx)

