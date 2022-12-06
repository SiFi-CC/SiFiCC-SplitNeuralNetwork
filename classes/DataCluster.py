import numpy as np


class DataCluster:

    def __init__(self, ary_meta, ary_features, ary_targets, ary_weights):
        self.meta = ary_meta
        self.features = ary_features
        self.targets = ary_targets
        self.weights = ary_weights

        self.p_train = 0.7
        self.p_test = 0.2
        self.p_valid = 0.1

        # generate shuffled indices with a random generator
        self.ary_idx = np.arange(0, len(self.targets), 1.0, dtype=int)
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

    def standardize(self):
        for i in range(self.features.shape[1]):
            self.features[:, i] = (self.features[:, i] - np.mean(self.features[:, i])) / np.std(self.features[:, i])
