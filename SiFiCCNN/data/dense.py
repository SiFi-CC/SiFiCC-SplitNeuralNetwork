import numpy as np


class Dense:
    def __init__(self, x=None, y=None):
        # TODO: check dimension of x and y
        self.x = x
        self.y = y

    @property
    def n_features(self):
        if self.x is not None:
            return self.x.shape[:1]
        else:
            return None

    @property
    def n_labels(self):
        if self.y is not None:
            shp = np.shape(self.y)
            return 1 if len(shp) == 0 else shp[-1]
        else:
            return None
