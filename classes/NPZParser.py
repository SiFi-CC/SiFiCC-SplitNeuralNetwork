import numpy as np

from classes import DataCluster


def parse(npz_file):
    """

    Args:
         npz_file (): numpy compressed file containing dictionary entries for:
                        - "META"
                        - "features"
                        - "targets"
                        - "weights"

    Return:
        DataCluster object
    """
    npz_data = np.load(npz_file)

    ary_meta = npz_data["META"]
    ary_features = npz_data["features"]
    ary_targets = npz_data["targets"]
    ary_weights = npz_data["weights"]

    return DataCluster.DataCluster(ary_meta, ary_features, ary_targets, ary_weights)
