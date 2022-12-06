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

    ary_meta = npz_file["META"]
    ary_features = npz_file["features"]
    ary_targets = npz_file["targets"]
    ary_weights = npz_file["weights"]

    return DataCluster.DataCluster(ary_meta, ary_features, ary_targets, ary_weights)
