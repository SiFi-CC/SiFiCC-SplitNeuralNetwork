import numpy as np

from src import DataCluster


def parse(npz_file):
    """

    Args:
         npz_file (): numpy compressed file containing dictionary entries for:
                        - "META"
                        - "features"
                        - "targets_clas"
                        - "targets_reg1"
                        - "targets_reg2"
                        - "weights"

    Return:
        DataCluster object
    """
    npz_data = np.load(npz_file)

    ary_meta = npz_data["META"]
    ary_features = npz_data["features"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_weights = npz_data["weights"]

    return DataCluster.DataCluster(ary_meta=ary_meta,
                                   ary_features=ary_features,
                                   ary_targets_clas=ary_targets_clas,
                                   ary_targets_reg1=ary_targets_reg1,
                                   ary_targets_reg2=ary_targets_reg2,
                                   ary_weights=ary_weights)
