import numpy as np

from src.SiFiCCNN.Data import DFCluster
from src.SiFiCCNN.Data import DFSiPM


def parse_cluster(npz_file,
                  n_frac=1.0):
    """
    Args:
         npz_file (): numpy compressed file containing dictionary entries for:
                        - "features"
                        - "targets_clas"
                        - "targets_reg1"
                        - "targets_reg2"
                        - "targets_reg3"
                        - "weights"
                        - "meta"
        n_frac (float): fraction of values taken from npz_file. Use to under sample dataset

    Return:
        DataCluster object
    """
    # load npz file anc extract all arrays
    npz_data = np.load(npz_file)
    ary_features = npz_data["features"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_targets_reg3 = npz_data["targets_reg3"]
    ary_weights = npz_data["weights"]
    ary_meta = npz_data["meta"]

    # apply fraction
    if n_frac != 1.0:
        n_frac = int(len(ary_features) * n_frac)
        ary_features = ary_features[:n_frac, :]
        ary_targets_clas = ary_targets_clas[:n_frac]
        ary_targets_reg1 = ary_targets_reg1[:n_frac, :]
        ary_targets_reg2 = ary_targets_reg2[:n_frac, :]
        ary_targets_reg3 = ary_targets_reg3[:n_frac]
        ary_weights = ary_weights[:n_frac]
        ary_meta = ary_meta[:n_frac]

    df_cluster = DFCluster.DFCluster(ary_features=ary_features,
                                     ary_targets_clas=ary_targets_clas,
                                     ary_targets_reg1=ary_targets_reg1,
                                     ary_targets_reg2=ary_targets_reg2,
                                     ary_targets_reg3=ary_targets_reg3,
                                     ary_weights=ary_weights,
                                     ary_meta=ary_meta)

    # Further preprocessing of the Dataframe
    df_cluster.weights *= df_cluster.get_classweights()

    return df_cluster
