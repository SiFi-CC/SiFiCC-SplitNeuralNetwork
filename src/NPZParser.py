import numpy as np

from src import DataCluster


def parse(npz_file,
          frac=1.0,
          set_testall=False,
          set_classweights=True):
    """
    Args:
         npz_file (): numpy compressed file containing dictionary entries for:
                        - "features"
                        - "targets_clas"
                        - "targets_reg1"
                        - "targets_reg2"
                        - "weights"
        frac (float): fraction of values taken from npz_file. Use to under sample dataset
        set_testall (bloolean): If True, sets test data fraction to 100%
        set_classweights (boolean): If True, sets classweights for classification task


    Return:
        DataCluster object
    """
    # load npz file anc extract all dataframes
    npz_data = np.load(npz_file)
    ary_features = npz_data["features"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_weights = npz_data["weights"]
    ary_meta = npz_data["meta"]

    # apply fraction
    if frac != 1.0:
        n_frac = int(len(ary_features) * frac)
        ary_features = ary_features[:n_frac, :]
        ary_targets_clas = ary_targets_clas[:n_frac]
        ary_targets_reg1 = ary_targets_reg1[:n_frac, :]
        ary_targets_reg2 = ary_targets_reg2[:n_frac, :]
        ary_weights = ary_weights[:n_frac]
        ary_meta = ary_meta[:n_frac]

    data_cluster = DataCluster.DataCluster(ary_features=ary_features,
                                           ary_targets_clas=ary_targets_clas,
                                           ary_targets_reg1=ary_targets_reg1,
                                           ary_targets_reg2=ary_targets_reg2,
                                           ary_weights=ary_weights,
                                           ary_meta=ary_meta)

    if set_testall:
        data_cluster.p_train = 0.0
        data_cluster.p_valid = 0.0
        data_cluster.p_test = 1.0

    if set_classweights:
        data_cluster.weights *= data_cluster.get_classweights()

    return data_cluster
