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

    ary_features = npz_data["features"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_weights = npz_data["weights"]

    return DataCluster.DataCluster(ary_features=ary_features,
                                   ary_targets_clas=ary_targets_clas,
                                   ary_targets_reg1=ary_targets_reg1,
                                   ary_targets_reg2=ary_targets_reg2,
                                   ary_weights=ary_weights)


def wrapper(npz_file,
            set_testall=False,
            set_classweights=True):
    # load npz file into DataCluster object
    data_cluster = parse(npz_file)

    # if predict_all, set test-train ratio to 1.0 to fully predict the dataset
    if set_testall:
        data_cluster.p_train = 0.0
        data_cluster.p_valid = 0.0
        data_cluster.p_test = 1.0

    if set_classweights:
        data_cluster.weights *= data_cluster.get_classweights()

    return data_cluster
