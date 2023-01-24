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


def wrapper(npz_file,
            set_testall=False,
            standardize=True,
            set_classweights = True,
            set_energyweights=False,
            set_peakweights=False):
    # load npz file into DataCluster object
    data_cluster = parse(npz_file)

    # if predict_all, set test-train ratio to 1.0 to fully predict the dataset
    if set_testall:
        data_cluster.p_train = 0.0
        data_cluster.p_valid = 0.0
        data_cluster.p_test = 1.0

    if standardize:
        data_cluster.standardize()

    if set_classweights:
        data_cluster.weights *= data_cluster.get_classweights()

    if set_energyweights:
        data_cluster.weights *= data_cluster.get_energyweights()

    if set_peakweights:
        data_cluster.weights *= data_cluster.get_peakweights()

    return data_cluster
