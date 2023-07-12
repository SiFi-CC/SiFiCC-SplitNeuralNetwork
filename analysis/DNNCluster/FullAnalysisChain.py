import numpy as np
import os
import json

import dataset
from ClassificationDNNCluster import setupModel

from SiFiCCNN.ImageReconstruction import IRExport


def main():
    # defining hyper parameters
    RUN_NAME = "DNNCluster_S4A6"
    threshold = 0.5

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "DenseClusterS4A6_OptimisedGeometry_Continuous_2e10protons_taggingv3"
    DATASET_0MM = "DenseClusterS4A6_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    DATASET_5MM = "DenseClusterS4A6_OptimisedGeometry_BP5mm_4e9protons_taggingv3"

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    path_results = path_main + "/results/" + RUN_NAME + "/"
    path_datasets = path_main + "/datasets/"

    # load all 3 tf models
    os.chdir(path_results)
    # classifier
    with open(RUN_NAME + "_classifier_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    model_clas = setupModel(**modelParameter)
    model_clas.load_weights(RUN_NAME + "_classifier" + ".h5")
    # regression energy
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    model_regE = setupModel(**modelParameter)
    model_regE.load_weights(RUN_NAME + "_regressionEnergy" + ".h5")
    # regression position
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    model_regP = setupModel(**modelParameter)
    model_regP.load_weights(RUN_NAME + "_regressionPosition" + ".h5")

    norm = np.load(RUN_NAME + "_classifier" + "_norm.npy")

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        # predict test dataset
        os.chdir(path_results + file + "/")
        # load dataset
        data = dataset.DenseCluster(file)

        # TEMPORARY: DROP OUT ALL EVENTS WITH MORE THAN ONE SCATTERER CLUSTER
        data.update_indexing_ordered()

        # set full dataset to test sample
        data.p_train = 0.0
        data.p_valid = 0.0
        data.p_test = 1.0
        # set normalization from training dataset
        data.standardize(norm, 10)

        # full neural network chain
        y_scores = model_clas.predict(data.x_test())
        y_pred_energy = model_regE.predict(data.x_test())
        y_pred_position = model_regP.predict(data.x_test())

        nn_pred = np.zeros(shape=(len(data.y_test()), 9))
        nn_pred[:, 0] = np.reshape(y_scores, newshape=(len(y_scores),))
        nn_pred[:, 1:3] = np.reshape(y_pred_energy,
                                     newshape=(y_pred_energy.shape[0], y_pred_energy.shape[1]))
        nn_pred[:, 3:] = np.reshape(y_pred_position,
                                    newshape=(y_pred_position.shape[0], y_pred_position.shape[1]))

        # This is done this way cause y_scores gets a really dumb shape from tensorflow
        idx_clas_pos = [float(y_scores[i]) > threshold for i in range(len(y_scores))]

        # export prediction to a usable npz file
        with open(file + "_prediction.npz", 'wb') as f_output:
            np.savez_compressed(f_output, nn_pred=nn_pred)

        # export to root file compatible with CC6 image reconstruction
        IRExport.export_CC6(ary_e=nn_pred[idx_clas_pos, 1],
                            ary_p=nn_pred[idx_clas_pos, 2],
                            ary_ex=nn_pred[idx_clas_pos, 3],
                            ary_ey=nn_pred[idx_clas_pos, 4],
                            ary_ez=nn_pred[idx_clas_pos, 5],
                            ary_px=nn_pred[idx_clas_pos, 6],
                            ary_py=nn_pred[idx_clas_pos, 7],
                            ary_pz=nn_pred[idx_clas_pos, 8],
                            filename="CC6IR_NNRECO_" + file + "_theta" + str(threshold).replace(".",
                                                                                                ""),
                            verbose=1,
                            veto=True)


if __name__ == "__main__":
    main()
