import numpy as np
import os
import json
import tensorflow as tf

from spektral.layers import EdgeConv, GlobalMaxPool
from spektral.data.loaders import DisjointLoader
from SiFiCCNN.utils.layers import ReZero

from analysis.EdgeConvResNetCluster import dataset
from SiFiCCNN.ImageReconstruction import IRExport


def main():
    # defining hyper parameters
    RUN_NAME = "EdgeConvResNetCluster"
    RUN_CODE = "ECRNCluster"
    threshold = 0.5

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "GraphCluster_OptimisedGeometry_Continuous_2e10protons_taggingv3"
    DATASET_0MM = "GraphCluster_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    DATASET_5MM = "GraphCluster_OptimisedGeometry_BP5mm_4e9protons_taggingv3"
    DATASET_m5MM = "GraphCluster_OptimisedGeometry_BPminus5mm_4e9_protons_taggingv3"

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
    # load model, model parameter, norm, history
    with open(RUN_NAME + "_classifier_parameter.json", "r") as json_file:
        classifier_modelParameter = json.load(json_file)
    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    classifier_model = tf.keras.models.load_model(RUN_NAME + "_classifier.tf",
                                                  custom_objects={"EdgeConv": EdgeConv,
                                                                  "GlobalMaxPool": GlobalMaxPool,
                                                                  "ReZero": ReZero})
    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")
    norm_e = np.load(RUN_NAME + "_classifier_norm_e.npy")
    print("Classifier model loaded!")
    # regression energy
    # load model, model parameter, norm, history
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        regEnergy_modelParameter = json.load(json_file)
    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    regEnergy_model = tf.keras.models.load_model(RUN_NAME + "_regressionEnergy.tf",
                                                 custom_objects={"EdgeConv": EdgeConv,
                                                                 "GlobalMaxPool": GlobalMaxPool,
                                                                 "ReZero": ReZero})
    print("Energy Regression model loaded!")
    # regression position
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        regPosition_modelParameter = json.load(json_file)

    # load tensorflow model
    # Custom layers have to be stated to load accordingly
    regPosition_model = tf.keras.models.load_model(RUN_NAME + "_regressionPosition.tf",
                                                   custom_objects={"EdgeConv": EdgeConv,
                                                                   "GlobalMaxPool": GlobalMaxPool,
                                                                   "ReZero": ReZero})
    print("Position Regression model loaded!")

    for file in [DATASET_0MM]:
        # gather all network predictions
        y_score_pred = get_model_pred(file, path_datasets, classifier_model, norm_x, norm_e, None)
        y_score_true = get_model_true(file, path_datasets, classifier_model, norm_x, norm_e, None)
        y_regE_pred = get_model_pred(file, path_datasets, regEnergy_model, norm_x, norm_e,
                                     "Energy")
        y_regP_pred = get_model_pred(file, path_datasets, regPosition_model, norm_x, norm_e,
                                     "Position")
        # define positive classified events
        idx_pos = np.zeros(shape=(len(y_score_pred,)))
        for i in range(len(idx_pos)):
            if y_score_true[i] == 0 and y_score_pred[i] > threshold:
                idx_pos[i] = 1
        idx_pos = idx_pos == 1

        os.chdir(path_results + file + "/")

        # export to root file compatible with CC6 image reconstruction
        IRExport.export_CC6(ary_e=y_regE_pred[idx_pos, 0],
                            ary_p=y_regE_pred[idx_pos, 1],
                            ary_ex=y_regP_pred[idx_pos, 0],
                            ary_ey=y_regP_pred[idx_pos, 1],
                            ary_ez=y_regP_pred[idx_pos, 2],
                            ary_px=y_regP_pred[idx_pos, 3],
                            ary_py=y_regP_pred[idx_pos, 4],
                            ary_pz=y_regP_pred[idx_pos, 5],
                            filename="CC6IR_FPONLY_{}_{}".format(RUN_CODE, file),
                            verbose=1,
                            veto=True)


def get_model_pred(name_dataset,
                   path,
                   classifier_model,
                   norm_x,
                   norm_e,
                   reg_type=None):
    # load dataset
    data = dataset.GraphCluster(name=name_dataset,
                                edge_atr=True,
                                adj_arg="binary",
                                norm_x=norm_x,
                                norm_e=norm_e,
                                reg_type=reg_type)
    loader_test = DisjointLoader(data,
                                 batch_size=64,
                                 epochs=1,
                                 shuffle=False)

    y_pred = []
    for batch in loader_test:
        inputs, target = batch
        p = classifier_model(inputs, training=False)
        y_pred.append(p.numpy())
    y_pred = np.vstack(y_pred)
    if reg_type is None:
        y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0],))
    if reg_type == "Energy":
        y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 2))
    if reg_type == "Position":
        y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 6))

    return y_pred

def get_model_true(name_dataset,
                   path,
                   classifier_model,
                   norm_x,
                   norm_e,
                   reg_type=None):
    # load dataset
    data = dataset.GraphCluster(name=name_dataset,
                                edge_atr=True,
                                adj_arg="binary",
                                norm_x=norm_x,
                                norm_e=norm_e,
                                reg_type=reg_type)
    loader_test = DisjointLoader(data,
                                 batch_size=64,
                                 epochs=1,
                                 shuffle=False)

    y_true = []
    for batch in loader_test:
        inputs, target = batch
        y_true.append(target)
    y_true = np.vstack(y_true)
    if reg_type is None:
        y_true = np.reshape(y_true, newshape=(y_true.shape[0],))
    if reg_type == "Energy":
        y_true = np.reshape(y_true, newshape=(y_true.shape[0], 2))
    if reg_type == "Position":
        y_true = np.reshape(y_true, newshape=(y_true.shape[0], 6))

    return y_true


if __name__ == "__main__":
    main()
