import numpy as np
import os
import json
import tensorflow as tf

import dataset
from spektral.data.loaders import DisjointLoader
from spektral.transforms import GCNFilter
from SiFiCCNN.ImageReconstruction import IRExport


def main():
    # defining hyper parameters
    RUN_NAME = "GCNCluster"
    threshold = 0.5

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "GraphCluster_OptimisedGeometry_Mixed_taggingv3"
    DATASET_0MM = "GraphCluster_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    DATASET_5MM = "GraphCluster_OptimisedGeometry_BP5mm_4e9protons_taggingv3"

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
    model_clas = tf.keras.models.load_model(RUN_NAME + "_classifier")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=list_metrics)

    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")
    norm_e = np.load(RUN_NAME + "_classifier_norm_e.npy")

    # regression energy
    with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    model_regE = tf.keras.models.load_model(RUN_NAME + "_regressionEnergy")
    # norm_x = np.load(RUN_NAME + "_regressionEnergy_norm_x.npy")
    # norm_e = np.load(RUN_NAME + "_regressionEnergy_norm_e.npy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    model_regE.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=list_metrics)

    # regression position
    with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
        modelParameter = json.load(json_file)
    model_regP = tf.keras.models.load_model(RUN_NAME + "_regressionPosition")
    # norm_x = np.load(RUN_NAME + "_regressionPosition_norm_x.npy")
    # norm_e = np.load(RUN_NAME + "_regressionPosition_norm_e.npy")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "mean_absolute_error"
    list_metrics = ["mean_absolute_error"]
    model_regP.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=list_metrics)

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        # predict test dataset
        os.chdir(path_results + file + "/")

        # load dataset
        data = dataset.GraphCluster(name=file,
                                    edge_atr=True,
                                    adj_arg="binary",
                                    norm_x=norm_x,
                                    norm_e=norm_e)

        # apply GCN filter, generate disjoint loaders from dataset
        data.apply(GCNFilter())
        loader_test = DisjointLoader(data,
                                     batch_size=128,
                                     epochs=1,
                                     shuffle=False)

        # classification score
        y_scores = []
        for batch in loader_test:
            inputs, target = batch
            p = model_clas(inputs, training=False)
            y_scores.append(p.numpy())
        y_scores = np.vstack(y_scores)
        y_scores = np.reshape(y_scores, newshape=(y_scores.shape[0],))

        # regression energy
        y_pred_E = []
        for batch in loader_test:
            inputs, target = batch
            p = model_regE(inputs, training=False)
            y_pred_E.append(p.numpy())

        y_pred_E = np.vstack(y_pred_E)
        y_pred_E = np.array(y_pred_E)
        y_pred_E = np.reshape(y_pred_E, newshape=(y_pred_E.shape[0], 2))

        # regression position
        y_pred_P = []
        for batch in loader_test:
            inputs, target = batch
            p = model_regP(inputs, training=False)
            y_pred_P.append(p.numpy())
        y_pred_P = np.vstack(y_pred_P)
        y_pred_P = np.array(y_pred_P)
        y_pred_P = np.reshape(y_pred_P, newshape=(y_pred_P.shape[0], 6))

        # full neural network chain
        nn_pred = np.zeros(shape=(len(y_scores), 9))
        nn_pred[:, 0] = y_scores
        nn_pred[:, 1:3] = y_pred_E
        nn_pred[:, 3:] = y_pred_P

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
