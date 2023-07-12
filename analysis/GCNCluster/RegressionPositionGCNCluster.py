import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset
import downloader

from ClassificationGCNCluster import GCNmodel

from spektral.layers import GCNConv, ECCConv, GlobalSumPool
from spektral.data.loaders import DisjointLoader
from spektral.transforms import GCNFilter

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_regression, plot_position_error


def lr_scheduler(epoch):
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5


def main():
    # defining hyper parameters
    dropout = 0.0
    nNodes = 64
    batch_size = 64
    nEpochs = 40

    trainsplit = 0.7
    valsplit = 0.1

    RUN_NAME = "GCNCluster"
    do_training = True
    do_evaluate = True

    # create dictionary for model parameter
    modelParameter = {"nOutput": 6,
                      "OutputActivation": "linear",
                      "dropout": dropout,
                      "nNodes": nNodes}

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "GraphCluster_OptimisedGeometry_Continuous_2e10protons_taggingv3"
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

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    if do_training:
        data = dataset.GraphCluster(name=DATASET_CONT,
                                    edge_atr=True,
                                    adj_arg="binary",
                                    TPOnly=True,
                                    regression="Position")
        tf_model = GCNmodel(**modelParameter)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
        tf_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=list_metrics)
        l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

        # apply GCN filter
        # generate disjoint loader from dataset
        data.apply(GCNFilter())
        idx1 = int(trainsplit * len(data))
        idx2 = int((trainsplit + valsplit) * len(data))
        dataset_tr = data[:idx1]
        dataset_va = data[idx1:idx2]
        dataset_te = data[idx2:]
        loader_train = DisjointLoader(dataset_tr,
                                      batch_size=batch_size,
                                      epochs=nEpochs)
        loader_valid = DisjointLoader(dataset_va,
                                      batch_size=batch_size)

        history = tf_model.fit(loader_train,
                               epochs=nEpochs,
                               steps_per_epoch=loader_train.steps_per_epoch,
                               validation_data=loader_valid,
                               validation_steps=loader_valid.steps_per_epoch,
                               verbose=1,
                               callbacks=[l_callbacks])

        os.chdir(path_results)
        # save model
        print("Saving model at: ", RUN_NAME + "_regressionPosition" + ".h5")
        tf_model.save(RUN_NAME + "_regressionPosition")
        # save training history (not needed tbh)
        with open(RUN_NAME + "_regressionPosition_history" + ".hst", 'wb') as f_hist:
            pkl.dump(history.history, f_hist)
        # save norm
        np.save(RUN_NAME + "_regressionPosition" + "_norm_x", data.norm_x)
        np.save(RUN_NAME + "_regressionPosition" + "_norm_e", data.norm_e)
        # save model parameter as json
        with open(RUN_NAME + "_regressionPosition_parameter.json", "w") as json_file:
            json.dump(modelParameter, json_file)

        # plot training history
        plot_history_regression(history.history, RUN_NAME + "_history_regressionPosition")

    if do_evaluate:
        os.chdir(path_results)
        # load model, model parameter, norm, history
        with open(RUN_NAME + "_regressionPosition_parameter.json", "r") as json_file:
            modelParameter = json.load(json_file)
        tf_model = tf.keras.models.load_model(RUN_NAME + "_regressionPosition")
        norm_x = np.load(RUN_NAME + "_regressionPosition_norm_x.npy")
        norm_e = np.load(RUN_NAME + "_regressionPosition_norm_e.npy")

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
        tf_model.compile(optimizer=optimizer,
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
                                        norm_e=norm_e,
                                        TPOnly=True,
                                        regression="Position")

            # apply GCN filter, generate disjoint loaders from dataset
            data.apply(GCNFilter())
            loader_test = DisjointLoader(data,
                                         batch_size=batch_size,
                                         epochs=1,
                                         shuffle=False)

            y_true = []
            y_pred = []
            for batch in loader_test:
                inputs, target = batch
                p = tf_model(inputs, training=False)
                y_true.append(target)
                y_pred.append(p.numpy())

            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_true = np.reshape(y_true, newshape=(y_true.shape[0], 6))
            y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 6))

            # evaluate model:
            plot_position_error(y_pred, y_true, "error_regression_position")


if __name__ == "__main__":
    main()
