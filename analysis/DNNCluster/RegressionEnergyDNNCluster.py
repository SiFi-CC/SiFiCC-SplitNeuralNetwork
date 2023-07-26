import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset

from ClassificationDNNCluster import setupModel
from SiFiCCNN.utils.plotter import plot_history_regression, plot_energy_error


def lr_scheduler(epoch):
    """
    Learning rate scheduler used for training. Given to callblack.

    Args:
        epoch:

    Returns:
        learning rate
    """
    if epoch < 20:
        return 1e-4
    if epoch < 30:
        return 5e-5
    if epoch < 40:
        return 1e-5
    return 1e-6


def main():
    # defining hyper parameters
    sx = 4
    ax = 6
    dropout = 0.0
    nNodes = 64
    batch_size = 64
    nEpochs = 50

    RUN_NAME = "DNNCluster_" + "S" + str(sx) + "A" + str(ax)
    do_training = True
    do_evaluate = True

    # create dictionary for model parameter
    modelParameter = {"nOutput": 2,
                      "OutputActivation": "relu",
                      "dropout": dropout,
                      "nNodes": nNodes,
                      "nCluster": 10,
                      "activation": "relu"}

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

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    if do_training:
        data = dataset.DenseCluster(name=DATASET_CONT)
        data.update_indexing_positives()

        tf_model = setupModel(**modelParameter)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = "mean_absolute_error"
        list_metrics = ["mean_absolute_error"]
        tf_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=list_metrics)
        l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

        # set normalization from training dataset
        # set correct targets, restrict sample to true positive events only
        norm = data.get_standardization(10, 10)
        data.standardize(norm, 10)
        data.update_targets_energy()

        history = tf_model.fit(data.x_train(),
                               data.y_train(),
                               validation_data=(data.x_valid(), data.y_valid()),
                               epochs=nEpochs,
                               batch_size=batch_size,
                               verbose=1,
                               callbacks=[l_callbacks])

        os.chdir(path_results)
        # save model
        print("Saving model at: ", RUN_NAME + "_regressionEnergy" + ".h5")
        tf_model.save(RUN_NAME + "_regressionEnergy" + ".h5")
        # save training history (not needed tbh)
        with open(RUN_NAME + "_regressionEnergy_history" + ".hst", 'wb') as f_hist:
            pkl.dump(history.history, f_hist)
        # save norm
        np.save(RUN_NAME + "_regressionEnergy" + "_norm.npy", norm)

        # plot training history
        plot_history_regression(history.history, RUN_NAME + "_history_regressionEnergy")
        # save model parameter as json
        with open(RUN_NAME + "_regressionEnergy_parameter.json", "w") as json_file:
            json.dump(modelParameter, json_file)

    if do_evaluate:
        os.chdir(path_results)
        # load model, model parameter, norm, history
        with open(RUN_NAME + "_regressionEnergy_parameter.json", "r") as json_file:
            modelParameter = json.load(json_file)
        tf_model = setupModel(**modelParameter)
        tf_model.load_weights(RUN_NAME + "_regressionEnergy" + ".h5")
        norm = np.load(RUN_NAME + "_regressionEnergy" + "_norm.npy")

        for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
            # predict test dataset
            os.chdir(path_results + file + "/")

            # load dataset
            data = dataset.DenseCluster(file)
            data.update_indexing_positives()

            if file in [DATASET_0MM, DATASET_5MM]:
                data.p_train = 0.0
                data.p_valid = 0.0
                data.p_test = 1.0

            # set normalization from training dataset
            data.standardize(norm, 10)
            data.update_targets_energy()

            y_pred = tf_model.predict(data.x_test())
            y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 2))
            y_true = data.y_test()

            # evaluate model:
            plot_energy_error(y_pred, y_true, "error_regression_energy")


if __name__ == "__main__":
    main()
