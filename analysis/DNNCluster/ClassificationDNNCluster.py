import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_classifier, \
    plot_score_distribution, \
    plot_roc_curve, \
    plot_efficiencymap, \
    plot_sp_distribution, \
    plot_pe_distribution, \
    plot_2dhist_ep_score, \
    plot_2dhist_sp_score


def setupModel(nOutput,
               OutputActivation,
               dropout,
               nNodes,
               nCluster=10,
               activation="relu", ):
    """
    Method for building the keras sequential model.

    Args:
        nOutput: int, number of output nodes
        OutputActivation: activation function of output layer
        dropout: float, dropout percentage
        nNodes: int, number of nodes in network
        nCluster: int, number of total clusters used in input
        activation: main activation function for all hidden layers

    Returns:
        keras model
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(nCluster, 10)))
    model.add(tf.keras.layers.Dense(nNodes, activation=activation))
    model.add(tf.keras.layers.Dense(nNodes, activation=activation))

    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(int(nNodes / 2), activation=activation))
    model.add(tf.keras.layers.Dense(int(nNodes / 2), activation=activation))

    model.add(tf.keras.layers.Dense(nOutput, activation=OutputActivation))

    return model


def lr_scheduler(epoch):
    """
    Learning rate scheduler used for training. Given to callblack.

    Args:
        epoch:

    Returns:
        learning rate
    """
    if epoch < 5:
        return 1e-3
    if epoch < 10:
        return 5e-4
    if epoch < 20:
        return 1e-4
    return 1e-5


def main():
    # defining hyper parameters
    sx = 4
    ax = 6
    dropout = 0.10
    nNodes = 64
    batch_size = 64
    nEpochs = 20

    RUN_NAME = "DNNCluster_" + "S" + str(sx) + "A" + str(ax)
    do_training = True
    do_evaluate = True

    # create dictionary for model parameter
    modelParameter = {"nOutput": 1,
                      "OutputActivation": "sigmoid",
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
        # main training loop:
        # Load dataset class, build keras model, calculate standardization parameter, apply them
        # apply class-weights for classification, fit the model, save all results
        data = dataset.DenseCluster(name=DATASET_CONT)
        tf_model = setupModel(**modelParameter)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss = "binary_crossentropy"
        list_metrics = ["Precision", "Recall"]
        tf_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=list_metrics)
        l_callbacks = [tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]

        # set normalization from training dataset
        norm = data.get_standardization(10, 10)
        data.standardize(norm, 10)
        class_weights = data.get_classweights()

        history = tf_model.fit(data.x_train(),
                               data.y_train(),
                               validation_data=(data.x_valid(), data.y_valid()),
                               epochs=nEpochs,
                               batch_size=batch_size,
                               class_weight=class_weights,
                               verbose=1,
                               callbacks=[l_callbacks])

        os.chdir(path_results)
        # save model
        print("Saving model at: ", RUN_NAME + "_classifier" + ".h5")
        tf_model.save(RUN_NAME + "_classifier" + ".h5")
        # save training history (not needed tbh)
        with open(RUN_NAME + "_classifier_history" + ".hst", 'wb') as f_hist:
            pkl.dump(history.history, f_hist)
        # save norm
        np.save(RUN_NAME + "_classifier" + "_norm.npy", norm)
        # plot training history
        plot_history_classifier(history.history, RUN_NAME + "_history_classifier")
        # save model parameter as json
        with open(RUN_NAME + "_classifier_parameter.json", "w") as json_file:
            json.dump(modelParameter, json_file)

    if do_evaluate:
        os.chdir(path_results)
        # load model, model parameter, norm, history
        with open(RUN_NAME + "_classifier_parameter.json", "r") as json_file:
            modelParameter = json.load(json_file)
        tf_model = setupModel(**modelParameter)
        tf_model.load_weights(RUN_NAME + "_classifier" + ".h5")
        norm = np.load(RUN_NAME + "_classifier" + "_norm.npy")

        for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
            # predict test dataset
            os.chdir(path_results + file + "/")

            # load dataset
            data = dataset.DenseCluster(file)

            if file in [DATASET_0MM, DATASET_5MM]:
                data.p_train = 0.0
                data.p_valid = 0.0
                data.p_test = 1.0

            # set normalization from training dataset
            data.standardize(norm, 10)

            y_scores = tf_model.predict(data.x_test())
            y_true = data.y_test()
            y_scores = np.reshape(y_scores, newshape=(y_scores.shape[0],))

            # evaluate model:
            #   - ROC analysis
            #   - Score distribution#
            #   - Binary classifier metrics

            _, theta_opt, (list_fpr, list_tpr) = fastROCAUC.fastROCAUC(y_scores,
                                                                       y_true,
                                                                       return_score=True)
            plot_roc_curve(list_fpr, list_tpr, "rocauc_curve")
            plot_score_distribution(y_scores, y_true, "score_dist")
            metrics.write_metrics_classifier(y_scores, y_true)
            plot_efficiencymap(y_pred=y_scores,
                               y_true=y_true,
                               y_sp=data.sp[data.idx_test()],
                               figure_name="efficiencymap")
            plot_sp_distribution(ary_sp=data.sp[data.idx_test()],
                                 ary_score=y_scores,
                                 ary_true=y_true,
                                 figure_name="sp_distribution")
            plot_pe_distribution(ary_pe=data.pe[data.idx_test()],
                                 ary_score=y_scores,
                                 ary_true=y_true,
                                 figure_name="pe_distribution")
            plot_2dhist_sp_score(sp=data.sp[data.idx_test()],
                                 y_score=y_scores,
                                 y_true=y_true,
                                 figure_name="2dhist_sp_score")
            plot_2dhist_ep_score(pe=data.pe[data.idx_test()],
                                 y_score=y_scores,
                                 y_true=y_true,
                                 figure_name="2dhist_pe_score")


if __name__ == "__main__":
    main()
