import numpy as np
import os
import pickle as pkl
import json
import tensorflow as tf

import dataset
import downloader

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_classifier, \
    plot_score_distribution, \
    plot_roc_curve


def generate_dataset(n=None):
    from SiFiCCNN.root import Root

    # Used root files
    ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
    ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
    ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path

    path_root = path_main + "/root_files/"
    path_datasets = path_main + "/datasets/"

    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(path_root + file)
        downloader.load(root,
                        path=path_datasets,
                        n=n)


def setupModel(nOutput,
               OutputActivation,
               dropout,
               nNodes,
               nCluster=10,
               activation="relu", ):
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
    if epoch < 20:
        return 1e-3
    if epoch < 30:
        return 5e-4
    if epoch < 40:
        return 1e-4
    return 1e-5


def main():
    # defining hyper parameters
    sx = 4
    ax = 6
    dropout = 0.1
    nNodes = 64
    batch_size = 64
    nEpochs = 10

    RUN_NAME = "DNNCluster_" + "S" + str(sx) + "A" + str(ax)
    do_training = False
    do_evaluate = False

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
    DATASET_CONT = "DenseClusterS4A6_OptimisedGeometry_Continuous_2e10protons"
    DATASET_0MM = "DenseClusterS4A6_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
    DATASET_5MM = "DenseClusterS4A6_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

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
            data = dataset.DenseCluster(DATASET_CONT)

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


if __name__ == "__main__":
    gen_dataset = False
    if gen_dataset:
        generate_dataset(n=100000)
    else:
        main()
