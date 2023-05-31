import numpy as np
import os
import pickle as pkl
import tensorflow as tf

import dataset
import downloader

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.utils.plotter import plot_history_regression, plot_energy_error


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


def setupModel(nCluster,
               nOutput,
               dropout,
               nNodes,
               activation="relu",
               output_activation="sigmoid"):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(nCluster, 10)))
    model.add(tf.keras.layers.Dense(nNodes, activation=activation))
    model.add(tf.keras.layers.Dense(nNodes, activation=activation))

    if dropout > 0:
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(nOutput, activation=output_activation))

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
    dropout = 0.2
    learning_rate = 1e-3
    nConnectedNodes = 64
    batch_size = 64
    nEpochs = 20

    RUN_NAME = "DNNCluster_" + "S" + str(sx) + "A" + str(ax)
    do_training = True
    do_evaluate = True

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
        tf_model = setupModel(nCluster=10,
                              nOutput=2,
                              dropout=dropout,
                              nNodes=nConnectedNodes,
                              activation="relu",
                              output_activation="relu")

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
        data.update_indexing_positives()
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

    if do_evaluate:
        os.chdir(path_results)
        # load model, norm, history
        tf_model = setupModel(nCluster=10,
                              nOutput=2,
                              dropout=dropout,
                              nNodes=nConnectedNodes,
                              activation="relu",
                              output_activation="relu")
        tf_model.load_weights(RUN_NAME + "_regressionEnergy" + ".h5")
        norm = np.load(RUN_NAME + "_regressionEnergy" + "_norm.npy")

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
            data.update_indexing_positives()
            data.update_targets_energy()

            y_pred = tf_model.predict(data.x_test())
            y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0], 2))
            y_true = data.y_test()

            # evaluate model:

            plot_energy_error(y_pred, y_true, "error_regression_energy")


if __name__ == "__main__":
    gen_dataset = False
    if gen_dataset:
        generate_dataset(n=100000)
    else:
        main()
