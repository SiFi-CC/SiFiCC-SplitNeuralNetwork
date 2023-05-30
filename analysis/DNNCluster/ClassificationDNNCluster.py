import numpy as np
import os
import pickle as pkl
import tensorflow as tf

import dataset
import downloader

from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.plotting import plt_models


def generate_dataset(n=None):
    from SiFiCCNN.root import Root

    # Used root files
    ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
    ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
    ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"

    # grab current directory, cd backwards 2 times to reach main folder
    # very dumb solution right now and needs improvement
    path_repo_main = os.getcwd()
    path_repo_main = os.path.abspath(os.path.join(path_repo_main, os.pardir))
    path_repo_main = os.path.abspath(os.path.join(path_repo_main, os.pardir))

    path_root = path_repo_main + "/root_files/"
    path_datasets = path_repo_main + "/datasets/SiFiCCNN_DenseCluster/"

    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(path_root + file)
        downloader.load(root, n=n)


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
    do_training = False
    do_evaluate = False

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "DenseClusterS4A6_OptimisedGeometry_Continuous_2e10protons"
    DATASET_0MM = "DenseClusterS4A6_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
    DATASET_5MM = "DenseClusterS4A6_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

    # grab current directory, cd backwards 2 times to reach main folder
    # very dumb solution right now and needs improvement
    path_repo_main = os.getcwd()
    path_repo_main = os.path.abspath(os.path.join(path_repo_main, os.pardir))
    path_repo_main = os.path.abspath(os.path.join(path_repo_main, os.pardir))

    path_root = path_repo_main + "/root_files/"
    path_datasets = path_repo_main + "/datasets/SiFiCCNN/"
    path_results = path_repo_main + "/results/" + RUN_NAME + "/"

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        if not os.path.isdir(path_results + "/" + file + "/"):
            os.mkdir(path_results + "/" + file + "/")

    if do_training:
        data = dataset.DenseCluster(name=DATASET_CONT)
        tf_model = setupModel(nCluster=10,
                              nOutput=1,
                              dropout=dropout,
                              nNodes=nConnectedNodes,
                              activation="relu",
                              output_activation="sigmoid")

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
            pkl.dump(history, f_hist)
        # save norm
        np.save(RUN_NAME + "_classifier" + "_norm.npy", norm)

    if do_evaluate:
        os.chdir(path_results)
        # load model, norm, history
        tf_model = setupModel(nCluster=10,
                              nOutput=1,
                              dropout=dropout,
                              nNodes=nConnectedNodes,
                              activation="relu",
                              output_activation="sigmoid")
        tf_model.load(RUN_NAME + "_classifier" + ".h5")
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
            plt_models.roc_curve(list_fpr, list_tpr, "rocauc_curve")
            plt_models.score_distribution(y_scores, y_true, "score_dist")
            metrics.write_metrics_classifier(y_scores, y_true)


if __name__ == "__main__":
    gen_dataset = True
    if gen_dataset:
        generate_dataset(n=1000)
    else:
        main()
