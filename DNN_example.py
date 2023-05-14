import numpy as np
import os
import sys
import tensorflow as tf

from src.SiFiCCNN.Root import Root
from src.SiFiCCNN.dataset import downloader, datasets

from src.SiFiCCNN.models import DNN_SXAX
from src.SiFiCCNN.analysis import evaluation
from src.SiFiCCNN.plotting import plt_history

################################################################################
# Global settings
################################################################################

RUN_NAME = "DNN_master"

train_classifier = False
train_regression_energy = False
train_regression_position = False
train_regression_theta = False

eval_classifier = False
eval_regression_energy = True
eval_regression_position = False
eval_regression_theta = False

generate_datasets = False

################################################################################
# Datasets
################################################################################

# Root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"
# Training file used for classification and regression training
# Generated via an input generator, contain one Bragg-peak position
NPZ_FILE_TRAIN = "OptimisedGeometry_Continuous_2e10protons_SiFiCCNNDenseDNN_S4A6.npz"
NPZ_FILE_EVAL_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_SiFiCCNNDenseDNN_S4A6.npz"
NPZ_FILE_EVAL_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_SiFiCCNNDenseDNN_S4A6.npz"

################################################################################
# Set paths
################################################################################

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_datasets = dir_main + "/datasets/SiFiCCNN/"
dir_results = dir_main + "/results/"

# create subdirectory for run output
if not os.path.isdir(dir_results + RUN_NAME + "/"):
    os.mkdir(dir_results + RUN_NAME + "/")
for file in [NPZ_FILE_TRAIN, NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file[:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

################################################################################
# Load dataset , custom setting possible
################################################################################

if generate_datasets:
    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(dir_root + file)
        downloader.sificcnn_dense_sxax(root, n=None)
    sys.exit()

################################################################################
# Training
################################################################################

dropout = 0.0
learning_rate = 1e-3
nConnectedNodes = 64
batch_size = 64

trainsplit = 0.7
valsplit = 0.2
nEpochs = 50

l_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(DNN_SXAX.lr_scheduler)]

if train_classifier:
    data_train = datasets.DatasetCluster(
        npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    m_clas = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=1,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="sigmoid",
                                 loss="binary_clas")

    # set normalization from training dataset
    norm_mean, norm_std = data_train.get_standardization(10, 10)
    data_train.standardize(norm_mean, norm_std, 10)
    class_weights = data_train.get_classweights()

    history = m_clas.fit(data_train.x_train(),
                         data_train.y_train(),
                         validation_data=(
                             data_train.x_valid(), data_train.y_valid()),
                         epochs=nEpochs,
                         batch_size=batch_size,
                         class_weight=class_weights,
                         verbose=1,
                         callbacks=[l_callbacks])

    plt_history.plot_history_classifier(history.history,
                                        RUN_NAME + "_history_classifier")

    # save model
    DNN_SXAX.save_model(m_clas, RUN_NAME + "_classifier")
    DNN_SXAX.save_history(RUN_NAME + "_classifier", history.history)
    DNN_SXAX.save_norm(RUN_NAME + "_classifier", norm_mean, norm_std)

if train_regression_energy:
    data_train = datasets.DatasetCluster(
        npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    m_regE = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=2,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="relu",
                                 loss="binary_clas")

    # set normalization from training dataset
    norm_mean, norm_std = data_train.get_standardization(10, 10)
    data_train.standardize(norm_mean, norm_std, 10)
    data_train.update_targets_energy()
    data_train.update_indexing_positives()

    history = m_regE.fit(data_train.x_train(),
                         data_train.y_train(),
                         validation_data=(
                             data_train.x_valid(), data_train.y_valid()),
                         epochs=nEpochs,
                         batch_size=batch_size,
                         verbose=1,
                         callbacks=[l_callbacks])

    plt_history.plot_history_regression(history.history,
                                        RUN_NAME + "_history_regressionEnergy")

    # save model
    DNN_SXAX.save_model(m_regE, RUN_NAME + "_regressionEnergy")
    DNN_SXAX.save_history(RUN_NAME + "_regressionEnergy", history.history)
    DNN_SXAX.save_norm(RUN_NAME + "_regressionEnergy", norm_mean, norm_std)

if train_regression_position:
    data_train = datasets.DatasetCluster(
        npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    m_regP = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=6,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="linear",
                                 loss="regression")

    # set normalization from training dataset
    norm_mean, norm_std = data_train.get_standardization(10, 10)
    data_train.standardize(norm_mean, norm_std, 10)
    data_train.update_targets_position()
    data_train.update_indexing_positives()

    history = m_regP.fit(data_train.x_train(),
                         data_train.y_train(),
                         validation_data=(
                             data_train.x_valid(), data_train.y_valid()),
                         epochs=nEpochs,
                         batch_size=batch_size,
                         verbose=1,
                         callbacks=[l_callbacks])

    plt_history.plot_history_regression(history.history,
                                        RUN_NAME + "_history_regressionPosition")

    # save model
    DNN_SXAX.save_model(m_regP, RUN_NAME + "_regressionPosition")
    DNN_SXAX.save_history(RUN_NAME + "_regressioPosition", history.history)
    DNN_SXAX.save_norm(RUN_NAME + "_regressionPosition", norm_mean, norm_std)

if train_regression_theta:
    data_train = datasets.DatasetCluster(
        npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    m_regT = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=1,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="linear",
                                 loss="regression")

    # set normalization from training dataset
    norm_mean, norm_std = data_train.get_standardization(10, 10)
    data_train.standardize(norm_mean, norm_std, 10)
    data_train.update_targets_theta()
    data_train.update_indexing_positives()

    history = m_regT.fit(data_train.x_train(),
                         data_train.y_train(),
                         validation_data=(
                             data_train.x_valid(), data_train.y_valid()),
                         epochs=nEpochs,
                         batch_size=batch_size,
                         verbose=1,
                         callbacks=[l_callbacks])

    plt_history.plot_history_regression(history.history,
                                        RUN_NAME + "_history_regressionTheta")

    # save model
    DNN_SXAX.save_model(m_regT, RUN_NAME + "_regressionTheta")
    DNN_SXAX.save_history(RUN_NAME + "_regressioTheta", history.history)
    DNN_SXAX.save_norm(RUN_NAME + "_regressionTheta", norm_mean, norm_std)

################################################################################
# Evaluate model
################################################################################

if eval_classifier:
    os.chdir(dir_results + RUN_NAME + "/")
    m_clas = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=1,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="sigmoid",
                                 loss="binary_clas")

    m_clas = DNN_SXAX.load_model(m_clas, RUN_NAME + "_classifier")
    norm_mean, norm_std = DNN_SXAX.load_norm(RUN_NAME + "_classifier")

    for file in [NPZ_FILE_TRAIN, NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        # load dataset
        data_train = datasets.DatasetCluster(
            npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)
        if file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
            data_train.p_train = 0.0
            data_train.p_valid = 0.0
            data_train.p_test = 1.0

        # set normalization from training dataset
        data_train.standardize(norm_mean, norm_std, 10)

        y_scores = m_clas.predict(data_train.x_test())
        y_true = data_train.y_test()
        y_scores = np.reshape(y_scores, newshape=(y_scores.shape[0],))

        evaluation.eval_classifier(y_scores=y_scores,
                                   y_true=y_true,
                                   theta=0.5)

if eval_regression_energy:
    os.chdir(dir_results + RUN_NAME + "/")
    m_regE = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=2,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="relu",
                                 loss="regression")

    m_regE = DNN_SXAX.load_model(m_regE, RUN_NAME + "_regressionEnergy")
    norm_mean, norm_std = DNN_SXAX.load_norm(RUN_NAME + "_regressionEnergy")

    for file in [NPZ_FILE_TRAIN, NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        # load dataset
        data_train = datasets.DatasetCluster(
            npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)
        if file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
            data_train.p_train = 0.0
            data_train.p_valid = 0.0
            data_train.p_test = 1.0

        # set normalization from training dataset
        data_train.standardize(norm_mean, norm_std, 10)
        data_train.update_targets_energy()
        data_train.update_indexing_positives()

        y_pred = m_regE.predict(data_train.x_test())
        y_true = data_train.y_test()

        evaluation.eval_regression_energy(y_pred, y_true)

if eval_regression_position:
    os.chdir(dir_results + RUN_NAME + "/")
    m_regP = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=6,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="linear",
                                 loss="regression")

    m_regP = DNN_SXAX.load_model(m_regP, RUN_NAME + "_regressionPosition")
    norm_mean, norm_std = DNN_SXAX.load_norm(RUN_NAME + "_regressionPosition")

    for file in [NPZ_FILE_TRAIN, NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        # load dataset
        data_train = datasets.DatasetCluster(
            npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)
        if file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
            data_train.p_train = 0.0
            data_train.p_valid = 0.0
            data_train.p_test = 1.0

        # set normalization from training dataset
        data_train.standardize(norm_mean, norm_std, 10)
        data_train.update_targets_position()
        data_train.update_indexing_positives()

        y_pred = m_regP.predict(data_train.x_test())
        y_true = data_train.y_test()

        evaluation.eval_regression_position(y_pred, y_true)

if eval_regression_theta:
    os.chdir(dir_results + RUN_NAME + "/")
    m_regT = DNN_SXAX.setupModel(nCluster=10,
                                 nOutput=1,
                                 dropout=dropout,
                                 nNodes=nConnectedNodes,
                                 activation="relu",
                                 output_activation="linear",
                                 loss="regression")

    m_regT = DNN_SXAX.load_model(m_regT, RUN_NAME + "_regressionTheta")
    norm_mean, norm_std = DNN_SXAX.load_norm(RUN_NAME + "_regressionTheta")

    for file in [NPZ_FILE_TRAIN, NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        # load dataset
        data_train = datasets.DatasetCluster(
            npz_file=dir_datasets + NPZ_FILE_TRAIN[:-4] + "/" + NPZ_FILE_TRAIN)
        if file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
            data_train.p_train = 0.0
            data_train.p_valid = 0.0
            data_train.p_test = 1.0

        # set normalization from training dataset
        data_train.standardize(norm_mean, norm_std, 10)
        data_train.update_targets_theta()
        data_train.update_indexing_positives()

        y_pred = m_regT.predict(data_train.x_test())
        y_true = data_train.y_test()

        evaluation.eval_regression_theta(y_pred, y_true)
