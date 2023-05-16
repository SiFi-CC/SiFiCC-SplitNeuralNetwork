import numpy as np
import os
import sys
import tensorflow as tf

from SiFiCCNN.root import Root
from SiFiCCNN.models.DNNCluster import dataset, downloader, model
from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.plotting import plt_models

################################################################################
# Global settings
################################################################################

RUN_NAME = "DNNCluster_S4A6"

train_clas = False
train_regE = False
train_regP = False
train_regT = False

eval_clas = True
eval_regE = False
eval_regP = False
eval_regT = False

generate_datasets = False

# Neural Network settings
dropout = 0.2
learning_rate = 1e-3
nConnectedNodes = 64
batch_size = 64
nEpochs = 20

l_callbacks = [tf.keras.callbacks.LearningRateScheduler(model.lr_scheduler)]

################################################################################
# Datasets
################################################################################

# root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"
# Training file used for classification and regression training
# Generated via an input generator, contain one Bragg-peak position
DATASET_CONT = "DenseCluster_S4A6_OptimisedGeometry_Continuous_2e10protons"
DATASET_0MM = "DenseCluster_S4A6_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
DATASET_5MM = "DenseCluster_S4A6_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

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
for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file[:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

################################################################################
# Load dataset , custom setting possible
################################################################################

if generate_datasets:
    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(dir_root + file)
        downloader.load(root, n=None)
    sys.exit()

################################################################################
# Training/Evaluation Classifier model
################################################################################

if train_clas:
    data = dataset.DenseCluster(DATASET_CONT)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    model_clas = model.setupModel(nCluster=10,
                                  nOutput=1,
                                  dropout=dropout,
                                  nNodes=nConnectedNodes,
                                  activation="relu",
                                  output_activation="sigmoid")
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    # set normalization from training dataset
    norm_mean, norm_std = data.get_standardization(10, 10)
    data.standardize(norm_mean, norm_std, 10)
    class_weights = data.get_classweights()

    history = model_clas.fit(data.x_train(),
                             data.y_train(),
                             validation_data=(data.x_valid(), data.y_valid()),
                             epochs=nEpochs,
                             batch_size=batch_size,
                             class_weight=class_weights,
                             verbose=1,
                             callbacks=[l_callbacks])

    plt_models.plot_history_classifier(history.history,
                                       RUN_NAME + "_history_classifier")

    # save model
    model.save_model(model_clas, RUN_NAME + "_classifier")
    model.save_history(RUN_NAME + "_classifier", history.history)
    model.save_norm(RUN_NAME + "_classifier", norm_mean, norm_std)

if eval_clas:
    os.chdir(dir_results + RUN_NAME + "/")
    model_clas = model.setupModel(nCluster=10,
                                  nOutput=1,
                                  dropout=dropout,
                                  nNodes=nConnectedNodes,
                                  activation="relu",
                                  output_activation="sigmoid")

    model_clas = model.load_model(model_clas, RUN_NAME + "_classifier")
    norm_mean, norm_std = model.load_norm(RUN_NAME + "_classifier")

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        # load dataset
        data = dataset.DenseCluster(DATASET_CONT)

        if file in [DATASET_0MM, DATASET_5MM]:
            data.p_train = 0.0
            data.p_valid = 0.0
            data.p_test = 1.0

        # set normalization from training dataset
        data.standardize(norm_mean, norm_std, 10)

        y_scores = model_clas.predict(data.x_test())
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
