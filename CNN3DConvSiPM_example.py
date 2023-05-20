import numpy as np
import os
import sys
import tensorflow as tf

from SiFiCCNN.root import Root
from SiFiCCNN.models.CNN3DConvSiPM import dataset, downloader, model
from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.plotting import plt_models

################################################################################
# Global settings
################################################################################

RUN_NAME = "CNN3DConvSiPM"

train_clas = True
train_regE = False
train_regP = False
train_regT = False

eval_clas = False
eval_regE = False
eval_regP = False
eval_regT = False

generate_datasets = False

# Neural Network settings
dropout = 0.0
learning_rate = 1e-3
nConnectedNodes = 64
batch_size = 64
nEpochs = 20

l_callbacks = [tf.keras.callbacks.LearningRateScheduler(model.lr_scheduler)]

################################################################################
# Datasets
################################################################################

# root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "FinalDetectorVersion_RasterCoupling_OPM_38e8protons.root"
ROOT_FILE_BP5mm = "FinalDetectorVersion_RasterCoupling_OPM_BP5mm_4e9protons.root"
ROOT_FILE_CONT = "FinalDetectorVersion_RasterCoupling_OPM_Continuous_2e10protons.root"
# Training file used for classification and regression training
# Generated via an input generator, contain one Bragg-peak position
DATASET_CONT = "DenseSiPM_FinalDetectorVersion_RasterCoupling_OPM_Continuous_2e10protons"
DATASET_0MM = "DenseSiPM_FinalDetectorVersion_RasterCoupling_OPM_38e8protons"
DATASET_5MM = "DenseSiPM_FinalDetectorVersion_RasterCoupling_OPM_BP5mm_4e9protons"

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
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file + "/")

################################################################################
# Load dataset , custom setting possible
################################################################################

if generate_datasets:
    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(dir_root + file)
        downloader.load(root, n=1000000)
    sys.exit()

################################################################################
# Training/Evaluation Classifier model
################################################################################

if train_clas:
    # setup generator for training
    loader_train = dataset.DenseSiPM(name=DATASET_CONT,
                                     batch_size=batch_size,
                                     slicing="train",
                                     shuffle=True)

    loader_valid = dataset.DenseSiPM(name=DATASET_CONT,
                                     batch_size=batch_size,
                                     slicing="valid",
                                     shuffle=True)

    os.chdir(dir_results + RUN_NAME + "/")
    # classifier model
    model_clas = model.setupModel(nOutput=1,
                                  dropout=dropout,
                                  output_activation="sigmoid")
    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    history = model_clas.fit(loader_train,
                             epochs=nEpochs,
                             steps_per_epoch=loader_train.steps_per_epoch,
                             validation_data=loader_valid,
                             validation_steps=loader_valid.steps_per_epoch,
                             verbose=1,
                             callbacks=[l_callbacks])

    plt_models.plot_history_classifier(history.history,
                                       RUN_NAME + "_history_classifier")

    # save model
    model.save_model(model_clas, RUN_NAME + "_classifier")
    model.save_history(RUN_NAME + "_classifier", history.history)

if eval_clas:
    os.chdir(dir_results + RUN_NAME + "/")
    model_clas = model.setupModel(nOutput=1,
                                  dropout=dropout,
                                  output_activation="sigmoid")
    model_clas = model.load_model(model_clas, RUN_NAME + "_classifier")

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file + "/")

        loader_eval = dataset.DenseSiPM(name=DATASET_CONT,
                                        batch_size=batch_size,
                                        slicing="",
                                        shuffle=True)

        y_scores = model_clas.predict(loader_eval)
        y_true = loader_eval.y()
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
