"""
Dynamic Graph Neural Networks for event classification and event topology
regression for the SiFiCC-project.
This Network operates on the Base setting: X scatterer cluster and
X absorber cluster (X > 0) per event.
Graphs are created using the spektral package

"""

import os
import numpy as np
import sys
import spektral

from src.SiFiCCNN.GCN import SiFiCCdatasets, IGSiFICCCluster
from src.SiFiCCNN.models import GCN
from src.SiFiCCNN.analysis import evaluation
from src.SiFiCCNN.plotting import plt_history


from spektral.data.loaders import DisjointLoader
import tensorflow as tf

################################################################################
# Global settings
################################################################################

RUN_NAME = "GCN_example"

train_classifier = True
train_regression_energy = False
train_regression_position = False
train_regression_theta = False

eval_classifier = False
eval_regression_energy = False
eval_regression_position = False
eval_regression_theta = False

generate_datasets = False

################################################################################
# Datasets
################################################################################

DATASET = "SiFiCCCluster"
DATASET_TRAIN = "OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster"

################################################################################
# Set paths
################################################################################
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_datasets = dir_main + "/datasets/"
dir_results = dir_main + "/results/"

# create subdirectory for run output
if not os.path.isdir(dir_results + RUN_NAME + "/"):
    os.mkdir(dir_results + RUN_NAME + "/")
for file in [DATASET_TRAIN]:
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file[:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

################################################################################
# Load dataset , custom setting possible
################################################################################

if generate_datasets:
    from src.SiFiCCNN.root import RootFiles
    from src.SiFiCCNN.root import Root

    rootparser_cont = Root.Root(
        dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)
    IGSiFICCCluster.gen_SiFiCCCluster(rootparser_cont, n=None)

    sys.exit()

################################################################################
# Build Model
################################################################################


################################################################################
# Training
################################################################################


dropout = 0.2
learning_rate = 1e-3
nConnectedNodes = 16
nFilter = 16
batch_size = 64
activation = "relu"

trainsplit = 0.7
valsplit = 0.2
nEpochs = 50

# model version 1
modelParameters = {"dropout": dropout,
                   "learning_rate": learning_rate,
                   "nFilter": nFilter,
                   "activation": activation}

l_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(GCN.lr_scheduler)]

if train_classifier:
    dataset = SiFiCCdatasets.SiFiCCdatasets(
        name="OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster",
        edge_atr=False,
        adj_arg="binary",
        dataset_path=dir_datasets)

    # Train/test split
    idx1 = int(trainsplit * len(dataset))
    idx2 = int((trainsplit + valsplit) * len(dataset))
    dataset_tr = dataset[:idx1]
    dataset_va = dataset[idx1:idx2]
    dataset_te = dataset[idx2:]

    loader_train = DisjointLoader(dataset_tr,
                                  batch_size=batch_size,
                                  epochs=nEpochs)
    loader_valid = DisjointLoader(dataset_va,
                                  batch_size=batch_size)

    # create class weight dictionary
    class_weights = dataset.get_classweight_dict()
    class_wts = tf.constant([class_weights[0], class_weights[1]])

    m_clas = GCN.setupModel(**modelParameters)
    history = m_clas.fit(loader_train,
                         epochs=nEpochs,
                         steps_per_epoch=loader_train.steps_per_epoch,
                         validation_data=loader_valid,
                         validation_steps=loader_valid.steps_per_epoch,
                         class_weight=class_weights,
                         verbose=1)

    loader_test = DisjointLoader(dataset_te,
                                 batch_size=batch_size,
                                 epochs=1)

    # predict test dataset
    os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

    y_true = []
    y_scores = []
    for batch in loader_test:
        inputs, target = batch
        p = m_clas(inputs, training=False)
        y_true.append(target)
        y_scores.append(p.numpy())

    y_true = np.vstack(y_true)
    y_scores = np.vstack(y_scores)
    y_true = np.reshape(y_true, newshape=(y_true.shape[0],))
    y_scores = np.reshape(y_scores, newshape=(y_scores.shape[0],))

    evaluation.eval_classifier(y_scores=y_scores,
                               y_true=y_true,
                               theta=0.5)


################################################################################
# Evaluate model
################################################################################
"""

from src.SiFiCCNN.NeuralNetwork import NNAnalysis
from src.SiFiCCNN.NeuralNetwork import FastROCAUC

from src.SiFiCCNN.Plotter import PTClassifier

for file in [DATASET_TRAIN]:
    os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

    y_true = []
    y_pred = []
    for batch in loader_test:
        inputs, target = batch
        p = model(inputs, training=False)
        y_true.append(target)
        y_pred.append(p.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_true = np.reshape(y_true, newshape=(y_true.shape[0],))
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0],))

    # ROC-AUC Analysis
    FastROCAUC.fastROCAUC(y_pred, y_true, save_fig="ROCAUC")
    _, theta_opt = FastROCAUC.fastROCAUC(y_pred, y_true, return_score=True)

    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    PTClassifier.plot_score_distribution(y_pred, y_true, "score_dist")

    # write general binary classifier metrics into console and .txt file
    NNAnalysis.write_metrics_classifier(y_pred, y_true)
"""