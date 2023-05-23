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
import tensorflow as tf

from spektral.data.loaders import DisjointLoader
from spektral.transforms import GCNFilter

from SiFiCCNN.models.GCNCluster import dataset, downloader, model
from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.plotting import plt_models

################################################################################
# Global settings
################################################################################

RUN_NAME = "GCNCluster"

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
dropout = 0.1
learning_rate = 1e-3
nConnectedNodes = 64
batch_size = 64
nEpochs = 5
trainsplit = 0.7
valsplit = 0.1

l_callbacks = [tf.keras.callbacks.LearningRateScheduler(model.lr_scheduler)]

################################################################################
# Datasets
################################################################################

# root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"

DATASET_CONT = "GraphCluster_OptimisedGeometry_Continuous_2e10protons"
DATASET_0MM = "GraphCluster_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
DATASET_5MM = "GraphCluster_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

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
for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file + "/")

################################################################################
# Load dataset , custom setting possible
################################################################################

if generate_datasets:
    from SiFiCCNN.root import Root

    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(dir_root + file)
        downloader.load(root, n=1000000)
    sys.exit()

################################################################################
# Custom GCN model
################################################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool


class GCNmodel(Model):

    def __init__(self,
                 n_labels,
                 output_activation,
                 dropout=0.0):
        super().__init__()
        self.graph_conv = GCNConv(32)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(dropout)
        self.dense = Dense(n_labels, output_activation)

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out


################################################################################
# Training
################################################################################

if train_clas:
    data = dataset.GraphCluster(
        name=DATASET_CONT,
        edge_atr=False,
        adj_arg="binary")
    data.apply(GCNFilter())

    os.chdir(dir_results + RUN_NAME + "/")

    # Train/test split
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

    # create class weight dictionary
    class_weights = data.get_classweight_dict()
    print("# CLass weights: ")
    print(class_weights)

    model_clas = GCNmodel(n_labels=1,
                          output_activation="sigmoid",
                          dropout=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=list_metrics)

    history = model_clas.fit(loader_train,
                             epochs=nEpochs,
                             steps_per_epoch=loader_train.steps_per_epoch,
                             validation_data=loader_valid,
                             validation_steps=loader_valid.steps_per_epoch,
                             class_weight=class_weights,
                             verbose=1)

    plt_models.plot_history_classifier(history.history,
                                       RUN_NAME + "_history_classifier")

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        data = dataset.GraphCluster(
            name=file,
            edge_atr=False,
            adj_arg="binary")
        data.apply(GCNFilter())

        os.chdir(dir_results + RUN_NAME + "/")
        loader_test = DisjointLoader(data,
                                     batch_size=batch_size,
                                     epochs=1)

        # predict test dataset
        os.chdir(dir_results + RUN_NAME + "/" + file + "/")

        y_true = []
        y_scores = []
        for batch in loader_test:
            inputs, target = batch
            p = model_clas(inputs, training=False)
            y_true.append(target)
            y_scores.append(p.numpy())

        y_true = np.vstack(y_true)
        y_scores = np.vstack(y_scores)
        y_true = np.reshape(y_true, newshape=(y_true.shape[0],))
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
