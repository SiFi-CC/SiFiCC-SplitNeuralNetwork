"""
Dynamic Graph Neural Networks for event classification and event topology
regression for the SiFiCC-project.
This Network operates on the Base setting: X scatterer cluster and
X absorber cluster (X > 0) per event.
Graphs are created using the spektral package

"""

import os
import numpy as np
import spektral


from src.SiFiCCNN.GCN import SiFiCCdatasets, IGSiFICCCluster
from src.SiFiCCNN.GCN import Spektral_NeuralNetwork

from spektral.data.loaders import DisjointLoader

import tensorflow as tf
from spektral.layers import GlobalSumPool, ECCConv, GCNConv

# ------------------------------------------------------------------------------
# Global settings

RUN_NAME = "GCN_SXAX"

DATASET = "SiFiCCCluster"
DATASET_TRAIN = "OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster"

gen_datasets = True

# ------------------------------------------------------------------------------
# Set paths, check for datasets

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

# ------------------------------------------------------------------------------
# Load dataset, generate Datasets if needed
# Create a disjoint loader

if gen_datasets:
    from src.SiFiCCNN.Root import RootFiles
    from src.SiFiCCNN.Root import RootParser

    rootparser_cont = RootParser.Root(
        dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)
    IGSiFICCCluster.gen_SiFiCCCluster(rootparser_cont, n=10000)

dataset = SiFiCCdatasets.SiFiCCdatasets(
    name="OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster",
    dataset_path=dir_datasets)


# ------------------------------------------------------------------------------
# Custom adj layer from 3B deeplearning wiki


class ConcatAdj(tf.keras.layers.Layer):
    def __init__(self, adj, **kwargs):
        super(ConcatAdj, self).__init__(**kwargs)
        self.adj = adj

        # Set the tf tensor
        adj = spektral.utils.gcn_filter(adj)
        self.adj_tensor = tf.constant(adj)

    def call(self, input):
        return [input, self.adj_tensor]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config,
                "adj": self.adj}


# ------------------------------------------------------------------------------
# Build Model

def setupModel(dropout,
               learning_rate,
               nFilter=32,
               activation="relu"):
    # original feature dimensionality
    F = 10
    # Model definition
    xIn = tf.keras.layer.Input(shape=(F,))
    aIn = tf.keras.layer.Input(shape=(None,), sparse=True)
    eIn = tf.keras.layers.Input(shape=())
    iIn = tf.keras.layer.Input(shape=(), dtype=tf.int64)

    x = GCNConv(nFilter, activation=activation, use_bias=True)([xIn, aIn])
    x = GlobalSumPool([x, iIn])
    x = tf.keras.layer.Flatten()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Build model
    model = tf.keras.models.Model(input=[xIn, aIn, eIn, iIn])
    optimizer = tf.keras.optimizer.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    return model


# ------------------------------------------------------------------------------
# Create model and train

dropout = 0.2
learning_rate = 1e-3
nConnectedNodes = 32
nFilter = 32
batch_size = 64
activation = "relu"

trainsplit = 0.8
valsplit = 0.1
nEpochs = 60

modelParameters = {"dropout": dropout,
                   "learning_rate": learning_rate,
                   "nFilter": nFilter,
                   "activation": activation}

model = setupModel(**modelParameters)
model.summary()

# generator setup
idx1 = int(trainsplit * len(dataset))
idx2 = int((trainsplit + valsplit) * len(dataset))
dataset_tr = dataset[:idx1]
dataset_va = dataset[idx1:idx2]
dataset_te = dataset[idx2:]

loader_train = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=nEpochs)
loader_valid = DisjointLoader(dataset_va, batch_size=batch_size)
loader_test = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

# create class weight dictionary
class_weights = dataset.get_classweight_dict()

model.fit(loader_train,
          epochs=nEpochs,
          steps_per_epoch=loader_train.steps_per_epoch,
          validation_data=loader_valid,
          validation_steps=loader_valid.steps_per_epoch,
          class_weight=class_weights,
          verbose=1)

# ------------------------------------------------------------------------------
# Evaluate Network

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

    for i in range(20):
        print(y_true[i], y_pred[i])

    # ROC-AUC Analysis
    FastROCAUC.fastROCAUC(y_pred, y_true, save_fig="ROCAUC")
    _, theta_opt = FastROCAUC.fastROCAUC(y_pred, y_true, return_score=True)

    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    PTClassifier.plot_score_distribution(y_pred, y_true, "score_dist")

    # write general binary classifier metrics into console and .txt file
    NNAnalysis.write_metrics_classifier(y_pred, y_true)
