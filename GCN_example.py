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
from src.SiFiCCNN.GCN.dl_layers import ConcatAdj, ReZero, GCNConvResNetBlock

from spektral.data.loaders import DisjointLoader

import tensorflow as tf
from spektral.layers import GlobalMaxPool, ECCConv, GCNConv

# ------------------------------------------------------------------------------
# Global settings

RUN_NAME = "GCN_example"

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

dataset = SiFiCCdatasets.SiFiCCdatasets(
    name="OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster",
    edge_atr=False,
    adj_arg="gcn_distance",
    dataset_path=dir_datasets)


################################################################################
# Build Model
################################################################################

def setupModel(dropout,
               learning_rate,
               nFilter=32,
               activation="relu"):
    # original feature dimensionality
    F = 10
    S = 3
    # Model definition
    xIn = tf.keras.layers.Input(shape=(F,))
    aIn = tf.keras.layers.Input(shape=(None,), sparse=True)
    iIn = tf.keras.layers.Input(shape=(), dtype=tf.int64)

    x = GCNConv(nFilter, activation=activation, use_bias=True)([xIn, aIn])
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GCNConvResNetBlock(*[x, aIn], nFilter, activation)
    x = GlobalMaxPool()([x, iIn])
    x = tf.keras.layers.Flatten()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(nFilter, activation="relu")(x)
    x = tf.keras.layers.Dense(nFilter, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Build model
    model = tf.keras.models.Model(inputs=[xIn, aIn, iIn], outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=["Precision", "Recall"])

    return model


################################################################################
# Training
################################################################################


dropout = 0.2
learning_rate = 1e-3
nConnectedNodes = 32
nFilter = 32
batch_size = 64
activation = "relu"

trainsplit = 0.7
valsplit = 0.2
nEpochs = 20

# model version 1
modelParameters = {"dropout": dropout,
                   "learning_rate": learning_rate,
                   "nFilter": nFilter,
                   "activation": activation}

model = setupModel(**modelParameters)

# generator setup
idx1 = int(trainsplit * len(dataset))
idx2 = int((trainsplit + valsplit) * len(dataset))
dataset_tr = dataset[:idx1]
dataset_va = dataset[idx1:idx2]
dataset_te = dataset[idx2:]

loader_train = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=nEpochs)
loader_valid = DisjointLoader(dataset_va, batch_size=batch_size)
loader_test = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

print(loader_train.tf_signature())

# create class weight dictionary
class_weights = dataset.get_classweight_dict()

model.fit(loader_train,
          epochs=nEpochs,
          steps_per_epoch=loader_train.steps_per_epoch,
          validation_data=loader_valid,
          validation_steps=loader_valid.steps_per_epoch,
          class_weight=class_weights,
          verbose=1)

################################################################################
# Evaluate model
################################################################################


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
