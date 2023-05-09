"""
Dynamic Graph Neural Networks for event classification and event topology
regression for the SiFiCC-project.
This Network operates on the Base setting: X scatterer cluster and
X absorber cluster (X > 0) per event.
Graphs are created using the spektral package

"""

import os
import numpy as np
import spektral.utils

from src.SiFiCCNN.GCN import SiFiCCdatasets, IGSiFICCCluster
from src.SiFiCCNN.GCN import Spektral_NeuralNetwork

from spektral.data.loaders import DisjointLoader

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.losses import BinaryCrossentropy
from keras.metrics import Recall, Precision, binary_accuracy
from keras.optimizers import Adam
from spektral.layers import GlobalSumPool, ECCConv, GCNConv

# ------------------------------------------------------------------------------
# Global settings

RUN_NAME = "GCN_SXAX"

DATASET = "SiFiCCCluster"
DATASET_TRAIN = "OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster"

# Neural Network settings
learning_rate = 1e-3  # Learning rate
epochs = 10  # Number of training epochs
batch_size = 32  # Batch size

gen_datasets = False

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
    IGSiFICCCluster.gen_SiFiCCCluster(rootparser_cont, n=100000)

dataset = SiFiCCdatasets.SiFiCCdatasets(
    name="OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster",
    dataset_path=dir_datasets)

# Parameters
F = dataset.n_node_features
S = dataset.n_edge_features
n_out = dataset.n_labels

# Train/test split
idx1 = int(0.7 * len(dataset))
idx2 = int(0.8 * len(dataset))
dataset_tr = dataset[:idx1]
dataset_va = dataset[idx1:idx2]
dataset_te = dataset[idx2:]

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(dataset_va, batch_size=batch_size)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size, epochs=1)

# create class weight dictionary
class_weights = dataset.get_classweight_dict()
class_wts = tf.constant([class_weights[0], class_weights[1]])


# ------------------------------------------------------------------------------
# Build Model

class Net(Model):

    def __init__(
            self,
            n_labels,
            channels=16,
            activation="relu",
            output_activation="sigmoid",
            use_bias=True,
            dropout_rate=0.2,
            l2_reg=2.5e-4,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        reg = tf.keras.regularizers.l2(l2_reg)

        self._gcn0 = GCNConv(
            channels,
            activation=activation,
            kernel_regularizer=reg,
            use_bias=use_bias
        )

        self._drop = tf.keras.layers.Dropout(dropout_rate)
        self._pooling = GlobalSumPool()
        self._flatten = Flatten()
        self._dense1 = Dense(
            channels,
            activation=self.activation,
        )
        self._dense2 = Dense(
            1,
            activation=self.output_activation
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs

        a = spektral.utils.add_self_loops(a, value=1)
        a = spektral.utils.gcn_filter(a, symmetric=True)

        x = self._gcn0([x, a])
        x = self._flatten(x)
        x = self._dense1(x)
        x = self._dense2(x)
        return x


model = Net(n_labels=1)
optimizer = Adam(learning_rate)
pur = Precision
eff = Recall
loss_fn = BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=[eff, pur])

# ------------------------------------------------------------------------------
# Fit model

model.fit(loader_tr,
          epochs=epochs,
          steps_per_epoch=loader_tr.steps_per_epoch,
          validation_data=loader_va,
          validation_steps=loader_va.steps_per_epoch,
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
    for batch in loader_te:
        inputs, target = batch
        p = model(inputs, training=False)
        y_true.append(target)
        y_pred.append(p.numpy())

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_true = np.reshape(y_true, newshape=(y_true.shape[0],))
    y_pred = np.reshape(y_pred, newshape=(y_pred.shape[0],))

    model_loss = loss_fn(y_true, y_pred)

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
