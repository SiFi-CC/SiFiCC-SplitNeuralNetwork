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
"""
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
    eIn = tf.keras.layers.Input(shape=(S, ))
    iIn = tf.keras.layers.Input(shape=(), dtype=tf.int64)

    x = GCNConv(nFilter, activation=activation, use_bias=True)([xIn, aIn])
    x = GlobalSumPool()([x, iIn])
    x = tf.keras.layers.Flatten()(x)

    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)

    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Build model
    model = tf.keras.models.Model(inputs=[xIn, aIn, eIn, iIn], outputs=output)
    optimizer = tf.keras.optimizer.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    return model
"""


# ------------------------------------------------------------------------------
# Model version 2

class Net(tf.keras.models.Model):
    def __init__(self, channels, dropout):
        super().__init__()

        self.gcn = GCNConv(channels)
        self.pool = GlobalSumPool()
        self.dense1 = tf.keras.layers.Dense(channels, activation="relu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        x, a, i = inputs

        x = self.gcn([x, a])
        x = self.pool([x, i])
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


# ------------------------------------------------------------------------------
# model version 3


class GCN(tf.keras.Model):
    """
    This model, with its default hyperparameters, implements the architecture
    from the paper:
    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling
    **Mode**: single, disjoint, mixed, batch.
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`
    - Weighted adjacency matrix of shape `([batch], n_nodes, n_nodes)`
    **Output**
    - Softmax predictions with shape `([batch], n_nodes, n_labels)`.
    **Arguments**
    - `n_labels`: number of channels in output;
    - `channels`: number of channels in first GCNConv layer;
    - `activation`: activation of the first GCNConv layer;
    - `output_activation`: activation of the second GCNConv layer;
    - `use_bias`: whether to add a learnable bias to the two GCNConv layers;
    - `dropout_rate`: `rate` used in `Dropout` layers;
    - `l2_reg`: l2 regularization strength;
    - `**kwargs`: passed to `Model.__init__`.
    """

    def __init__(
        self,
        n_labels,
        channels=16,
        activation="relu",
        output_activation="softmax",
        use_bias=False,
        dropout_rate=0.5,
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
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = GCNConv(
            n_labels, activation=output_activation, use_bias=use_bias
        )

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader

        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])

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
nEpochs = 10

"""
# model version 1
modelParameters = {"dropout": dropout,
                   "learning_rate": learning_rate,
                   "nFilter": nFilter,
                   "activation": activation}

model = setupModel(**modelParameters)
"""

# model version 2
# model = Net(32, 0.2)
model = GCN(1, channels=32, use_bias=True, output_activation="sigmoid")
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="binary_crossentropy")

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
