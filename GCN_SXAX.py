"""
Dynamic Graph Neural Networks for event classification and event topology
regression for the SiFiCC-project.
This Network operates on the Base setting: X scatterer cluster and
X absorber cluster (X > 0) per event.
Graphs are created using the spektral package

"""

import os
import numpy as np

from src.SiFiCCNN.GCN import SiFiCCdatasets, IGSiFICCCluster
from src.SiFiCCNN.GCN import Spektral_NeuralNetwork

from spektral.data.loaders import DisjointLoader

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout
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
class_wts = tf.cast(class_wts, dtype=tf.float32)

# ------------------------------------------------------------------------------
# Build Model

X_in = Input(shape=(F,))
A_in = Input(shape=(None,), sparse=True)
E_in = Input(shape=(S,))
I_in = Input(shape=(), dtype=tf.int64)

X_1 = GCNConv(32, activation="relu")([X_in, A_in])
X_2 = GCNConv(32, activation="relu")([X_1, A_in])
X_3 = GlobalSumPool()([X_2, I_in])
output = Dense(n_out, activation="sigmoid")(X_3)

model = Model(inputs=[X_in, A_in, E_in, I_in], outputs=output)
optimizer = Adam(learning_rate)
loss_fn = BinaryCrossentropy()


# ------------------------------------------------------------------------------
# Fit model

def weighted_binary_crossentropy(labels, predictions, weights):
    labels = tf.cast(labels, tf.int32)
    loss = loss_fn(labels, predictions) + sum(model.losses)
    class_weights = tf.gather(weights, labels)
    return tf.reduce_mean(class_weights * loss)


@tf.function(input_signature=loader_tr.tf_signature(),
             experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = weighted_binary_crossentropy(target, predictions, class_wts)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(binary_accuracy(target, predictions))
    eff = 0  # tf.reduce_mean(Recall(target, predictions))
    pur = 0  # tf.reduce_mean(Precision(target, predictions))
    return loss, acc, eff, pur


def evaluate(loader):
    output = []
    step = 0
    for batch in loader:
        step += 1
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss = loss_fn(target, predictions)
        acc = tf.reduce_mean(binary_accuracy(target, predictions))
        output.append(
            (loss, acc, 0, 0, len(target)))  # Keep track of batch size
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


epoch = step = 0
results = []
for batch in loader_tr:
    step += 1

    # Training step
    inputs, target = batch

    loss, acc, eff, pur = train_step(inputs, target)
    results.append((loss, acc, eff, pur, len(target)))
    if step == loader_tr.steps_per_epoch:
        results_va = evaluate(loader_va)
        results = np.array(results)
        results = np.average(results[:, :-1], 0, weights=results[:, -1])

        step = 0
        epoch += 1

        print("Ep. {} - Loss: {:.3f} - Acc: {:.3f} |"
              "Test Loss: {:.3f} - Test Acc: {:.3f} |".format(
            epoch, *results[:2], *results_va[:2])
        )

        results = []

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
