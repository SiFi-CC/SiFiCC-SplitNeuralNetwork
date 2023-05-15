"""
This example implements the model from the paper
    > [Design Space for Graph Neural Networks](https://arxiv.org/abs/2011.08843)<br>
    > Jiaxuan You, Rex Ying, Jure Leskovec

using the SiFiCC-Cluster dataset.
The configuration at the top of the file is the best one identified in the
paper, and should work well for many different datasets without changes.
Note: the results reported in the paper are averaged over 3 random repetitions
with an 80/20 split.
"""

import os
import numpy as np
import sys
import spektral.utils

from src.SiFiCCNN.GCN import SiFiCCdatasets, IGSiFICCCluster
from src.SiFiCCNN.models import DNN_SXAX, GNN_SXAX
from src.SiFiCCNN.analysis import evaluation
from src.SiFiCCNN.plotting import plt_history

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN

import tensorflow as tf
from tensorflow import keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.metrics import binary_accuracy

################################################################################
# Global settings
################################################################################

RUN_NAME = "GeneralGNN_master"

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
# Set paths, check for datasets
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
# Load dataset, generate Datasets if needed
################################################################################

if generate_datasets:
    from src.SiFiCCNN.root import RootFiles
    from src.SiFiCCNN.root import Root

    rootparser_cont = Root.Root(
        dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)
    IGSiFICCCluster.gen_SiFiCCCluster(rootparser_cont, n=None)

    sys.exit()

################################################################################
# Training
################################################################################

batch_size = 32
learning_rate = 0.01
nEpochs = 50

trainsplit = 0.6
valsplit = 0.2

l_callbacks = [
    tf.keras.callbacks.LearningRateScheduler(DNN_SXAX.lr_scheduler)]

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

    m_clas = GeneralGNN(dataset.n_labels, activation="sigmoid")
    optimizer = Adam(learning_rate)
    loss_fn = BinaryCrossentropy()

    m_clas.compile(optimizer=optimizer,
                   loss=loss_fn,
                   metrics=["Precision", "Recall"])

    history = m_clas.fit(loader_train,
                         epochs=nEpochs,
                         steps_per_epoch=loader_train.steps_per_epoch,
                         validation_data=loader_valid,
                         validation_steps=loader_valid.steps_per_epoch,
                         class_weight=class_weights,
                         verbose=1,
                         callbacks=[l_callbacks])

    plt_history.plot_history_classifier(history.history,
                                        RUN_NAME + "_history_classifier")

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

    # save model
    GNN_SXAX.save_model(m_clas, RUN_NAME + "_classifier")
    GNN_SXAX.save_history(RUN_NAME + "_classifier", history.history)

################################################################################
# Fit model
################################################################################
"""
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
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(binary_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


epoch = step = 0
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1
        results_va = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                epoch, *np.mean(results, 0), *results_va
            )
        )
        results = []
"""
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
"""

if eval_classifier:

    os.chdir(dir_results + RUN_NAME + "/")
    m_clas = GeneralGNN(1, activation="sigmoid")
    m_clas = DNN_SXAX.load_model(m_clas, RUN_NAME + "_classifier")

    for file in [DATASET_TRAIN]:
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
