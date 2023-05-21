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
import tensorflow as tf

from spektral.data.loaders import DisjointLoader
from spektral.models import GeneralGNN

from SiFiCCNN.root import Root
from SiFiCCNN.models.GCNCluster import dataset, downloader, model
from SiFiCCNN.analysis import fastROCAUC, metrics
from SiFiCCNN.plotting import plt_models

################################################################################
# Global settings
################################################################################

RUN_NAME = "GeneralGNNCluster"

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
dropout = 0.1
learning_rate = 1e-3
nConnectedNodes = 64
batch_size = 64
nEpochs = 1
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
DATASET_0MM = "GraphCluster_S4A6_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
DATASET_5MM = "GraphCluster_S4A6_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

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
# Training
################################################################################


if train_clas:
    dataset = dataset.GraphCluster(
        name=DATASET_CONT,
        edge_atr=False,
        adj_arg="binary")

    os.chdir(dir_results + RUN_NAME + "/")

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

    model_clas = GeneralGNN(dataset.n_labels, activation="sigmoid")
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
                             class_weight=class_weights,
                             verbose=1,
                             callbacks=[l_callbacks])

    plt_models.plot_history_classifier(history.history,
                                       RUN_NAME + "_history_classifier")

    loader_test = DisjointLoader(dataset_te,
                                 batch_size=batch_size,
                                 epochs=1)

    # save model
    model.save_model(model_clas, RUN_NAME + "_classifier")
    model.save_history(RUN_NAME + "_classifier", history.history)

    """
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
    """

if eval_clas:
    os.chdir(dir_results + RUN_NAME + "/")
    model_clas = GeneralGNN(1, activation="sigmoid")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)
    model_clas.build()
    model_clas.load_weights(RUN_NAME + "_classifier" + ".h5")


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
"""