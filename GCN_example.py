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

####################################################################################################
# Global settings
####################################################################################################

RUN_NAME = "GCNCluster"

train_clas = False
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
nEpochs = 20
trainsplit = 0.7
valsplit = 0.1

l_callbacks = [tf.keras.callbacks.LearningRateScheduler(model.lr_scheduler)]

####################################################################################################
# Datasets
####################################################################################################

# root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps.root"
ROOT_FILE_CONT = "OptimisedGeometry_Continuous_2e10protons.root"

DATASET_CONT = "GraphCluster_OptimisedGeometry_Continuous_2e10protons"
DATASET_0MM = "GraphCluster_OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
DATASET_5MM = "GraphCluster_OptimisedGeometry_BP5mm_4e9protons_withTimestamps"

####################################################################################################
# Set paths
####################################################################################################
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

####################################################################################################
# Load dataset , custom setting possible
####################################################################################################

if generate_datasets:
    from SiFiCCNN.root import Root

    for file in [ROOT_FILE_CONT, ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root = Root.Root(dir_root + file)
        downloader.load(root,
                        path=dir_datasets,
                        n=100000)
    sys.exit()
"""
####################################################################################################
# Custom GCN model
####################################################################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from spektral.layers import GCNConv, ECCConv, GlobalSumPool


class GCNmodel(Model):

    def __init__(self,
                 n_labels,
                 output_activation,
                 dropout=0.0):
        super().__init__()

        self.n_labels = n_labels
        self.output_activation = output_activation
        self.dropout_val = dropout

        self.graph_gcnconv1 = GCNConv(32, activation="relu")
        self.graph_gcnconv2 = GCNConv(64, activation="relu")
        self.graph_eccconv1 = ECCConv(32, activation="relu")
        self.graph_eccconv2 = ECCConv(64, activation="relu")
        self.pool = GlobalSumPool()
        self.dropout = Dropout(dropout)
        self.dense1 = Dense(64, activation="relu")
        self.dense_out = Dense(n_labels, output_activation)
        self.concatenate = Concatenate()

    def call(self, inputs):
        xIn, aIn, eIn, iIn = inputs
        out1 = self.graph_gcnconv1([xIn, aIn])
        out2 = self.graph_gcnconv2([out1, aIn])
        out3 = self.graph_eccconv1([xIn, aIn, eIn])
        out4 = self.graph_eccconv2([out3, aIn, eIn])
        out5 = self.pool([out2, iIn])
        out6 = self.pool([out4, iIn])

        out7 = self.concatenate([out5, out6])
        out8 = self.dense1(out7)
        out9 = self.dropout(out8)
        out_final = self.dense_out(out9)

        return out_final

    def get_config(self):
        return {"n_labels": self.n_labels,
                "output_activation": self.output_activation,
                "dropout": self.dropout_val}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


"""
####################################################################################################
# Training
####################################################################################################

if train_clas:
    # load dataset, apply GCN filter
    # apply train-test-split, generate  Disjoint loader for training and validation set
    data = dataset.GraphCluster(
        name=DATASET_CONT,
        edge_atr=True,
        adj_arg="binary")
    data.apply(GCNFilter())
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

    # create class weight dictionary for re-balancing class distributions
    class_weights = data.get_classweight_dict()
    print("# Class weights: ")
    print(class_weights)

    model_clas = model.GCNmodel(n_labels=1,
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

    # change to final result directory
    # store history, metrics, model, norm
    os.chdir(dir_results + RUN_NAME + "/")
    data.save_norm(RUN_NAME + "_classifier")
    plt_models.plot_history_classifier(history.history,
                                       RUN_NAME + "_history_classifier")
    model_clas.save(RUN_NAME + "_classifier")

if eval_clas:
    # change directory to saved model
    # load classifier model, load dataset normalization
    os.chdir(dir_results + RUN_NAME + "/")
    norm_x = np.load(RUN_NAME + "_classifier_norm_x.npy")
    norm_e = np.load(RUN_NAME + "_classifier_norm_e.npy")
    model_clas = tf.keras.models.load_model(RUN_NAME + "_classifier")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = "binary_crossentropy"
    list_metrics = ["Precision", "Recall"]
    model_clas.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=list_metrics)

    for file in [DATASET_CONT, DATASET_0MM, DATASET_5MM]:
        data = dataset.GraphCluster(name=file,
                                    edge_atr=True,
                                    adj_arg="binary",
                                    norm_x=norm_x,
                                    norm_e=norm_e)
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

####################################################################################################
# Training Regression Energy
####################################################################################################
