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

from keras.models import Model
from keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool

# ------------------------------------------------------------------------------
# Global settings

RUN_NAME = "GCN_SXAX"

DATASET = "SiFiCCCluster"
DATASET_TRAIN = "OptimisedGeometry_Continuous_2e10protons_SiFiCCCluster"

# Neural Network settings
epochs_clas = 20
batchsize_clas = 32

gen_datasets = False

theta = 0.5
n_frac = 1.0

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

# Train/test split
np.random.shuffle(dataset)
split = int(0.8 * len(dataset))
data_tr, data_te = dataset[:split], dataset[split:]

# Data loaders
loader_tr = DisjointLoader(data_tr,
                           node_level=False,
                           batch_size=batchsize_clas)
loader_te = DisjointLoader(data_te,
                           node_level=False,
                           batch_size=batchsize_clas)

# create class weight dictionary
class_weights = dataset.get_classweight_dict()


# ------------------------------------------------------------------------------
# Build Model

class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(32, activation="relu")
        self.conv2 = GCNConv(32, activation="relu")
        self.conv3 = GCNConv(32, activation="relu")
        self.Dropout = Dropout(0.3)
        self.global_pool = GlobalSumPool()
        self.dense1 = Dense(26, activation="relu")
        self.dense2 = Dense(1, activation="sigmoid")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        # x = self.Dropout(x)
        output = self.global_pool([x, i])
        output = self.dense1(output)
        output = self.dense2(output)

        return output


# ------------------------------------------------------------------------------
# generate tensorflow model

# model_clas = GeneralGNN(dataset.n_labels, activation="sigmoid")
model_clas = Net()
neuralnetwork = Spektral_NeuralNetwork.NeuralNetwork(model=model_clas,
                                                     model_name=RUN_NAME + "_clas",
                                                     epochs=epochs_clas,
                                                     class_weights=class_weights,
                                                     batch_size=batchsize_clas)

# ------------------------------------------------------------------------------
# Training

neuralnetwork.train(loader_tr)

# ------------------------------------------------------------------------------
# Evaluate Network

from src.SiFiCCNN.NeuralNetwork import NNAnalysis
from src.SiFiCCNN.NeuralNetwork import FastROCAUC

from src.SiFiCCNN.Plotter import PTClassifier

for file in [DATASET_TRAIN]:
    os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

    y_scores = neuralnetwork.predict(loader_te)
    y_true = neuralnetwork.y(loader_te)

    # ROC-AUC Analysis
    FastROCAUC.fastROCAUC(y_scores, y_true, save_fig="ROCAUC")
    _, theta_opt = FastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)

    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    PTClassifier.plot_score_distribution(y_scores, y_true, "score_dist")

    # write general binary classifier metrics into console and .txt file
    NNAnalysis.write_metrics_classifier(y_scores, y_true)
