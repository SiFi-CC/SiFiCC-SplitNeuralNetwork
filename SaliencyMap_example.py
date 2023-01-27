import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from src import NPZParser
from src import NeuralNetwork
from src import TrainingHandler
from src import EvaluationHandler
from src import SaliencyMap

########################################################################################################################
# Global Settings
########################################################################################################################

# define file settings
# Training file
NPZ_FILE_TRAIN = "OptimizedGeometry_BP05_Base.npz"
# Evaluation file (can be list)
NPZ_FILE_EVAL = ["OptimisedGeometry_BP0mm_2e10protons_DNN_Base.npz",
                 "OptimisedGeometry_BP5mm_4e9protons_DNN_Base.npz"]

# GLOBAL SETTINGS
RUN_NAME = "DNN_Base"
RUN_TAG = "baseline"

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"

########################################################################################################################
# Training schedule
########################################################################################################################

# load up the Tensorflow model
from models import DNN_base_classifier

tfmodel_clas = DNN_base_classifier.return_model(72)
neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                 model_name=RUN_NAME,
                                                 model_tag=RUN_TAG + "_clas")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

# generate DataCluster object from npz file
data_cluster = NPZParser.wrapper(dir_npz + NPZ_FILE_TRAIN,
                                 standardize=True,
                                 set_classweights=True,
                                 set_peakweights=False)

neuralnetwork_clas.load()

########################################################################################################################
# Saliency Maps
########################################################################################################################

for i in range(10):
    x_feat = np.array([data_cluster.features[i], ])
    score_true = data_cluster.targets_clas[i]
    score_pred = float(neuralnetwork_clas.predict(x_feat))
    print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))

    smap = SaliencyMap.get_smap(neuralnetwork_clas.model, x_feat)
    smap = np.reshape(smap, (8, 9))
    str_title = "Event ID: {}\nTrue class: {:.1f}\nPred class: {:.2f}".format(data_cluster.meta[i, 0], score_true,
                                                                              score_pred)
    SaliencyMap.smap_plot(smap, x_feat, str_title, "sample_" + str(i))

"""
x_feat = np.array([data_cluster.features[8], ])
score_true = data_cluster.targets_clas[8]
score_pred = float(neuralnetwork_clas.predict(x_feat))
print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))

x_feat = np.array([data_cluster.features[8], ])
x_feat[:, 22] = 0.0
score_true = data_cluster.targets_clas[8]
score_pred = float(neuralnetwork_clas.predict(x_feat))
print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))

x_feat = np.array([data_cluster.features[8], ])
x_feat[:, 31] = 0.0
score_true = data_cluster.targets_clas[8]
score_pred = float(neuralnetwork_clas.predict(x_feat))
print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))
"""
