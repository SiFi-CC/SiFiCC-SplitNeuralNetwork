import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from src import NPZParser
from src import NeuralNetwork
from src import TrainingHandler
from src import EvaluationHandler

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

def get_saliency_map(model, features):
    # from : https://stackoverflow.com/questions/63107141/how-to-compute-saliency-map-using-keras-backend

    # conversion from numpy array to tensorflow tensor
    image = tf.convert_to_tensor(features)
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(prediction, image)

    # convert to numpy
    gradient = gradient.numpy()

    # normalize between 0 and 1
    min_val, max_val = np.min(gradient), np.max(gradient)
    smap = (gradient - min_val) / (max_val - min_val + keras.backend.epsilon())

    return smap


def smap_plot(smap, title, file_name):
    plt.figure()
    plt.title(title)
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
               labels=["No. fibers", "Energy", "Pos X", "Pos Y", "Pos Z", "Unc. Energy", "Unc. Pos X", "Unc. Pos Y",
                       "Unc. Pos Z"],
               rotation=90)
    plt.yticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7],
               labels=["S1", "S2", "A1", "A2", "A2", "A4", "A5", "A6"])
    plt.imshow(smap, vmin=0.0, vmax=1.0, cmap="Reds")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(file_name + ".png")

tfmodel_clas_untrained = DNN_base_classifier.return_model(72)
neuralnetwork_clas_untrained = NeuralNetwork.NeuralNetwork(model=tfmodel_clas_untrained,
                                                           model_name=RUN_NAME,
                                                           model_tag=RUN_TAG + "_clas_untrained")

for i in range(10):
    x_feat = np.array([data_cluster.features[i], ])
    score_true = data_cluster.targets_clas[i]
    score_pred = float(neuralnetwork_clas_untrained.predict(x_feat))
    print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))

    smap = get_saliency_map(neuralnetwork_clas_untrained.model, x_feat)
    smap = np.reshape(smap, (8, 9))
    str_title = "Event ID: {}\nTrue class: {:.1f}\nPred class: {:.2f}".format(data_cluster.meta[i, 0], score_true,
                                                                              score_pred)
    smap_plot(smap, str_title, "sample_" + str(i))

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
