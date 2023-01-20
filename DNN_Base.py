"""
Deep Neural Network Model for event classification and event topology regression for the SiFiCC-project.
This Network operates on the S1AX setting: 1 scatterer cluster and X absorber cluster (X > 0) per event.

Currently implemented:

- Classification via Deep Neural Network
- Training on Mixed dataset
- Evaluation on dataset with a single bragg peak position ( = one beam spot)
- Export to MLEM Image reconstruction
    - via classic event topology regression
    - with and without applied filters

"""

import os
import numpy as np

from src import NPZParser
from src import NeuralNetwork
from src import TrainingHandler
from src import EvaluationHandler

########################################################################################################################
# Global Settings
########################################################################################################################

# define file settings
# Root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons.root"
# Training file
NPZ_FILE_TRAIN = "OptimizedGeometry_BP05_Base.npz"
# NPZ_FILE_TRAIN = "OptimisedGeometry_BP0mm_2e10protons_DNN_S1AX.npz"
# Evaluation file (can be list)
NPZ_FILE_EVAL = ["OptimisedGeometry_BP0mm_2e10protons_DNN_Base.npz",
                 "OptimisedGeometry_BP5mm_4e9protons_DNN_Base.npz"]

# GLOBAL SETTINGS
RUN_NAME = "DNN_Base"
RUN_TAG = "eweights"

b_training = True
b_mlemexport = False

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"
# generate needed directories in the "results" subdirectory
# create subdirectory for run output
if not os.path.isdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/"):
    os.mkdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

for i in range(len(NPZ_FILE_EVAL)):
    if not os.path.isdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + NPZ_FILE_EVAL[i][:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + NPZ_FILE_EVAL[i][:-4] + "/")

########################################################################################################################
# Training schedule
########################################################################################################################

# load up the Tensorflow model
from models import DNN_base_classifier

tf_model = DNN_base_classifier.return_model(72)
neuralnetwork_classifier = NeuralNetwork.NeuralNetwork(model=tf_model,
                                                       model_name=RUN_NAME,
                                                       model_tag=RUN_TAG)

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

if b_training:
    TrainingHandler.train_clas(neuralnetwork_classifier, dir_npz + NPZ_FILE_TRAIN, verbose=1)
else:
    neuralnetwork_classifier.load()

########################################################################################################################
# Evaluation schedule
########################################################################################################################

for i in range(len(NPZ_FILE_EVAL)):
    os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + NPZ_FILE_EVAL[i][:-4] + "/")
    # npz wrapper
    data_cluster = EvaluationHandler.npz_wrapper(dir_npz + NPZ_FILE_EVAL[i],
                                                 predict_all=True,
                                                 standardize=True)

    EvaluationHandler.eval_classifier(neuralnetwork_classifier,
                                      data_cluster=data_cluster)

    if b_mlemexport:
        EvaluationHandler.export_mlem_simpleregression(neuralnetwork_classifier,
                                                       dir_npz + NPZ_FILE_EVAL[i],
                                                       NPZ_FILE_EVAL[i])
