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
NPZ_FILE_TRAIN = "OptimizedGeometry_BP05_S1AX.npz"
# NPZ_FILE_TRAIN = "OptimisedGeometry_BP0mm_2e10protons_DNN_S1AX.npz"
# Evaluation file (can be list)
NPZ_FILE_EVAL = ["OptimisedGeometry_BP0mm_2e10protons_DNN_S1AX.npz",
                 "OptimisedGeometry_BP5mm_4e9protons_DNN_S1AX.npz"]

# GLOBAL SETTINGS
RUN_NAME = "DNN_S1AX"
RUN_TAG = "mixed"

b_training = True
b_mlemexport = True

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
from models import DNN_base_regression_energy

tf_model = DNN_base_regression_energy.return_model(54)
neuralnetwork_regression = NeuralNetwork.NeuralNetwork(model=tf_model,
                                                       model_name=RUN_NAME,
                                                       model_tag=RUN_TAG + "_regEnergy")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

if b_training:
    TrainingHandler.train_regEnergy(neuralnetwork_regression, dir_npz + NPZ_FILE_TRAIN, verbose=1)
else:
    neuralnetwork_regression.load()


for i in range(len(NPZ_FILE_EVAL)):
    os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + NPZ_FILE_EVAL[i][:-4] + "/")
    EvaluationHandler.eval_regression_energy(neuralnetwork_regression, dir_npz + NPZ_FILE_EVAL[i])
