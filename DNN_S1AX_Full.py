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
RUN_NAME = "DNN_S1AX_Full"
RUN_TAG = "filter"

b_mlemexport = True
b_eval = True
b_loadclas = True
b_loadregE = True
b_loadregP = True

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

from models import DNN_base_classifier
from models import DNN_base_regression_energy
from models import DNN_base_regression_position

tf_model_clas = DNN_base_classifier.return_model(54)
tf_model_regE = DNN_base_regression_energy.return_model(54)
tf_model_regP = DNN_base_regression_position.return_model(54)

nn_clas = NeuralNetwork.NeuralNetwork(model=tf_model_clas,
                                      model_name=RUN_NAME,
                                      model_tag=RUN_TAG + "_clas")
nn_regE = NeuralNetwork.NeuralNetwork(model=tf_model_regE,
                                      model_name=RUN_NAME,
                                      model_tag=RUN_TAG + "_regE")
nn_regP = NeuralNetwork.NeuralNetwork(model=tf_model_regP,
                                      model_name=RUN_NAME,
                                      model_tag=RUN_TAG + "_regP")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

if not b_loadclas:
    TrainingHandler.train_clas(nn_clas, dir_npz + NPZ_FILE_TRAIN, verbose=1)
else:
    nn_clas.load()
if not b_loadregE:
    TrainingHandler.train_regEnergy(nn_regE, dir_npz + NPZ_FILE_TRAIN, verbose=1)
else:
    nn_regE.load()
if not b_loadregP:
    TrainingHandler.train_regPosition(nn_regP, dir_npz + NPZ_FILE_TRAIN, verbose=1)
else:
    nn_regP.load()
########################################################################################################################
# Evaluation schedule
########################################################################################################################

for i in range(len(NPZ_FILE_EVAL)):
    os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + NPZ_FILE_EVAL[i][:-4] + "/")
    #EvaluationHandler.eval_classifier(nn_clas, dir_npz + NPZ_FILE_EVAL[i], predict_full=False)
    #EvaluationHandler.eval_regression_energy(nn_regE, dir_npz + NPZ_FILE_EVAL[i], predict_full=False)
    #EvaluationHandler.eval_regression_position(nn_regP, dir_npz + NPZ_FILE_EVAL[i], predict_full=False)

    EvaluationHandler.eval_full(nn_clas,
                                nn_regE,
                                nn_regP,
                                dir_npz + NPZ_FILE_EVAL[i],
                                file_name=NPZ_FILE_EVAL[i][:-4] + "_" + RUN_NAME + "_" + RUN_TAG)
