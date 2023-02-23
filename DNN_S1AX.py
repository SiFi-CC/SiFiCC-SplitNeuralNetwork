"""
Deep Neural Network Model for event classification and event topology regression for the SiFiCC-project.
This Network operates on the Base setting: X scatterer cluster and X absorber cluster (X > 0) per event.

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
NPZ_FILE_TRAIN = "OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz"
# Evaluation file (can be list)
NPZ_FILE_EVAL_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz"
NPZ_FILE_EVAL_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz"

NPZ_LOOKUP_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz"
NPZ_LOOKUP_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz"
LOOK_UP_FILES = [NPZ_LOOKUP_0MM, NPZ_LOOKUP_5MM]

# GLOBAL SETTINGS
RUN_NAME = "DNN_S1AX"
RUN_TAG = "continuous"

epochs = 10

train_clas = False
train_regE = False
train_regP = False
eval_clas = True
eval_regE = True
eval_regP = False

mlemexport = False

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"
# generate needed directories in the "results" subdirectory
# create subdirectory for run output
if not os.path.isdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/"):
    os.mkdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

for file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
    if not os.path.isdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + file[:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + file[:-4] + "/")

########################################################################################################################
# Training schedule
########################################################################################################################

# load up the Tensorflow model
from models import DNN_base_classifier
from models import DNN_base_regression_energy
from models import DNN_base_regression_position

tfmodel_clas = DNN_base_classifier.return_model(80)
tfmodel_regE = DNN_base_regression_energy.return_model(80)
tfmodel_regP = DNN_base_regression_position.return_model(80)

neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                 model_name=RUN_NAME,
                                                 model_tag=RUN_TAG + "_clas")

neuralnetwork_regE = NeuralNetwork.NeuralNetwork(model=tfmodel_regE,
                                                 model_name=RUN_NAME,
                                                 model_tag=RUN_TAG + "_regE")

neuralnetwork_regP = NeuralNetwork.NeuralNetwork(model=tfmodel_regP,
                                                 model_name=RUN_NAME,
                                                 model_tag=RUN_TAG + "_regP")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

# generate DataCluster object from npz file
data_cluster = NPZParser.wrapper(dir_npz + NPZ_FILE_TRAIN,
                                 standardize=True,
                                 set_classweights=True,
                                 set_peakweights=False)

if train_clas:
    TrainingHandler.train_clas(neuralnetwork_clas,
                               data_cluster,
                               verbose=1,
                               epochs=epochs)
if eval_clas:
    neuralnetwork_clas.load()

# generate DataCluster object from npz file
data_cluster = NPZParser.wrapper(dir_npz + NPZ_FILE_TRAIN,
                                 standardize=True,
                                 set_classweights=False,
                                 set_peakweights=False)

if train_regE:
    TrainingHandler.train_regE(neuralnetwork_regE,
                               data_cluster,
                               verbose=1,
                               epochs=epochs)
if eval_regE:
    neuralnetwork_regE.load()

if train_regP:
    TrainingHandler.train_regP(neuralnetwork_regP,
                               data_cluster,
                               verbose=1,
                               epochs=epochs)
if eval_regP:
    neuralnetwork_regP.load()

########################################################################################################################
# Evaluation schedule
########################################################################################################################

for i, file in enumerate([NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]):
    os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/" + file[:-4] + "/")
    # npz wrapper

    if train_clas or eval_clas:
        data_cluster = NPZParser.wrapper(dir_npz + file,
                                         set_testall=True,
                                         standardize=True)
        EvaluationHandler.evaluate_classifier(neuralnetwork_clas,
                                              data_cluster=data_cluster)
    if train_regE or eval_regE:
        data_cluster = NPZParser.wrapper(dir_npz + file,
                                         set_testall=True,
                                         standardize=True)
        EvaluationHandler.eval_regression_energy(neuralnetwork_regE, DataCluster=data_cluster)

    if train_regP or eval_regP:
        data_cluster = NPZParser.wrapper(dir_npz + file,
                                         set_testall=True,
                                         standardize=True)
        EvaluationHandler.eval_regression_position(neuralnetwork_regP, DataCluster=data_cluster)

    data_cluster = NPZParser.wrapper(dir_npz + file,
                                     set_testall=True,
                                     standardize=True)

    EvaluationHandler.eval_full(neuralnetwork_clas,
                                neuralnetwork_regE,
                                neuralnetwork_regP,
                                DataCluster=data_cluster,
                                lookup_file=dir_npz + LOOK_UP_FILES[i],
                                theta=0.3,
                                file_name=file[:-4])
