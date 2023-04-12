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

from src import NPZParser
from src import NeuralNetwork
from src import NNTraining
from src import NNEvaluation

# ----------------------------------------------------------------------------------------------------------------------
# Global Settings

# Root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons.root"
# Training file used for classification and regression training
# Generated via an input generator
NPZ_FILE_TRAIN = "OptimisedGeometry_Continuous_2e10protons_RNN_Base.npz"
# Evaluation files
# Generated via an input generator, contain one Bragg-peak position
NPZ_FILE_EVAL_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_RNN_Base.npz"
NPZ_FILE_EVAL_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_RNN_Base.npz"
EVALUATION_FILES = [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]

# Lookup files
# One for each evaluated file must ge given
# Containing Monte-Carlo truth and Cut-Based reconstruction information
NPZ_LOOKUP_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_Base_lookup.npz"
NPZ_LOOKUP_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_Base_lookup.npz"
LOOK_UP_FILES = [NPZ_LOOKUP_0MM, NPZ_LOOKUP_5MM]

# GLOBAL SETTINGS
RUN_NAME = "RNN_Base"

# Neural Network settings
epochs_clas = 10
epochs_regE = 100
epochs_regP = 100
batchsize_clas = 64
batchsize_regE = 64
batchsize_regP = 64
theta = 0.5

# Global switches to turn on/off training or analysis steps
train_clas = False
train_regE = False
train_regP = False
eval_clas = True
eval_regE = False
eval_regP = False
eval_full = False

# MLEM export setting: None (to disable export), "Reco" (for classical), "Pred" (For Neural Network predictions)
mlemexport = "PRED"

# ----------------------------------------------------------------------------------------------------------------------
# define directory paths

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"
# generate needed directories in the "results" subdirectory
# create subdirectory for run output
if not os.path.isdir(dir_results + RUN_NAME + "/"):
    os.mkdir(dir_results + RUN_NAME + "/")
# Evaluation directories
for file in [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]:
    if not os.path.isdir(dir_results + RUN_NAME + "/" + file[:-4] + "/"):
        os.mkdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")
# Training evaluation directory
if not os.path.isdir(dir_results + RUN_NAME + "/" + NPZ_FILE_TRAIN[:-4] + "/"):
    os.mkdir(dir_results + RUN_NAME + "/" + NPZ_FILE_TRAIN[:-4] + "/")

# ----------------------------------------------------------------------------------------------------------------------
# Training schedule

# load up the Tensorflow model
from models import RNN_classifier

tfmodel_clas = RNN_classifier.return_model(10, 10)
neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                 model_name=RUN_NAME,
                                                 model_tag="clas")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "/")

# generate DataCluster object from npz file
data_cluster = NPZParser.parse(dir_npz + NPZ_FILE_TRAIN,
                               frac=0.1,
                               set_classweights=True)

if train_clas:
    NNTraining.train_clas(neuralnetwork_clas,
                          data_cluster,
                          verbose=1,
                          epochs=epochs_clas,
                          batch_size=batchsize_clas)
if eval_clas:
    neuralnetwork_clas.load()

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation schedule

# evaluation of training data
os.chdir(dir_results + RUN_NAME + "/" + NPZ_FILE_TRAIN[:-4] + "/")
if eval_clas:
    data_cluster = NPZParser.parse(dir_npz + NPZ_FILE_TRAIN, set_testall=False)
    NNEvaluation.training_clas(neuralnetwork_clas, data_cluster, theta)

# Evaluation of test dataset
for i, file in enumerate([NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]):
    os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")
    # npz wrapper

    if train_clas or eval_clas:
        data_cluster = NPZParser.parse(dir_npz + file, set_testall=True)
        NNEvaluation.evaluate_classifier(neuralnetwork_clas, DataCluster=data_cluster)
