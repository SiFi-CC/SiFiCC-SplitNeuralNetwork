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

from src.SiFiCCNN.DataFrame import DFParser

from src.SiFiCCNN.NeuralNetwork import NeuralNetwork
from src.SiFiCCNN.NeuralNetwork import NNTraining
from src.SiFiCCNN.NeuralNetwork import NNEvaluation

from src.SiFiCCNN.Model import DNN_SXAX_classifier
from src.SiFiCCNN.Model import DNN_SXAX_regression_energy
from src.SiFiCCNN.Model import DNN_SXAX_regression_position
from src.SiFiCCNN.Model import DNN_SXAX_regression_theta

# ----------------------------------------------------------------------------------------------------------------------
# Global Settings

# Root files are purely optimal and are left as legacy settings
ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons.root"
ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons.root"
# Training file used for classification and regression training
# Generated via an input generator
NPZ_FILE_TRAIN = "OptimisedGeometry_Continuous_2e10protons_RNN_S4A6.npz"
# Evaluation files
# Generated via an input generator, contain one Bragg-peak position
NPZ_FILE_EVAL_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_RNN_S4A6.npz"
NPZ_FILE_EVAL_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_RNN_S4A6.npz"
EVALUATION_FILES = [NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM]

# Lookup files
# One for each evaluated file must ge given
# Containing Monte-Carlo truth and Cut-Based reconstruction information
NPZ_LOOKUP_0MM = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_CBRECO.npz"
NPZ_LOOKUP_5MM = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_CBRECO.npz"
NPZ_LOOKUP_TRAIN = "OptimisedGeometry_Continuous_2e10protons_CBRECO.npz"
LOOK_UP_FILES = [NPZ_LOOKUP_0MM, NPZ_LOOKUP_5MM, NPZ_LOOKUP_TRAIN]

# GLOBAL SETTINGS
RUN_NAME = "DNN_S4A6_master"

# Neural Network settings
epochs_clas = 50
epochs_regE = 100
epochs_regP = 100
epochs_regT = 100
batchsize_clas = 64
batchsize_regE = 64
batchsize_regP = 64
batchsize_regT = 64
theta = 0.5
n_frac = 1.0

# Global switches to turn on/off training or analysis steps
train_clas = False
train_regE = False
train_regP = False
train_regT = False
eval_clas = False
eval_regE = False
eval_regP = False
eval_regT = False
eval_full = True

# MLEM export setting: None (to disable export), "Reco" (for classical), "Pred" (For Neural Network predictions)
export_npz = False
export_cc6 = False

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
# TODO: Change input dimension dynamic

tfmodel_clas = DNN_SXAX_classifier.return_model(10, 10)
tfmodel_regE = DNN_SXAX_regression_energy.return_model(10, 10)
tfmodel_regP = DNN_SXAX_regression_position.return_model(10, 10)
tfmodel_regT = DNN_SXAX_regression_theta.return_model(10, 10)

neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                 model_name=RUN_NAME,
                                                 model_tag="clas")
neuralnetwork_regE = NeuralNetwork.NeuralNetwork(model=tfmodel_regE,
                                                 model_name=RUN_NAME,
                                                 model_tag="regE")
neuralnetwork_regP = NeuralNetwork.NeuralNetwork(model=tfmodel_regP,
                                                 model_name=RUN_NAME,
                                                 model_tag="regP")
neuralnetwork_regT = NeuralNetwork.NeuralNetwork(model=tfmodel_regT,
                                                 model_name=RUN_NAME,
                                                 model_tag="regT")

# CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
# TODO: fix this pls
os.chdir(dir_results + RUN_NAME + "/")

if train_clas:
    # generate DataCluster object from npz file
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN,
                                          n_frac=n_frac)
    NNTraining.train_classifier(neuralnetwork_clas,
                                data_cluster,
                                verbose=1,
                                epochs=epochs_clas,
                                batch_size=batchsize_clas)
if eval_clas:
    neuralnetwork_clas.load()

if train_regE:
    # generate DataCluster object from npz file
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN,
                                          n_frac=n_frac)

    NNTraining.train_regression_energy(neuralnetwork_regE,
                                       data_cluster,
                                       verbose=1,
                                       epochs=epochs_regE,
                                       batch_size=batchsize_regE)
if eval_regE:
    neuralnetwork_regE.load()

if train_regP:
    # generate DataCluster object from npz file
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN,
                                          n_frac=n_frac)

    NNTraining.train_regression_position(neuralnetwork_regP,
                                         data_cluster,
                                         verbose=1,
                                         epochs=epochs_regP,
                                         batch_size=batchsize_regP)
if eval_regP:
    neuralnetwork_regP.load()

if train_regT:
    # generate DataCluster object from npz file
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN,
                                          n_frac=n_frac)

    NNTraining.train_regression_theta(neuralnetwork_regT,
                                      data_cluster,
                                      verbose=1,
                                      epochs=epochs_regT,
                                      batch_size=batchsize_regT)
if eval_regT:
    neuralnetwork_regT.load()

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation schedule

# Evaluation of test dataset
for i, file in enumerate([NPZ_FILE_EVAL_0MM, NPZ_FILE_EVAL_5MM, NPZ_FILE_TRAIN]):
    os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

    if train_clas or eval_clas:
        data_cluster = DFParser.parse_cluster(dir_npz + file,
                                              n_frac=n_frac)
        if file in EVALUATION_FILES:
            data_cluster.p_test = 1.0
            data_cluster.p_train = 0.0
            data_cluster.p_valid = 0.0

        NNEvaluation.evaluate_classifier(neuralnetwork_clas,
                                         DataCluster=data_cluster,
                                         theta=theta)

    if train_regE or eval_regE:
        data_cluster = DFParser.parse_cluster(dir_npz + file,
                                              n_frac=n_frac)
        if file in EVALUATION_FILES:
            data_cluster.p_test = 1.0
            data_cluster.p_train = 0.0
            data_cluster.p_valid = 0.0

        NNEvaluation.evaluate_regression_energy(neuralnetwork_regE,
                                                DataCluster=data_cluster)

    if train_regP or eval_regP:
        data_cluster = DFParser.parse_cluster(dir_npz + file,
                                              n_frac=n_frac)
        if file in EVALUATION_FILES:
            data_cluster.p_test = 1.0
            data_cluster.p_train = 0.0
            data_cluster.p_valid = 0.0

        NNEvaluation.evaluate_regression_position(neuralnetwork_regP,
                                                  DataCluster=data_cluster)

    if train_regT or eval_regT:
        data_cluster = DFParser.parse_cluster(dir_npz + file,
                                              n_frac=n_frac)
        if file in EVALUATION_FILES:
            data_cluster.p_test = 1.0
            data_cluster.p_train = 0.0
            data_cluster.p_valid = 0.0

        NNEvaluation.evaluate_regression_theta(neuralnetwork_regT,
                                               DataCluster=data_cluster)

    if eval_full:
        os.chdir(dir_results + RUN_NAME + "/")
        neuralnetwork_clas.load()
        neuralnetwork_regE.load()
        neuralnetwork_regP.load()
        neuralnetwork_regT.load()
        os.chdir(dir_results + RUN_NAME + "/" + file[:-4] + "/")

        data_cluster = DFParser.parse_cluster(dir_npz + file,
                                              n_frac=n_frac)
        """        
        NNEvaluation.eval_complete(neuralnetwork_clas,
                                   neuralnetwork_regE,
                                   neuralnetwork_regP,
                                   neuralnetwork_regT,
                                   DataCluster=data_cluster,
                                   theta=theta,
                                   file_name=file[:-4],
                                   export_npz=export_npz,
                                   export_CC6=export_cc6)
        """
        NNEvaluation.eval_reco_compare(neuralnetwork_regE,
                                       neuralnetwork_regP,
                                       neuralnetwork_regT,
                                       DataCluster=data_cluster,
                                       reco_file=dir_npz + LOOK_UP_FILES[i])
