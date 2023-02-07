import numpy as np
import os
import matplotlib.pyplot as plt
from src import RootParser
from src import root_files
from src import NPZParser


def generate_npz_file(root_parser,
                      DataCluster,
                      NeuralNetwork_clas,
                      NeuralNetwork_regE,
                      NeuralNetwork_regP):
    # grab identified tag from  root file
    ary_root_identified = root_parser.events["Identified"].array()

    # load npz file
    npz_features = DataCluster.features
    npz_targetC = DataCluster.targets_clas
    npz_targetE = DataCluster.targets_reg1
    npz_targetP = DataCluster.targets_reg2

    y_scores = NeuralNetwork_clas.predict(npz_features)
    y_pred_energy = NeuralNetwork_regE.predict(npz_features)
    y_pred_position = NeuralNetwork_regP.predict(npz_features)

    str_savefile = "OptimisedGeometry_BP0mm_statistics.npz"
    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            identified=ary_root_identified,
                            pred_score=y_scores,
                            true_score=npz_targetC,
                            pred_energy=y_pred_energy,
                            true_energy=npz_targetE,
                            pred_position=y_pred_position,
                            true_position=npz_targetP)


#########################################################################################

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"
"""
# Reading root files and export it to npz files
root1 = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
data_cluster = NPZParser.wrapper(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_BaseTime.npz",
                                 standardize=True,
                                 set_classweights=False,
                                 set_peakweights=False)

# ----------------------------------------------------------------------------------------------------
# loading neural network models
RUN_NAME = "DNN_BaseTime"
RUN_TAG = "Baseline"
os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")

from src import NeuralNetwork
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
neuralnetwork_clas.load()
neuralnetwork_regE.load()
neuralnetwork_regP.load()

generate_npz_file(root1,
                  data_cluster,
                  neuralnetwork_clas,
                  neuralnetwork_regE,
                  neuralnetwork_regP)
"""

# ------------------------------------------------------------------------------------------------------
# Analysis script

npz_data = np.load("OptimisedGeometry_BP0mm_statistics.npz")
ary_identified = npz_data["identified"]
ary_pred_score = npz_data["pred_score"]
ary_true_score = npz_data["true_score"]
ary_pred_energy = npz_data["pred_energy"]
ary_true_energy = npz_data["true_energy"]
ary_pred_position = npz_data["pred_position"]
ary_true_position = npz_data["true_position"]

idx_pos = [float(ary_pred_score[i]) > 0.5 for i in range(len(ary_pred_score))]
print(ary_pred_score)

bins = np.arange(-20.0, 20.0, 0.05)
plt.figure()
plt.hist(ary_pred_position[idx_pos, 5] - ary_true_position[idx_pos, 5], bins=bins, histtype=u"step")
plt.show()
