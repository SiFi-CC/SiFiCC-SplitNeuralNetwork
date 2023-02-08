import numpy as np
import math
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


def generate_npz_file():
    # --------------------------------------------
    # General input for data generation:
    # - root file
    # - npz file for Neural network features
    # - Neural Network Classification, Regression models
    from src import RootParser
    from src import root_files
    from src import NPZParser

    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"
    dir_results = dir_main + "/results/"

    root_parser = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
    DataCluster = NPZParser.wrapper(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_BaseTime.npz",
                                    standardize=True,
                                    set_classweights=False,
                                    set_peakweights=False)

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
    os.chdir(dir_main)

    # prepare final arrays
    n_entries = root_parser.events_entries
    ary_cb_pred = np.zeros(shape=(n_entries, 9))
    ary_nn_pred = np.zeros(shape=(n_entries, 9))
    ary_mc_true = np.erzos(shape=(n_entries, 9))

    # grab identified tag from  root file
    ary_root_eventnumber = root_parser.events["EventNumber"].array()
    ary_root_identified = root_parser.events["Identified"].array()
    ary_root_source_position = root_parser.events["MCPosition_source"].array().z

    for i, event in enumerate(root_parser.iterate_events(n=None)):
        if event.Identified == 0:
            continue

        e_e, _ = event.get_electron_energy()
        e_p, _ = event.get_electron_position()
        p_e, _ = event.get_photon_energy()
        p_p, _ = event.get_photon_position()

        ary_cb_pred[i, :] = [ary_root_identified[i], e_e, p_e, e_p.x, e_p.y, e_p.z, p_p.x, p_p.y, p_p.z]

    # load npz file
    npz_features = DataCluster.features

    ary_nn_pred[:, 0] = neuralnetwork_clas.predict(npz_features)
    ary_nn_pred[:, 1:3] = neuralnetwork_regE.predict(npz_features)
    ary_nn_pred[:, 3:9] = neuralnetwork_regP.predict(npz_features)

    ary_mc_true[:, 0] = DataCluster.targets_clas
    ary_mc_true[:, 1:3] = DataCluster.targets_reg1
    ary_mc_true[:, 3:9] = DataCluster.targets_reg2

    str_savefile = "OptimisedGeometry_BP0mm_statistics.npz"
    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            identified=ary_root_identified,
                            nn_pred=ary_nn_pred,
                            source_position=ary_root_source_position)

generate_npz_file()
"""
# ----------------------------------------------------------------------------------------------------------------------
# Analysis script

# Grab all information from the target file
npz_data = np.load("OptimisedGeometry_BP0mm_statistics.npz")
ary_identified = npz_data["identified"]
ary_pred_score = npz_data["pred_score"]
ary_true_score = npz_data["true_score"]
ary_pred_energy = npz_data["pred_energy"]
ary_true_energy = npz_data["true_energy"]
ary_pred_position = npz_data["pred_position"]
ary_true_position = npz_data["true_position"]
ary_source_position = npz_data["source_position"]

# grab all indices of all ideal compton events
idx_ic = [float(ary_true_score[i]) > 0.5 for i in range(len(ary_true_score))]

# ---------------------------------------------------------------------------
# angle error
from src import MLEMBackprojection

counter = 0
for k in range(len(ary_true_score)):
    if ary_true_score[k] == 0:
        continue

    angle_pred = MLEMBackprojection.calculate_theta(ary_pred_energy[k, 0], ary_pred_energy[k, 1])
    if ary_pred_energy[k, 0] == 0.0 or ary_pred_energy[k, 1] == 0.0:
        print("Invalid energy: {:.2f} {:.2f} | {:.2f} {:.2f}".format(ary_pred_energy[k, 0],
                                                                     ary_pred_energy[k, 1],
                                                                     ary_true_energy[k, 0],
                                                                     ary_true_energy[k, 1]))

    if math.isnan(angle_pred):
        print("Invalid arccos:  {} | {:.2f} {:.2f} | {:.2f} {:.2f}".format(ary_identified[k],
                                                                           ary_pred_energy[k, 0],
                                                                           ary_pred_energy[k, 1],
                                                                           ary_true_energy[k, 0],
                                                                           ary_true_energy[k, 1]))

bins_sp = np.arange(-40.0, 10.0, 1.0)
list_angle_mean = []
list_angle_std = []

for i in range(len(bins_sp[:-1])):
    list_angle_err_temp = []
    print(bins_sp[i], bins_sp[i + 1])

    for k in range(len(ary_true_score)):
        if ary_true_score[k] == 0:
            continue
        if bins_sp[i] < ary_source_position[k] < bins_sp[i + 1]:
            if ary_pred_energy[k, 0] == 0.0 or ary_pred_energy[k, 1] == 0.0:
                list_angle_err_temp.append(-np.pi)
            else:
                angle_pred = MLEMBackprojection.calculate_theta(ary_pred_energy[k, 0], ary_pred_energy[k, 1])
                angle_true = MLEMBackprojection.calculate_theta(ary_true_energy[k, 0], ary_true_energy[k, 1])
                list_angle_err_temp.append(angle_pred - angle_true)
    print(np.mean(list_angle_err_temp), np.std(list_angle_err_temp))
    list_angle_mean.append(np.mean(list_angle_err_temp))
    list_angle_std.append(np.std(list_angle_err_temp))

plt.figure()
bins = np.arange(-np.pi, np.pi, 0.05)
plt.xlabel("MCSource_Position.z [mm]")
plt.ylabel(r"E[$\theta^{pred}-\theta^{true}$] [rad]")
plt.errorbar(bins_sp[:-1] + 0.5, list_angle_mean, list_angle_std, color="blue")
plt.tight_layout()
plt.show()
"""
