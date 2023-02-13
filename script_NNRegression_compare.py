import numpy as np
import math
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


def generate_export():
    # --------------------------------------------
    # General input for data generation:
    # - root file
    # - npz file for Neural network features
    # - Neural Network Classification, Regression models
    from src import RootParser
    from src import root_files
    from src import NPZParser

    from src import NeuralNetwork
    from models import DNN_base_classifier
    from models import DNN_base_regression_energy
    from models import DNN_base_regression_position

    # ---------------------------------------------
    # definition of directory paths
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"
    dir_results = dir_main + "/results/"

    # ---------------------------------------------
    # loading neural network models
    RUN_NAME = "DNN_BaseTime"
    RUN_TAG = "emod"
    LOOKUP_NAME_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_lookup.npz"
    LOOKUP_NAME_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_lookup.npz"

    SAMPLE_NAME_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_BaseTime.npz"
    SAMPLE_NAME_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_BaseTime.npz"

    # ----------------------------------------------
    # load up all required datasets
    # - Lookup file for MC-Truth and CB-Reco
    # - NN predictions

    # Lookup datasets
    npz_lookup_bp0mm = np.load(dir_npz + LOOKUP_NAME_BP0mm)
    npz_lookup_bp5mm = np.load(dir_npz + LOOKUP_NAME_BP5mm)

    ary_mc_truth_bp0mm = npz_lookup_bp0mm["MC_TRUTH"]
    ary_mc_truth_bp5mm = npz_lookup_bp5mm["MC_TRUTH"]

    ary_cb_reco_bp0mm = npz_lookup_bp0mm["CB_RECO"]
    ary_cb_reco_bp5mm = npz_lookup_bp5mm["CB_RECO"]

    # get neural network predictions
    os.chdir(dir_results + RUN_NAME + "_" + RUN_TAG + "/")
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

    DataCluster_BP0mm = NPZParser.wrapper(dir_npz + SAMPLE_NAME_BP0mm,
                                          standardize=True,
                                          set_classweights=False,
                                          set_peakweights=False)
    DataCluster_BP5mm = NPZParser.wrapper(dir_npz + SAMPLE_NAME_BP5mm,
                                          standardize=True,
                                          set_classweights=False,
                                          set_peakweights=False)

    ary_nn_pred_bp0mm = np.zeros(shape=(DataCluster_BP0mm.entries, 9))
    ary_nn_pred_bp5mm = np.zeros(shape=(DataCluster_BP5mm.entries, 9))

    # get neural network predictions
    y_pred_bp0mm = neuralnetwork_clas.predict(DataCluster_BP0mm.features)
    ary_nn_pred_bp0mm[:, 0] = np.reshape(y_pred_bp0mm, newshape=(len(y_pred_bp0mm),))
    ary_nn_pred_bp0mm[:, 1:3] = neuralnetwork_regE.predict(DataCluster_BP0mm.features)
    ary_nn_pred_bp0mm[:, 3:9] = neuralnetwork_regP.predict(DataCluster_BP0mm.features)

    y_pred_bp5mm = neuralnetwork_clas.predict(DataCluster_BP5mm.features)
    ary_nn_pred_bp5mm[:, 0] = np.reshape(y_pred_bp5mm, newshape=(len(y_pred_bp5mm),))
    ary_nn_pred_bp5mm[:, 1:3] = neuralnetwork_regE.predict(DataCluster_BP5mm.features)
    ary_nn_pred_bp5mm[:, 3:9] = neuralnetwork_regP.predict(DataCluster_BP5mm.features)

    str_savefile = "BaseTime_emod.npz"
    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            mc_truth_0mm=ary_mc_truth_bp0mm,
                            mc_truth_5mm=ary_mc_truth_bp5mm,
                            cb_reco_0mm=ary_cb_reco_bp0mm,
                            cb_reco_5mm=ary_cb_reco_bp5mm,
                            nn_pred_0mm=ary_nn_pred_bp0mm,
                            nn_pred_5mm=ary_nn_pred_bp5mm)


# ----------------------------------------------------------------------------------------------------------------------
# Analysis script

npz_data = np.load("BaseTime_emod.npz")
ary_mc_truth_bp0mm = npz_data["mc_truth_0mm"]
ary_mc_truth_bp5mm = npz_data["mc_truth_5mm"]
ary_cb_reco_bp0mm = npz_data["cb_reco_0mm"]
ary_cb_reco_bp5mm = npz_data["cb_reco_5mm"]
ary_nn_pred_bp0mm = npz_data["nn_pred_0mm"]
ary_nn_pred_bp5mm = npz_data["nn_pred_5mm"]

npz_lookup_0mm = np.load(os.getcwd() + "/npz_files/" + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_lookup.npz")
npz_lookup_5mm = np.load(os.getcwd() + "/npz_files/" + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_lookup.npz")
ary_meta_0mm = npz_lookup_0mm["META"]
ary_meta_5mm = npz_lookup_5mm["META"]

# ----------------------------------------------------------------------------------------------------------------------
# Backprojection plots
from src import MLEMBackprojection

n = 50000

idx_pos_0mm = ary_nn_pred_bp0mm[:, 0] > 0.2
idx_pos_5mm = ary_nn_pred_bp5mm[:, 0] > 0.2

ary_nn_pred_bp0mm = ary_nn_pred_bp0mm[idx_pos_0mm, :]
ary_nn_pred_bp5mm = ary_nn_pred_bp5mm[idx_pos_5mm, :]

image_0mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp0mm[:n, 1],
                                                 ary_nn_pred_bp0mm[:n, 2],
                                                 ary_nn_pred_bp0mm[:n, 3],
                                                 ary_nn_pred_bp0mm[:n, 4],
                                                 ary_nn_pred_bp0mm[:n, 5],
                                                 ary_nn_pred_bp0mm[:n, 6],
                                                 ary_nn_pred_bp0mm[:n, 7],
                                                 ary_nn_pred_bp0mm[:n, 8])

image_5mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp5mm[:n, 1],
                                                 ary_nn_pred_bp5mm[:n, 2],
                                                 ary_nn_pred_bp5mm[:n, 3],
                                                 ary_nn_pred_bp5mm[:n, 4],
                                                 ary_nn_pred_bp5mm[:n, 5],
                                                 ary_nn_pred_bp5mm[:n, 6],
                                                 ary_nn_pred_bp5mm[:n, 7],
                                                 ary_nn_pred_bp5mm[:n, 8])
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm, "Backprojection NN prediction",
                                               "MLEM_backproj_NNPRED_emod_theta02")

"""
idx_id_0mm = ary_meta_0mm[:, 3] != 0
idx_id_5mm = ary_meta_5mm[:, 3] != 0

ary_cb_reco_bp0mm = ary_cb_reco_bp0mm[idx_id_0mm, :]
ary_cb_reco_bp5mm = ary_cb_reco_bp5mm[idx_id_5mm, :]
image_0mm = MLEMBackprojection.reconstruct_image(ary_cb_reco_bp0mm[:n, 0],
                                                 ary_cb_reco_bp0mm[:n, 1],
                                                 ary_cb_reco_bp0mm[:n, 2],
                                                 ary_cb_reco_bp0mm[:n, 3],
                                                 ary_cb_reco_bp0mm[:n, 4],
                                                 ary_cb_reco_bp0mm[:n, 5],
                                                 ary_cb_reco_bp0mm[:n, 6],
                                                 ary_cb_reco_bp0mm[:n, 7])

image_5mm = MLEMBackprojection.reconstruct_image(ary_cb_reco_bp5mm[:n, 0],
                                                 ary_cb_reco_bp5mm[:n, 1],
                                                 ary_cb_reco_bp5mm[:n, 2],
                                                 ary_cb_reco_bp5mm[:n, 3],
                                                 ary_cb_reco_bp5mm[:n, 4],
                                                 ary_cb_reco_bp5mm[:n, 5],
                                                 ary_cb_reco_bp5mm[:n, 6],
                                                 ary_cb_reco_bp5mm[:n, 7])
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection CB identified", "MLEM_backproj_CBreco")

"""

"""
idx_ic_0mm = ary_meta_0mm[:, 2] == 1
idx_ic_5mm = ary_meta_5mm[:, 2] == 1

ary_mc_truth_bp0mm = ary_mc_truth_bp0mm[idx_ic_0mm, :]
ary_mc_truth_bp5mm = ary_mc_truth_bp5mm[idx_ic_5mm, :]

image_0mm = MLEMBackprojection.reconstruct_image(ary_mc_truth_bp0mm[:n, 0],
                                                 ary_mc_truth_bp0mm[:n, 1],
                                                 ary_mc_truth_bp0mm[:n, 2],
                                                 ary_mc_truth_bp0mm[:n, 3],
                                                 ary_mc_truth_bp0mm[:n, 4],
                                                 ary_mc_truth_bp0mm[:n, 5],
                                                 ary_mc_truth_bp0mm[:n, 6],
                                                 ary_mc_truth_bp0mm[:n, 7])

image_5mm = MLEMBackprojection.reconstruct_image(ary_mc_truth_bp5mm[:n, 0],
                                                 ary_mc_truth_bp5mm[:n, 1],
                                                 ary_mc_truth_bp5mm[:n, 2],
                                                 ary_mc_truth_bp5mm[:n, 3],
                                                 ary_mc_truth_bp5mm[:n, 4],
                                                 ary_mc_truth_bp5mm[:n, 5],
                                                 ary_mc_truth_bp5mm[:n, 6],
                                                 ary_mc_truth_bp5mm[:n, 7])

MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection MC Truth (Ideal Compton)", "MLEM_backproj_MCTRUTH")
"""

"""
# ----------------------------------------------------------------------------------------------------------------------
# distribution of scattering angle
from src import MLEMBackprojection

list_theta_err_cb = []
list_theta_err_cb_peak = []
list_theta_err_nn = []
list_theta_err_nn_peak = []

for i in range(ary_mc_truth.shape[0]):
    if ary_mc_truth[i, 0] in [1.0]:
        theta_cb = MLEMBackprojection.calculate_theta(ary_cb_pred[i, 1], ary_cb_pred[i, 2])
        theta_nn = MLEMBackprojection.calculate_theta(ary_nn_pred[i, 1], ary_nn_pred[i, 2])
        theta_mc = MLEMBackprojection.calculate_theta(ary_mc_truth[i, 1], ary_mc_truth[i, 2])

        if math.isnan(theta_cb):
            list_theta_err_cb.append(-np.pi)
        else:
            list_theta_err_cb.append(theta_cb - theta_mc)

        if math.isnan(theta_nn):
            list_theta_err_nn.append(-np.pi)
        else:
            list_theta_err_nn.append(theta_nn - theta_mc)

        if -8.0 < ary_sp[i] < 2.0:
            if math.isnan(theta_cb):
                list_theta_err_cb_peak.append(-np.pi)
            else:
                list_theta_err_cb_peak.append(theta_cb - theta_mc)

            if math.isnan(theta_nn):
                list_theta_err_nn_peak.append(-np.pi)
            else:
                list_theta_err_nn_peak.append(theta_nn - theta_mc)


    else:
        continue

plt.figure()
bins = np.arange(-np.pi / 2, np.pi / 2, 0.01)
plt.xlabel(r"$\theta^{pred}-\theta^{true}$ [rad]")
plt.ylabel("Counts")
plt.title("Error scattering angle")
plt.hist(list_theta_err_cb, bins=bins, histtype=u"step", color="black", label="Cut-Based")
plt.hist(list_theta_err_nn, bins=bins, histtype=u"step", color="blue", label="NeuralNetwork")
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Position resolutions
# - cone apex
# - cone axis

list_apex_x_err_nn = []
list_apex_y_err_nn = []
list_apex_z_err_nn = []

list_apex_x_err_cb = []
list_apex_y_err_cb = []
list_apex_z_err_cb = []

list_axis_x_err_nn = []
list_axis_y_err_nn = []
list_axis_z_err_nn = []

list_axis_x_err_cb = []
list_axis_y_err_cb = []
list_axis_z_err_cb = []

for i in range(ary_mc_truth.shape[0]):
    if ary_mc_truth[i, 0] in [1.0]:
        list_apex_x_err_nn.append(ary_nn_pred[i, 3] - ary_mc_truth[i, 3])
        list_apex_y_err_nn.append(ary_nn_pred[i, 4] - ary_mc_truth[i, 4])
        list_apex_z_err_nn.append(ary_nn_pred[i, 5] - ary_mc_truth[i, 5])

        list_apex_x_err_cb.append(ary_cb_pred[i, 3] - ary_mc_truth[i, 3])
        list_apex_y_err_cb.append(ary_cb_pred[i, 4] - ary_mc_truth[i, 4])
        list_apex_z_err_cb.append(ary_cb_pred[i, 5] - ary_mc_truth[i, 5])

        list_axis_x_err_nn.append(ary_nn_pred[i, 6] - ary_mc_truth[i, 6])
        list_axis_y_err_nn.append(ary_nn_pred[i, 7] - ary_mc_truth[i, 7])
        list_axis_z_err_nn.append(ary_nn_pred[i, 8] - ary_mc_truth[i, 8])

        list_axis_x_err_cb.append(ary_cb_pred[i, 6] - ary_mc_truth[i, 6])
        list_axis_y_err_cb.append(ary_cb_pred[i, 7] - ary_mc_truth[i, 7])
        list_axis_z_err_cb.append(ary_cb_pred[i, 8] - ary_mc_truth[i, 8])

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
bins_x = np.arange(-10.0, 10.0, 0.1)
bins_y = np.arange(-40.0, 40.0, 0.1)
bins_z = np.arange(-10.0, 10.0, 0.1)

axs[0].set_title("Error coneapex.x")
axs[0].set_xlabel(r"$x^{pred}-x^{true}$")
axs[0].set_ylabel("counts")
axs[0].hist(list_apex_x_err_nn, bins=bins_x, histtype=u"step", color="blue", label="NeuralNetwork")
axs[0].hist(list_apex_x_err_cb, bins=bins_x, histtype=u"step", color="black", label="Cut-Based")
# axs[0].legend()

axs[1].set_title("Error coneapex.y")
axs[1].set_xlabel(r"$y^{pred}-y^{true}$")
axs[1].set_ylabel("counts")
axs[1].hist(list_apex_y_err_nn, bins=bins_y, histtype=u"step", color="blue", label="NeuralNetwork")
axs[1].hist(list_apex_y_err_cb, bins=bins_y, histtype=u"step", color="black", label="Cut-Based")
axs[1].legend()

axs[2].set_title("Error coneapex.z")
axs[2].set_xlabel(r"$z^{pred}-z^{true}$")
axs[2].set_ylabel("counts")
axs[2].hist(list_apex_z_err_nn, bins=bins_z, histtype=u"step", color="blue", label="NeuralNetwork")
axs[2].hist(list_apex_z_err_cb, bins=bins_z, histtype=u"step", color="black", label="Cut-Based")
# axs[2].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 6))
bins_x = np.arange(-10.0, 10.0, 0.1)
bins_y = np.arange(-40.0, 40.0, 0.1)
bins_z = np.arange(-10.0, 10.0, 0.1)

axs[0].set_title("Error coneaxis.x")
axs[0].set_xlabel(r"$x^{pred}-x^{true}$")
axs[0].set_ylabel("counts")
axs[0].hist(list_axis_x_err_nn, bins=bins_x, histtype=u"step", color="blue", label="NeuralNetwork")
axs[0].hist(list_axis_x_err_cb, bins=bins_x, histtype=u"step", color="black", label="Cut-Based")
# axs[0].legend()

axs[1].set_title("Error coneaxis.y")
axs[1].set_xlabel(r"$y^{pred}-y^{true}$")
axs[1].set_ylabel("counts")
axs[1].hist(list_axis_y_err_nn, bins=bins_y, histtype=u"step", color="blue", label="NeuralNetwork")
axs[1].hist(list_axis_y_err_cb, bins=bins_y, histtype=u"step", color="black", label="Cut-Based")
axs[1].legend()

axs[2].set_title("Error coneaxis.z")
axs[2].set_xlabel(r"$z^{pred}-z^{true}$")
axs[2].set_ylabel("counts")
axs[2].hist(list_axis_z_err_nn, bins=bins_z, histtype=u"step", color="blue", label="NeuralNetwork")
axs[2].hist(list_axis_z_err_cb, bins=bins_z, histtype=u"step", color="black", label="Cut-Based")
# axs[2].legend()

plt.tight_layout()
plt.show()
"""
"""
# ----------------------------------------------------------------------------------------------------------------------
# Back-projections
from src import MLEMBackprojection

idx_id = ary_cb_pred[:, 0] != 0
idx_ic = ary_mc_truth[:, 0] == 1
ary_cb_pred = ary_cb_pred[idx_id, :]
ary_nn_pred = ary_nn_pred[idx_ic, :]
ary_mc_truth = ary_mc_truth[idx_ic, :]

n = 100000

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[:n, 1],
                                             ary_nn_pred[:n, 2],
                                             ary_nn_pred[:n, 3],
                                             ary_nn_pred[:n, 4],
                                             ary_nn_pred[:n, 5],
                                             ary_nn_pred[:n, 6],
                                             ary_nn_pred[:n, 7],
                                             ary_nn_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "Backprojection NN prediction", "MLEM_backproj_NNPRED_emod")


image = MLEMBackprojection.reconstruct_image(ary_mc_truth[:n, 1],
                                             ary_mc_truth[:n, 2],
                                             ary_mc_truth[:n, 3],
                                             ary_mc_truth[:n, 4],
                                             ary_mc_truth[:n, 5],
                                             ary_mc_truth[:n, 6],
                                             ary_mc_truth[:n, 7],
                                             ary_mc_truth[:n, 8])
MLEMBackprojection.plot_backprojection(image, "Backprojection MC Truth (Ideal Compton)", "MLEM_backproj_MCTRUTH")


image = MLEMBackprojection.reconstruct_image(ary_cb_pred[:n, 1],
                                             ary_cb_pred[:n, 2],
                                             ary_cb_pred[:n, 3],
                                             ary_cb_pred[:n, 4],
                                             ary_cb_pred[:n, 5],
                                             ary_cb_pred[:n, 6],
                                             ary_cb_pred[:n, 7],
                                             ary_cb_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "Backprojection CB identified", "MLEM_backproj_CBiden")

image = MLEMBackprojection.reconstruct_image(ary_mc_truth[:n, 1],
                                             ary_mc_truth[:n, 2],
                                             ary_nn_pred[:n, 3],
                                             ary_nn_pred[:n, 4],
                                             ary_nn_pred[:n, 5],
                                             ary_nn_pred[:n, 6],
                                             ary_nn_pred[:n, 7],
                                             ary_nn_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "Backprojection NN energy corrected", "MLEM_backproj_NNPRED_energycorrect")


image = MLEMBackprojection.reconstruct_image(ary_nn_pred[:n, 1],
                                             ary_nn_pred[:n, 2],
                                             ary_mc_truth[:n, 3],
                                             ary_mc_truth[:n, 4],
                                             ary_mc_truth[:n, 5],
                                             ary_mc_truth[:n, 6],
                                             ary_mc_truth[:n, 7],
                                             ary_mc_truth[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_NNPRED_positioncorrect")


image = MLEMBackprojection.reconstruct_image(ary_nn_pred[:n, 1],
                                             ary_nn_pred[:n, 2],
                                             ary_mc_truth[:n, 3],
                                             ary_mc_truth[:n, 4],
                                             ary_mc_truth[:n, 5],
                                             ary_mc_truth[:n, 6],
                                             ary_mc_truth[:n, 7],
                                             ary_mc_truth[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_NNPRED_positioncorrect")



image = MLEMBackprojection.reconstruct_image(ary_cb_pred[:n, 1],
                                             ary_cb_pred[:n, 2],
                                             ary_cb_pred[:n, 3],
                                             ary_nn_pred[:n, 4],
                                             ary_cb_pred[:n, 5],
                                             ary_cb_pred[:n, 6],
                                             ary_nn_pred[:n, 7],
                                             ary_cb_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_NNPRED_optimal")
"""
