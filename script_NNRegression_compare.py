import numpy as np
import math
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})


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
    ary_mc_true = np.zeros(shape=(n_entries, 9))

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
    y_pred = neuralnetwork_clas.predict(npz_features)

    ary_nn_pred[:, 0] = np.reshape(y_pred, newshape=(len(y_pred),))
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
                            cb_pred=ary_cb_pred,
                            mc_truth=ary_mc_true,
                            source_position=ary_root_source_position)


# ----------------------------------------------------------------------------------------------------------------------
# Analysis script

# Grab all information from the target file
npz_data = np.load("OptimisedGeometry_BP0mm_statistics.npz")
ary_identified = npz_data["identified"]
ary_nn_pred = npz_data["nn_pred"]
ary_cb_pred = npz_data["cb_pred"]
ary_mc_truth = npz_data["mc_truth"]
ary_sp = npz_data["source_position"]

# ----------------------------------------------------------------------------------------------------------------------
# distribution of scattering angle
from src import MLEMBackprojection

"""
list_theta_err_cb = []
list_theta_err_cb_peak = []
list_theta_err_nn = []
list_theta_err_nn_peak = []

for i in range(ary_mc_truth.shape[0]):
    if ary_mc_truth[i, 0] == 1.0:
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

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
bins = np.arange(-np.pi / 2, np.pi / 2, 0.01)
axs[0].set_xlabel(r"$\theta^{pred}-\theta^{true}$ [rad]")
axs[0].set_ylabel("Counts")
axs[0].set_title("Ideal Compton events")
axs[0].hist(list_theta_err_cb, bins=bins, histtype=u"step", color="black")
axs[0].hist(list_theta_err_nn, bins=bins, histtype=u"step", color="blue")
axs[1].set_xlabel(r"$\theta^{pred}-\theta^{true}$ [rad]")
axs[1].set_ylabel("Counts")
axs[1].set_title("Ideal Compton\n Bragg peak events")
axs[1].hist(list_theta_err_cb_peak, bins=bins, histtype=u"step", color="black")
axs[1].hist(list_theta_err_nn_peak, bins=bins, histtype=u"step", color="blue")
plt.tight_layout()
plt.show()
"""
# ----------------------------------------------------------------------------------------------------------------------
# Back-projections

idx_ic = ary_mc_truth[:, 0] == 1
ary_mc_truth = ary_mc_truth[idx_ic, :]

n= -1
image = MLEMBackprojection.reconstruct_image(ary_mc_truth[:n, 1],
                                             ary_mc_truth[:n, 2],
                                             ary_mc_truth[:n, 3],
                                             ary_mc_truth[:n, 4],
                                             ary_mc_truth[:n, 5],
                                             ary_mc_truth[:n, 6],
                                             ary_mc_truth[:n, 7],
                                             ary_mc_truth[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_MCTRUTH")

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[:n, 1],
                                             ary_nn_pred[:n, 2],
                                             ary_nn_pred[:n, 3],
                                             ary_nn_pred[:n, 4],
                                             ary_nn_pred[:n, 5],
                                             ary_nn_pred[:n, 6],
                                             ary_nn_pred[:n, 7],
                                             ary_nn_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_NNPRED")

image = MLEMBackprojection.reconstruct_image(ary_cb_pred[:n, 1],
                                             ary_cb_pred[:n, 2],
                                             ary_cb_pred[:n, 3],
                                             ary_cb_pred[:n, 4],
                                             ary_cb_pred[:n, 5],
                                             ary_cb_pred[:n, 6],
                                             ary_cb_pred[:n, 7],
                                             ary_cb_pred[:n, 8])
MLEMBackprojection.plot_backprojection(image, "MLEM_backproj_CBPRED")