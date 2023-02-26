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
    RUN_NAME = "DNN_S1AX"
    RUN_TAG = "continuous"
    LOOKUP_NAME_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz"
    LOOKUP_NAME_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz"

    SAMPLE_NAME_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz"
    SAMPLE_NAME_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz"

    # ----------------------------------------------
    # load up all required datasets
    # - Lookup file for MC-Truth and CB-Reco
    # - NN predictions

    # Lookup datasets
    npz_lookup_bp0mm = np.load(dir_npz + LOOKUP_NAME_BP0mm)
    npz_lookup_bp5mm = np.load(dir_npz + LOOKUP_NAME_BP5mm)

    ary_mc_truth_bp0mm = npz_lookup_bp0mm["MC_TRUTH"]
    ary_mc_truth_bp5mm = npz_lookup_bp5mm["MC_TRUTH"]
    ary_mc_truth_bp0mm_meta = npz_lookup_bp0mm["META"]
    ary_mc_truth_bp5mm_meta = npz_lookup_bp5mm["META"]

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

    str_savefile = "S1AX_continuous.npz"
    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            mc_truth_0mm=ary_mc_truth_bp0mm,
                            mc_truth_5mm=ary_mc_truth_bp5mm,
                            cb_reco_0mm=ary_cb_reco_bp0mm,
                            cb_reco_5mm=ary_cb_reco_bp5mm,
                            nn_pred_0mm=ary_nn_pred_bp0mm,
                            nn_pred_5mm=ary_nn_pred_bp5mm)


def generate_export_temp():
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
    RUN_NAME = "DNN_S1AX"
    RUN_TAG = "continuous"
    LOOKUP_NAME = "OptimisedGeometry_Continuous_2e10protons_lookup.npz"
    SAMPLE_NAME = "OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz"

    # ----------------------------------------------
    # load up all required datasets
    # - Lookup file for MC-Truth and CB-Reco
    # - NN predictions

    # Lookup datasets
    npz_lookup_bp0mm = np.load(dir_npz + LOOKUP_NAME)

    ary_mc_truth_bp0mm = npz_lookup_bp0mm["MC_TRUTH"]
    ary_mc_truth_bp0mm_meta = npz_lookup_bp0mm["META"]
    ary_cb_reco_bp0mm = npz_lookup_bp0mm["CB_RECO"]

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

    DataCluster_BP0mm = NPZParser.wrapper(dir_npz + SAMPLE_NAME,
                                          standardize=True,
                                          set_classweights=False,
                                          set_peakweights=False)

    ary_nn_pred_bp0mm = np.zeros(shape=(DataCluster_BP0mm.entries, 9))

    # get neural network predictions
    y_pred_bp0mm = neuralnetwork_clas.predict(DataCluster_BP0mm.features)
    ary_nn_pred_bp0mm[:, 0] = np.reshape(y_pred_bp0mm, newshape=(len(y_pred_bp0mm),))
    ary_nn_pred_bp0mm[:, 1:3] = neuralnetwork_regE.predict(DataCluster_BP0mm.features)
    ary_nn_pred_bp0mm[:, 3:9] = neuralnetwork_regP.predict(DataCluster_BP0mm.features)

    str_savefile = "S1AX_continuous_train.npz"
    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            mc_truth=ary_mc_truth_bp0mm,
                            cb_reco=ary_cb_reco_bp0mm,
                            nn_pred=ary_nn_pred_bp0mm)

# ----------------------------------------------------------------------------------------------------------------------
# Analysis script

npz_data = np.load("S1AX_continuous.npz")
ary_nn_pred_bp0mm = npz_data["nn_pred_0mm"]
ary_nn_pred_bp5mm = npz_data["nn_pred_5mm"]

npz_lookup_0mm = np.load(os.getcwd() + "/npz_files/" +
                         "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(os.getcwd() + "/npz_files/" +
                         "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")
ary_meta_0mm = npz_lookup_0mm["META"]
ary_meta_5mm = npz_lookup_5mm["META"]
ary_mc_truth_bp0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_truth_bp5mm = npz_lookup_5mm["MC_TRUTH"]
ary_cb_reco_bp0mm = npz_lookup_0mm["CB_RECO"]
ary_cb_reco_bp5mm = npz_lookup_5mm["CB_RECO"]

"""
# ----------------------------------------------------------------------------------------------------------------------
# Continuous source position backprojection
from src import MLEMBackprojection

npz_data = np.load("S1AX_continuous.npz")
ary_nn_pred = npz_data["nn_pred"]
npz_lookup = np.load(os.getcwd() + "/npz_files/" + "OptimisedGeometry_Continuous_2e10protons_lookup.npz")
ary_meta = npz_lookup["META"]
ary_mc_truth = npz_lookup["MC_TRUTH"]
ary_cb_reco = npz_lookup["CB_RECO"]

n = 100000

ary_nn_pred = ary_nn_pred[:n]
ary_meta = ary_meta[:n]
ary_mc_truth = ary_mc_truth[:n]
ary_cb_reco = ary_cb_reco[:n]

idx_pos = ary_nn_pred[:, 0] > 0.3
idx_identified = ary_meta[:, 3] != 0
idx_ic = ary_meta[:, 2] == 1

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[idx_pos, 1],
                                             ary_nn_pred[idx_pos, 2],
                                             ary_nn_pred[idx_pos, 3],
                                             ary_nn_pred[idx_pos, 4],
                                             ary_nn_pred[idx_pos, 5],
                                             ary_nn_pred[idx_pos, 6],
                                             ary_nn_pred[idx_pos, 7],
                                             ary_nn_pred[idx_pos, 8],
                                             apply_filter=True)
MLEMBackprojection.plot_backprojection(image, "Backprojection NN prediction", "MLEM_backproj_S1AX_nnpred_continuous")

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[idx_pos, 1],
                                             ary_nn_pred[idx_pos, 2],
                                             ary_cb_reco[idx_pos, 2],
                                             ary_cb_reco[idx_pos, 3],
                                             ary_nn_pred[idx_pos, 5],
                                             ary_nn_pred[idx_pos, 6],
                                             ary_nn_pred[idx_pos, 7],
                                             ary_nn_pred[idx_pos, 8],
                                             apply_filter=True)
MLEMBackprojection.plot_backprojection(image, "Backprojection NN prediction", "MLEM_backproj_S1AX_nnpred_continuous_ecorrected")
"""
"""
# ----------------------------------------------------------------------------------------------------------------------
# Loss function estimation
from src import NNLoss

idx_pos_0mm = ary_meta_0mm[:, 2] == 1.0

ary_mc_truth_bp0mm = ary_mc_truth_bp0mm[idx_pos_0mm, :]
ary_nn_pred_bp0mm = ary_nn_pred_bp0mm[idx_pos_0mm, :]
ary_cb_reco_bp0mm = ary_cb_reco_bp0mm[idx_pos_0mm, :]

print("Mean absolute error")
print(NNLoss.loss_energy_mae(ary_nn_pred_bp0mm[:, 1:3], ary_mc_truth_bp0mm[:, 0:2]))
print(NNLoss.loss_energy_mae(ary_cb_reco_bp0mm[:, 0:2], ary_mc_truth_bp0mm[:, 0:2]))
print("\nMean squared error relative")
print(NNLoss.loss_energy_mse_relative(ary_nn_pred_bp0mm[:, 1:3], ary_mc_truth_bp0mm[:, 0:2]))
print(NNLoss.loss_energy_mse_relative(ary_cb_reco_bp0mm[:, 0:2], ary_mc_truth_bp0mm[:, 0:2]))
print("\nMean absolute error asymmetrical")
print(NNLoss.loss_energy_mae_asym(ary_nn_pred_bp0mm[:, 1:3], ary_mc_truth_bp0mm[:, 0:2]))
print(NNLoss.loss_energy_mae_asym(ary_cb_reco_bp0mm[:, 0:2], ary_mc_truth_bp0mm[:, 0:2]))
"""

# ----------------------------------------------------------------------------------------------------------------------
# Backprojection plots
from src import MLEMBackprojection

n = 50000

ary_nn_pred_bp0mm = ary_nn_pred_bp0mm[:n]
ary_nn_pred_bp5mm = ary_nn_pred_bp5mm[:n]
ary_meta_0mm = ary_meta_0mm[:n]
ary_meta_5mm = ary_meta_5mm[:n]
ary_mc_truth_bp0mm = ary_mc_truth_bp0mm[:n]
ary_mc_truth_bp5mm = ary_mc_truth_bp5mm[:n]
ary_cb_reco_bp0mm = ary_cb_reco_bp0mm[:n]
ary_cb_reco_bp5mm = ary_cb_reco_bp5mm[:n]

idx_pos_0mm = ary_nn_pred_bp0mm[:, 0] > 0.5
idx_pos_5mm = ary_nn_pred_bp5mm[:, 0] > 0.5

idx_identified_0mm = ary_meta_0mm[:, 3] != 0
idx_identified_5mm = ary_meta_5mm[:, 3] != 0

idx_ic_0mm = ary_meta_0mm[:, 2] == 1
idx_ic_5mm = ary_meta_5mm[:, 2] == 1

image_0mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp0mm[idx_pos_0mm, 1],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 2],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 3],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 4],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 5],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 6],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 7],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 8],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp5mm[idx_pos_5mm, 1],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 2],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 3],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 4],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 5],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 6],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 7],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 8],
                                                 apply_filter=True)
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection NN prediction", "MLEM_backproj_S1AX_continuous_theta05")
"""
image_0mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp0mm[idx_pos_0mm, 1],
                                                 ary_nn_pred_bp0mm[idx_pos_0mm, 2],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 2],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 3],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 4],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 5],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 6],
                                                 ary_cb_reco_bp0mm[idx_pos_0mm, 7],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp5mm[idx_pos_5mm, 1],
                                                 ary_nn_pred_bp5mm[idx_pos_5mm, 2],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 2],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 3],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 4],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 5],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 6],
                                                 ary_cb_reco_bp5mm[idx_pos_5mm, 7],
                                                 apply_filter=True)
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection CB identified", "MLEM_backproj_S1AX_theta03_classical")

image_0mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp0mm[idx_ic_0mm, 1],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 2],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 3],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 4],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 5],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 6],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 7],
                                                 ary_nn_pred_bp0mm[idx_ic_0mm, 8],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp5mm[idx_ic_5mm, 1],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 2],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 3],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 4],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 5],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 6],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 7],
                                                 ary_nn_pred_bp5mm[idx_ic_5mm, 8],
                                                 apply_filter=True)
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection CB identified", "MLEM_backproj_S1AX_theta03_idealcompton")

image_0mm = MLEMBackprojection.reconstruct_image(ary_cb_reco_bp0mm[:n, 0],
                                                 ary_cb_reco_bp0mm[:n, 1],
                                                 ary_nn_pred_bp0mm[:n, 3],
                                                 ary_nn_pred_bp0mm[:n, 4],
                                                 ary_nn_pred_bp0mm[:n, 5],
                                                 ary_nn_pred_bp0mm[:n, 6],
                                                 ary_nn_pred_bp0mm[:n, 7],
                                                 ary_nn_pred_bp0mm[:n, 8])

image_5mm = MLEMBackprojection.reconstruct_image(ary_cb_reco_bp5mm[:n, 0],
                                                 ary_cb_reco_bp5mm[:n, 1],
                                                 ary_nn_pred_bp5mm[:n, 3],
                                                 ary_nn_pred_bp5mm[:n, 4],
                                                 ary_nn_pred_bp5mm[:n, 5],
                                                 ary_nn_pred_bp5mm[:n, 6],
                                                 ary_nn_pred_bp5mm[:n, 7],
                                                 ary_nn_pred_bp5mm[:n, 8])
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "Backprojection CB identified", "MLEM_backproj_S1AX_theta05_ecorrect")



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
                                               "Backprojection CB identified", "MLEM_backproj_S1AX_theta05_fullcorrect")

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
                                               "Backprojection CB identified", "MLEM_backproj_S1AX_CBreco")
"""
