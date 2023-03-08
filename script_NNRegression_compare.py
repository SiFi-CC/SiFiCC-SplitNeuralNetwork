import numpy as np
import math
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

from src import MLEMBackprojection

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"

# ----------------------------------------------------------------------------------------------------------------------
# Analysis script
"""
npz_nn_0mm = np.load(
    dir_results + "DNN_S1AX_continuous_master/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz")
npz_nn_5mm = np.load(
    dir_results + "DNN_S1AX_continuous_master/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz")

ary_nn_0mm = npz_nn_0mm["NN_PRED"]
ary_nn_5mm = npz_nn_5mm["NN_PRED"]

npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")
ary_meta_0mm = npz_lookup_0mm["META"]
ary_meta_5mm = npz_lookup_5mm["META"]
ary_mc_0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_5mm = npz_lookup_5mm["MC_TRUTH"]
ary_cb_0mm = npz_lookup_0mm["CB_RECO"]
ary_cb_5mm = npz_lookup_5mm["CB_RECO"]

n = 30000

ary_nn_0mm = ary_nn_0mm[:n]
ary_nn_5mm = ary_nn_5mm[:n]
ary_meta_0mm = ary_meta_0mm[:n]
ary_meta_5mm = ary_meta_5mm[:n]
ary_mc_0mm = ary_mc_0mm[:n]
ary_mc_5mm = ary_mc_5mm[:n]
ary_cb_0mm = ary_cb_0mm[:n]
ary_cb_5mm = ary_cb_5mm[:n]

idx_pos_0mm = ary_nn_0mm[:, 0] > 0.3
idx_pos_5mm = ary_nn_5mm[:, 0] > 0.3

idx_identified_0mm = ary_meta_0mm[:, 3] != 0
idx_identified_5mm = ary_meta_5mm[:, 3] != 0

idx_ic_0mm = ary_meta_0mm[:, 2] == 1
idx_ic_5mm = ary_meta_5mm[:, 2] == 1

image_0mm = MLEMBackprojection.reconstruct_image(ary_nn_0mm[idx_pos_0mm, 1],
                                                 ary_nn_0mm[idx_pos_0mm, 2],
                                                 ary_nn_0mm[idx_pos_0mm, 3],
                                                 ary_nn_0mm[idx_pos_0mm, 4],
                                                 ary_nn_0mm[idx_pos_0mm, 5],
                                                 ary_nn_0mm[idx_pos_0mm, 6],
                                                 ary_nn_0mm[idx_pos_0mm, 7],
                                                 ary_nn_0mm[idx_pos_0mm, 8],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_nn_5mm[idx_pos_5mm, 1],
                                                 ary_nn_5mm[idx_pos_5mm, 2],
                                                 ary_nn_5mm[idx_pos_5mm, 3],
                                                 ary_nn_5mm[idx_pos_5mm, 4],
                                                 ary_nn_5mm[idx_pos_5mm, 5],
                                                 ary_nn_5mm[idx_pos_5mm, 6],
                                                 ary_nn_5mm[idx_pos_5mm, 7],
                                                 ary_nn_5mm[idx_pos_5mm, 8],
                                                 apply_filter=True)
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "S1AX_nnpred_base",
                                               "MLEMbackproj_S1AX_continuous_base_theta03")

image_0mm = MLEMBackprojection.reconstruct_image(ary_cb_0mm[idx_pos_0mm, 1],
                                                 ary_cb_0mm[idx_pos_0mm, 2],
                                                 ary_cb_0mm[idx_pos_0mm, 3],
                                                 ary_cb_0mm[idx_pos_0mm, 4],
                                                 ary_cb_0mm[idx_pos_0mm, 5],
                                                 ary_cb_0mm[idx_pos_0mm, 6],
                                                 ary_cb_0mm[idx_pos_0mm, 7],
                                                 ary_cb_0mm[idx_pos_0mm, 8],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_cb_5mm[idx_pos_5mm, 1],
                                                 ary_cb_5mm[idx_pos_5mm, 2],
                                                 ary_cb_5mm[idx_pos_5mm, 3],
                                                 ary_cb_5mm[idx_pos_5mm, 4],
                                                 ary_cb_5mm[idx_pos_5mm, 5],
                                                 ary_cb_5mm[idx_pos_5mm, 6],
                                                 ary_cb_5mm[idx_pos_5mm, 7],
                                                 ary_cb_5mm[idx_pos_5mm, 8],
                                                 apply_filter=True)
MLEMBackprojection.plot_backprojection_stacked(image_0mm, image_5mm,
                                               "S1AX_nnpred_base",
                                               "MLEMbackproj_S1AX_continuous_reco_theta05")
"""
"""
# ----------------------------------------------------------------------------------------------------------------------
# Continuous source position backprojection
from src import MLEMBackprojection

npz_data = np.load("S1AX_continuous_train_flip.npz")
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

idx_pos = ary_nn_pred[:, 0] > 0.5
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
MLEMBackprojection.plot_backprojection(image, "Backprojection NN prediction", "MLEM_backproj_S1AX_nnpred_continuous_flip")

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[idx_pos, 1],
                                             ary_nn_pred[idx_pos, 2],
                                             ary_nn_pred[idx_pos, 3]-0.63,
                                             ary_nn_pred[idx_pos, 4],
                                             ary_nn_pred[idx_pos, 5]+4.4,
                                             ary_nn_pred[idx_pos, 6]-1.05,
                                             ary_nn_pred[idx_pos, 7]-0.96,
                                             ary_nn_pred[idx_pos, 8]+0.41,
                                             apply_filter=True)
MLEMBackprojection.plot_backprojection(image, "Backprojection NN prediction", "MLEM_backproj_S1AX_nnpred_continuous_flip_corrected")
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
# Backprojection Stacked from Toy datasets

list_files = ["S1AX_continuous_an_pur10_toy.npz",
              "S1AX_continuous_an_pur02_toy.npz",
              "S1AX_continuous_an_pur015_toy.npz",
              "S1AX_continuous_an_pur01_toy.npz",
              "S1AX_continuous_an_pur005_toy.npz",
              "S1AX_continuous_an_pur00_toy.npz"]

list_labels = ["fPur = 1.0",
               "fPur = 0.2",
               "fPur = 0.15",
               "fPur = 0.1",
               "fPur = 0.05",
               "fPur = 0.0"]

npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")
ary_meta_0mm = npz_lookup_0mm["META"]
ary_meta_5mm = npz_lookup_5mm["META"]
ary_mc_0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_5mm = npz_lookup_5mm["MC_TRUTH"]
ary_cb_0mm = npz_lookup_0mm["CB_RECO"]
ary_cb_5mm = npz_lookup_5mm["CB_RECO"]

n = 30000
ary_meta_0mm = ary_meta_0mm[:n]
ary_meta_5mm = ary_meta_5mm[:n]
ary_mc_0mm = ary_mc_0mm[:n]
ary_mc_5mm = ary_mc_5mm[:n]
ary_cb_0mm = ary_cb_0mm[:n]
ary_cb_5mm = ary_cb_5mm[:n]

list_images = []
for i in range(len(list_files)):
    npz_nn = np.load(dir_toy + "S1AX_continuous_an/" + list_files[i])

    ary_nn_0mm = npz_nn["NN_PRED_0MM"]
    # ary_nn_5mm = npz_nn["NN_PRED_5MM"]
    ary_nn_0mm = ary_nn_0mm[:n]
    # ary_nn_5mm = ary_nn_5mm[:n]

    idx_pos = ary_nn_0mm[:, 0] > 0.5

    image = MLEMBackprojection.reconstruct_image(ary_nn_0mm[idx_pos, 1],
                                                 ary_nn_0mm[idx_pos, 2],
                                                 ary_nn_0mm[idx_pos, 3],
                                                 ary_nn_0mm[idx_pos, 4],
                                                 ary_nn_0mm[idx_pos, 5],
                                                 ary_nn_0mm[idx_pos, 6],
                                                 ary_nn_0mm[idx_pos, 7],
                                                 ary_nn_0mm[idx_pos, 8],
                                                 apply_filter=True)
    print("Image created for : ", list_files[i])
    list_images.append(image)

MLEMBackprojection.plot_backprojection_stacked(list_images, list_labels, "DNN S1AX NN Pred fPur stacked",
                                               "MLEM_backproj_S1AX_continuous_an_stacked_fpur")
