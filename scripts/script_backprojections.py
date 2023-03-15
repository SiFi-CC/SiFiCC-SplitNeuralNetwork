import numpy as np
import math
import os
import matplotlib.pyplot as plt
from src import MLEMBackprojection

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"


# ----------------------------------------------------------------------------------------------------------------------

def backproj_neuralnetwork_bias(run_name,
                                file_name,
                                plot_title,
                                n=100):
    str_npz = dir_results + run_name
    npz_data_cont = np.load(
        str_npz + "/OptimisedGeometry_Continuous_2e10protons_DNN_S1AX/OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz")
    npz_data_bp0mm = np.load(
        str_npz + "/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz")
    npz_data_bp5mm = np.load(
        str_npz + "/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz")

    ary_nn_pred_cont = npz_data_cont["NN_PRED"]
    ary_nn_pred_bp0mm = npz_data_bp0mm["NN_PRED"]
    ary_nn_pred_bp5mm = npz_data_bp5mm["NN_PRED"]

    # define index sequence determining all positive selected events
    idx_pos_cont = np.arange(0, len(ary_nn_pred_cont), 1.0, dtype=int)
    idx_pos_bp0mm = np.arange(0, len(ary_nn_pred_bp0mm), 1.0, dtype=int)
    idx_pos_bp5mm = np.arange(0, len(ary_nn_pred_bp5mm), 1.0, dtype=int)

    rng = np.random.default_rng()
    rng.shuffle(idx_pos_cont)
    rng.shuffle(idx_pos_bp0mm)
    rng.shuffle(idx_pos_bp5mm)

    idx_pos_cont = idx_pos_cont[:n]
    idx_pos_bp0mm = idx_pos_bp0mm[:n]
    idx_pos_bp5mm = idx_pos_bp5mm[:n]

    idx_pos_cont = idx_pos_cont[ary_nn_pred_cont[idx_pos_cont, 0] < 0.5]
    idx_pos_bp0mm = idx_pos_bp0mm[ary_nn_pred_bp0mm[idx_pos_bp0mm, 0] < 0.5]
    idx_pos_bp5mm = idx_pos_bp5mm[ary_nn_pred_bp5mm[idx_pos_bp5mm, 0] < 0.5]

    print(len(idx_pos_cont))
    print(len(idx_pos_bp0mm))
    print(len(idx_pos_bp5mm))

    image_cont = MLEMBackprojection.reconstruct_image(ary_nn_pred_cont[idx_pos_cont, 1],
                                                      ary_nn_pred_cont[idx_pos_cont, 2],
                                                      ary_nn_pred_cont[idx_pos_cont, 3],
                                                      ary_nn_pred_cont[idx_pos_cont, 4],
                                                      ary_nn_pred_cont[idx_pos_cont, 5],
                                                      ary_nn_pred_cont[idx_pos_cont, 6],
                                                      ary_nn_pred_cont[idx_pos_cont, 7],
                                                      ary_nn_pred_cont[idx_pos_cont, 8],
                                                      apply_filter=True)

    image_bp0mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp0mm[idx_pos_bp0mm, 1],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 2],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 3],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 4],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 5],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 6],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 7],
                                                       ary_nn_pred_bp0mm[idx_pos_bp0mm, 8],
                                                       apply_filter=True)

    image_bp5mm = MLEMBackprojection.reconstruct_image(ary_nn_pred_bp5mm[idx_pos_bp5mm, 1],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 2],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 3],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 4],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 5],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 6],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 7],
                                                       ary_nn_pred_bp5mm[idx_pos_bp5mm, 8],
                                                       apply_filter=True)

    MLEMBackprojection.plot_backprojection_stacked([image_cont, image_bp0mm, image_bp5mm],
                                                   ["Continuous", "BP0mm", "BP5mm"],
                                                   plot_title,
                                                   dir_plots + file_name)


# ----------------------------------------------------------------------------------------------------------------------
# Base image back-projection of Monte-Carlo truth
"""
npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")

ary_mc_0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_5mm = npz_lookup_5mm["MC_TRUTH"]

ary_mc_0mm = ary_mc_0mm[ary_mc_0mm[:, 0] == 1, :]
ary_mc_5mm = ary_mc_5mm[ary_mc_5mm[:, 0] == 1, :]

n = 100000
image_0mm = MLEMBackprojection.reconstruct_image(ary_mc_0mm[:n, 1],
                                                 ary_mc_0mm[:n, 2],
                                                 ary_mc_0mm[:n, 3],
                                                 ary_mc_0mm[:n, 4],
                                                 ary_mc_0mm[:n, 5],
                                                 ary_mc_0mm[:n, 6],
                                                 ary_mc_0mm[:n, 7],
                                                 ary_mc_0mm[:n, 8],
                                                 apply_filter=True)

image_5mm = MLEMBackprojection.reconstruct_image(ary_mc_5mm[:n, 1],
                                                 ary_mc_5mm[:n, 2],
                                                 ary_mc_5mm[:n, 3],
                                                 ary_mc_5mm[:n, 4],
                                                 ary_mc_5mm[:n, 5],
                                                 ary_mc_5mm[:n, 6],
                                                 ary_mc_5mm[:n, 7],
                                                 ary_mc_5mm[:n, 8],
                                                 apply_filter=True)

MLEMBackprojection.plot_backprojection_dual(image_0mm, image_5mm, "", dir_plots + "MLEM_backproj_S1AX_MC_truth_BP0MMBP5MM")
"""
# ----------------------------------------------------------------------------------------------------------------------
# Example
"""
npz_nn_0mm = np.load(
    dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz")
npz_nn_5mm = np.load(
    dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz")

ary_nn_0mm = npz_nn_0mm["NN_PRED"]
ary_nn_5mm = npz_nn_5mm["NN_PRED"]

n = 30000

ary_nn_0mm = ary_nn_0mm[:n]
ary_nn_5mm = ary_nn_5mm[:n]
idx_pos_0mm = ary_nn_0mm[:, 0] > 0.5
idx_pos_5mm = ary_nn_5mm[:, 0] > 0.5

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
MLEMBackprojection.plot_backprojection_dual(image_0mm, image_5mm,
                                            "",
                                            dir_plots + "MLEMbackproj_S1AX_continuous_an")
"""
# ----------------------------------------------------------------------------------------------------------------------
# Example Continuous source distribution analysis
# Continuous source position back-projection
"""
n = 100000

npz_data = np.load(
    dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_Continuous_2e10protons_DNN_S1AX/OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz")
ary_nn_pred = npz_data["NN_PRED"]
ary_nn_pred = ary_nn_pred[:n]
idx_pos = ary_nn_pred[:, 0] > 0.5

image = MLEMBackprojection.reconstruct_image(ary_nn_pred[idx_pos, 1],
                                             ary_nn_pred[idx_pos, 2],
                                             ary_nn_pred[idx_pos, 3],
                                             ary_nn_pred[idx_pos, 4],
                                             ary_nn_pred[idx_pos, 5],
                                             ary_nn_pred[idx_pos, 6],
                                             ary_nn_pred[idx_pos, 7],
                                             ary_nn_pred[idx_pos, 8],
                                             apply_filter=True)
MLEMBackprojection.plot_backprojection(image, "",
                                       dir_plots + "MLEMbackproj_S1AX_continuous_an_nnpred_continuous")
"""

# ----------------------------------------------------------------------------------------------------------------------

backproj_neuralnetwork_bias("DNN_S1AX_continuous_an",
                            "MLEMBackproj_DNN_S1AX_continuous_an_stackedbias",
                            "",
                            n=40000)
