import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt
from src import MLEMBackprojection

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"


# ----------------------------------------------------------------------------------------------------------------------


def cut_based_new():
    npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
    npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")

    ary_meta_0mm = npz_lookup_0mm["META"]
    ary_meta_5mm = npz_lookup_5mm["META"]
    ary_tag_0mm = npz_lookup_0mm["TAGS"]
    ary_tag_5mm = npz_lookup_5mm["TAGS"]
    ary_cb_0mm = npz_lookup_0mm["CB_RECO"]
    ary_cb_5mm = npz_lookup_5mm["CB_RECO"]

    ary_tag_realcoinc_0mm = np.zeros(shape=(len(ary_meta_0mm, )))
    ary_tag_realcoinc_5mm = np.zeros(shape=(len(ary_meta_5mm, )))
    ary_tag_compton_0mm = np.zeros(shape=(len(ary_meta_0mm, )))
    ary_tag_compton_5mm = np.zeros(shape=(len(ary_meta_5mm, )))
    ary_tag_pseudocomplete_0mm = np.zeros(shape=(len(ary_meta_0mm, )))
    ary_tag_pseudocomplete_5mm = np.zeros(shape=(len(ary_meta_5mm, )))
    ary_tag_pseudodist_0mm = np.zeros(shape=(len(ary_meta_0mm, )))
    ary_tag_pseudodist_5mm = np.zeros(shape=(len(ary_meta_5mm, )))
    ary_tag_dist_0mm = np.zeros(shape=(len(ary_meta_0mm, )))
    ary_tag_dist_5mm = np.zeros(shape=(len(ary_meta_5mm, )))

    for i in range(len(ary_meta_0mm)):
        if ary_tag_0mm[i, 0] == 1 and ary_tag_0mm[i, 1] == 0:
            ary_tag_realcoinc_0mm[i] = 1
        if ary_tag_0mm[i, 1] == 1 and ary_tag_0mm[i, 2] == 0:
            ary_tag_compton_0mm[i] = 1
        if ary_tag_0mm[i, 2] == 1 and ary_tag_0mm[i, 3] == 0:
            ary_tag_pseudocomplete_0mm[i] = 1
        if ary_tag_0mm[i, 3] == 1 and ary_tag_0mm[i, 4] == 0:
            ary_tag_pseudodist_0mm[i] = 1

    for i in range(len(ary_meta_5mm)):
        if ary_tag_5mm[i, 0] == 1 and ary_tag_5mm[i, 1] == 0:
            ary_tag_realcoinc_5mm[i] = 1
        if ary_tag_5mm[i, 1] == 1 and ary_tag_5mm[i, 2] == 0:
            ary_tag_compton_5mm[i] = 1
        if ary_tag_5mm[i, 2] == 1 and ary_tag_5mm[i, 3] == 0:
            ary_tag_pseudocomplete_5mm[i] = 1
        if ary_tag_5mm[i, 3] == 1 and ary_tag_5mm[i, 4] == 0:
            ary_tag_pseudodist_5mm[i] = 1

    f_sample_0mm = 0.025
    f_sample_5mm = 0.125

    proj0 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_0mm, ary_tag_realcoinc_0mm,
                                                                   f_sample=f_sample_0mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj1 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_5mm, ary_tag_realcoinc_5mm,
                                                                   f_sample=f_sample_5mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj2 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_0mm, ary_tag_compton_0mm,
                                                                   f_sample=f_sample_0mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj3 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_5mm, ary_tag_compton_5mm,
                                                                   f_sample=f_sample_5mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj4 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_0mm, ary_tag_pseudocomplete_0mm,
                                                                   f_sample=f_sample_0mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj5 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_5mm, ary_tag_pseudocomplete_5mm,
                                                                   f_sample=f_sample_5mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj6 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_0mm, ary_tag_pseudodist_0mm,
                                                                   f_sample=f_sample_0mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)
    proj7 = MLEMBackprojection.get_backprojection_cbreco_optimized(ary_cb_5mm, ary_tag_pseudodist_5mm,
                                                                   f_sample=f_sample_5mm, n_subsample=5, scatz=60.0,
                                                                   verbose=1)

    MLEMBackprojection.plot_backprojection_dual([proj0, proj2, proj4, proj6],
                                                [proj1, proj3, proj5, proj7],
                                                ["Real Coincidence",
                                                 "Compton",
                                                 "Compton pseudo complete",
                                                 "Compton pseudo distributed"],
                                                "",
                                                dir_plots + "MLEM_backproj_cbreco_id_tagging_inclusive_5e8")


def backprojection_neuralnetwork(run_name):
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

    f_sample_0mm = 0.01
    f_sample_5mm = 0.05

    proj_0mm = MLEMBackprojection.get_backprojection_nnpred_optimized(ary_nn_pred_bp0mm, ary_nn_pred_bp0mm[:, 0],
                                                                      theta=0.5,
                                                                      f_sample=f_sample_0mm, n_subsample=5, scatz=100.0,
                                                                      verbose=1)
    proj_5mm = MLEMBackprojection.get_backprojection_nnpred_optimized(ary_nn_pred_bp5mm, ary_nn_pred_bp5mm[:, 0],
                                                                      theta=0.5,
                                                                      f_sample=f_sample_5mm, n_subsample=5, scatz=100.0,
                                                                      verbose=1)

    MLEMBackprojection.plot_backprojection_dual([proj_0mm],
                                                [proj_5mm],
                                                ["DNN S1AX"],
                                                "",
                                                dir_plots + "IRBP_nnpred_test_2e8protons")

backprojection_neuralnetwork("DNN_S1AX_newtagging")

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
"""
backproj_neuralnetwork_bias("DNN_S1AX_oldnorm",
                            "MLEMBackproj_DNN_S1AX_oldnorm_stackedbias",
                            "",
                            n=40000)
"""

"""
toyset_stacked("S1AX_continuous_an",
               "",
               dir_plots + "MLEMbackproj_S1AX_continuous_an_tn00_fn00_stacked",
               ["S1AX_continuous_an_base_toy.npz", "S1AX_continuous_an_tn00_toy.npz", "S1AX_continuous_an_fn00_toy.npz"],
               ["base", "ftn = 0.0", "ffn = 0.0"],
               n=40000)
"""
"""
cut_based_tagging("",
                  dir_plots + "MLEMbackproj_cb_reco_tagging_stacked",
                  ["Compton", "Complete Compton", "Complete dist. Compton", "Ideal Compton", "Full Compton**"],
                  n=100000)
"""
"""

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

    idx_pos_cont = idx_pos_cont[ary_nn_pred_cont[idx_pos_cont, 0] > 0.5]
    idx_pos_bp0mm = idx_pos_bp0mm[ary_nn_pred_bp0mm[idx_pos_bp0mm, 0] > 0.5]
    idx_pos_bp5mm = idx_pos_bp5mm[ary_nn_pred_bp5mm[idx_pos_bp5mm, 0] > 0.5]

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


def toyset_stacked(toy_name,
                   plot_title,
                   plot_name,
                   list_files,
                   list_labels,
                   n=100):
    list_images_0mm = []
    list_images_5mm = []

    for i in range(len(list_files)):
        # load toy dataset
        npz_toy = np.load(dir_toy + toy_name + "/" + list_files[i])

        ary_toy_0mm = npz_toy["NN_PRED_0MM"]
        ary_toy_5mm = npz_toy["NN_PRED_5MM"]

        rng = np.random.default_rng()
        ary_idx_0mm = np.arange(0, len(ary_toy_0mm), 1.0, dtype=int)
        ary_idx_5mm = np.arange(0, len(ary_toy_5mm), 1.0, dtype=int)
        rng.shuffle(ary_idx_0mm)
        rng.shuffle(ary_idx_5mm)
        ary_toy_0mm = ary_toy_0mm[ary_idx_0mm, :]
        ary_toy_5mm = ary_toy_5mm[ary_idx_5mm, :]

        ary_toy_0mm = ary_toy_0mm[:n, :]
        ary_toy_5mm = ary_toy_5mm[:n, :]

        idx_pos_0mm = ary_toy_0mm[:, 0] > 0.5
        idx_pos_5mm = ary_toy_5mm[:, 0] > 0.5

        image_0mm = MLEMBackprojection.reconstruct_image(ary_toy_0mm[idx_pos_0mm, 1],
                                                         ary_toy_0mm[idx_pos_0mm, 2],
                                                         ary_toy_0mm[idx_pos_0mm, 3],
                                                         ary_toy_0mm[idx_pos_0mm, 4],
                                                         ary_toy_0mm[idx_pos_0mm, 5],
                                                         ary_toy_0mm[idx_pos_0mm, 6],
                                                         ary_toy_0mm[idx_pos_0mm, 7],
                                                         ary_toy_0mm[idx_pos_0mm, 8],
                                                         apply_filter=True)

        image_5mm = MLEMBackprojection.reconstruct_image(ary_toy_5mm[idx_pos_5mm, 1],
                                                         ary_toy_5mm[idx_pos_5mm, 2],
                                                         ary_toy_5mm[idx_pos_5mm, 3],
                                                         ary_toy_5mm[idx_pos_5mm, 4],
                                                         ary_toy_5mm[idx_pos_5mm, 5],
                                                         ary_toy_5mm[idx_pos_5mm, 6],
                                                         ary_toy_5mm[idx_pos_5mm, 7],
                                                         ary_toy_5mm[idx_pos_5mm, 8],
                                                         apply_filter=True)

        print("Image created for : ", list_files[i])
        list_images_0mm.append(image_0mm)
        list_images_5mm.append(image_5mm)

    MLEMBackprojection.plot_backprojection_stacked_dual(list_images_0mm, list_images_5mm, list_labels,
                                                        plot_title, plot_name)


def cut_based_tagging(plot_title,
                      plot_name,
                      list_labels,
                      n=100):
    list_images_0mm = []
    list_images_5mm = []

    for i in [0, 1, 2, 3, 4]:
        # load toy dataset
        npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
        npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")

        ary_tag_0mm = npz_lookup_0mm["TAGS"]
        ary_tag_5mm = npz_lookup_5mm["TAGS"]
        ary_cb_0mm = npz_lookup_0mm["CB_RECO"]
        ary_cb_5mm = npz_lookup_5mm["CB_RECO"]

        # index selection
        rng = np.random.default_rng()
        ary_idx_0mm = np.arange(0, len(ary_cb_0mm), 1.0, dtype=int)
        ary_idx_5mm = np.arange(0, len(ary_cb_5mm), 1.0, dtype=int)
        rng.shuffle(ary_idx_0mm)
        rng.shuffle(ary_idx_5mm)
        ary_idx_0mm = ary_idx_0mm[:n]
        ary_idx_5mm = ary_idx_5mm[:n]

        idx_pos_0mm = ary_idx_0mm[ary_tag_0mm[ary_idx_0mm, i] == 1]
        idx_pos_5mm = ary_idx_5mm[ary_tag_5mm[ary_idx_5mm, i] == 1]

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

        print("Image created for tag 0{}".format(i))
        list_images_0mm.append(image_0mm)
        list_images_5mm.append(image_5mm)

    MLEMBackprojection.plot_backprojection_stacked_dual(list_images_0mm, list_images_5mm, list_labels,
                                                        plot_title, plot_name)



"""
