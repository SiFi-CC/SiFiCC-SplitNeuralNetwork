import numpy as np
import os

from src import Plotter


def create_toy_set(FILE_NAME,
                   TAG,
                   PATH_NN_PRED_0MM,
                   PATH_NN_PRED_5MM,
                   PATH_MC_TRUTH_0MM,
                   PATH_MC_TRUTH_5MM,
                   f_reg_ee=1.0,
                   f_reg_ep=1.0,
                   f_reg_xe=1.0,
                   f_reg_ye=1.0,
                   f_reg_ze=1.0,
                   f_reg_xp=1.0,
                   f_reg_yp=1.0,
                   f_reg_zp=1.0,
                   f_tp=1.0,
                   f_fp=1.0,
                   f_tn=1.0,
                   f_fn=1.0,
                   mod_bg=False):
    # define directory paths
    dir_main = os.getcwd()
    dir_toy = dir_main + "/toy/"

    if not os.path.isdir(dir_toy + FILE_NAME + "/"):
        os.mkdir(dir_toy + FILE_NAME + "/")

    theta = 0.5

    # ---------------------------------------------------------

    # load neural network prediction and Monte Carlo Truth
    npz_nn_pred_0mm = np.load(PATH_NN_PRED_0MM)
    npz_nn_pred_5mm = np.load(PATH_NN_PRED_5MM)
    ary_nn_pred_0mm = npz_nn_pred_0mm["NN_PRED"]
    ary_nn_pred_5mm = npz_nn_pred_5mm["NN_PRED"]

    npz_mc_truth_0mm = np.load(PATH_MC_TRUTH_0MM)
    npz_mc_truth_5mm = np.load(PATH_MC_TRUTH_5MM)
    ary_mc_truth_0mm = npz_mc_truth_0mm["MC_TRUTH"]
    ary_mc_truth_5mm = npz_mc_truth_5mm["MC_TRUTH"]

    # ---------------------------------------------------------
    # Main loop

    # exception for efficiency and purity factor
    # only one option should be respected
    # therefore at least one needs to be set to 1.0

    list_f = [f_reg_ee, f_reg_ep, f_reg_xe, f_reg_ye, f_reg_ze, f_reg_xp, f_reg_yp, f_reg_zp]
    for i in range(len(ary_nn_pred_0mm)):
        for j in range(len(list_f)):
            if ary_mc_truth_0mm[i, 0] == 1.0:
                ary_nn_pred_0mm[i, j + 1] = ary_mc_truth_0mm[i, j + 1] + list_f[j] * (
                        ary_nn_pred_0mm[i, j + 1] - ary_mc_truth_0mm[i, j + 1])
            else:
                if mod_bg:
                    ary_nn_pred_0mm[i, j + 1] = ary_mc_truth_0mm[i, j + 1] + list_f[j] * (
                            ary_nn_pred_0mm[i, j + 1] - ary_mc_truth_0mm[i, j + 1])
                continue

    for i in range(len(ary_nn_pred_5mm)):
        for j in range(len(list_f)):
            if ary_mc_truth_5mm[i, 0] == 1.0:
                ary_nn_pred_5mm[i, j + 1] = ary_mc_truth_5mm[i, j + 1] + list_f[j] * (
                        ary_nn_pred_5mm[i, j + 1] - ary_mc_truth_5mm[i, j + 1])
            else:
                if mod_bg:
                    ary_nn_pred_5mm[i, j + 1] = ary_mc_truth_5mm[i, j + 1] + list_f[j] * (
                            ary_nn_pred_5mm[i, j + 1] - ary_mc_truth_5mm[i, j] + 1)
                continue

    # -----------------------------------------------------------------------------
    # efficiency and purity manipulation
    # First sort all events by score. Count number of FP and TP events. Scale number
    # of FP/TP events and remove (my modifying score) until scaled number is matched
    ary_idx_0mm = np.arange(0.0, len(ary_nn_pred_0mm), 1.0, dtype=int)
    ary_idx_5mm = np.arange(0.0, len(ary_nn_pred_5mm), 1.0, dtype=int)
    ary_scores_0mm = ary_nn_pred_0mm[:, 0]
    ary_scores_5mm = ary_nn_pred_5mm[:, 0]

    ary_idx_0mm = np.flip(ary_idx_0mm[ary_scores_0mm.argsort()])
    ary_idx_5mm = np.flip(ary_idx_5mm[ary_scores_5mm.argsort()])
    ary_scores_0mm = np.flip(ary_scores_0mm[ary_scores_0mm.argsort()])
    ary_scores_5mm = np.flip(ary_scores_5mm[ary_scores_5mm.argsort()])

    if f_fp != 1.0:

        n_fp = 0
        for i in range(len(ary_nn_pred_0mm)):
            if ary_mc_truth_0mm[i, 0] == 0.0 and ary_nn_pred_0mm[i, 0] > theta:
                n_fp += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 0.0 and ary_nn_pred_0mm[ary_idx_0mm[i], 0] > theta:
                ary_nn_pred_0mm[ary_idx_0mm[i], 0] = 0.0
                counter += 1
            if counter >= n_fp * (1 - f_fp):
                break

        n_fp = 0
        for i in range(len(ary_nn_pred_5mm)):
            if ary_mc_truth_5mm[i, 0] == 0.0 and ary_nn_pred_5mm[i, 0] > theta:
                n_fp += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 0.0 and ary_nn_pred_5mm[ary_idx_5mm[i], 0] > theta:
                ary_nn_pred_5mm[ary_idx_5mm[i], 0] = 0.0
                counter += 1
            if counter >= n_fp * (1 - f_fp):
                break

    if f_tp != 1.0:
        n_tp = 0
        for i in range(len(ary_nn_pred_0mm)):
            if ary_mc_truth_0mm[i, 0] == 1.0 and ary_nn_pred_0mm[i, 0] > theta:
                n_tp += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 1.0 and ary_nn_pred_0mm[ary_idx_0mm[i], 0] > theta:
                ary_nn_pred_0mm[ary_idx_0mm[i], 0] = 0.0
                counter += 1
            if counter >= n_tp * (1 - f_tp):
                break

        n_tp = 0
        for i in range(len(ary_nn_pred_5mm)):
            if ary_mc_truth_5mm[i, 0] == 1.0 and ary_nn_pred_5mm[i, 0] > theta:
                n_tp += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 1.0 and ary_nn_pred_5mm[ary_idx_5mm[i], 0] > theta:
                ary_nn_pred_5mm[ary_idx_5mm[i], 0] = 0.0
                counter += 1
            if counter >= n_tp * (1 - f_tp):
                break

    if f_tn != 1.0:
        n_tn = 0
        for i in range(len(ary_nn_pred_0mm)):
            if ary_mc_truth_0mm[i, 0] == 0.0 and ary_nn_pred_0mm[i, 0] < theta:
                n_tn += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 0.0 and ary_nn_pred_0mm[ary_idx_0mm[i], 0] < theta:
                ary_nn_pred_0mm[ary_idx_0mm[i], 0] = 1.0
                counter += 1
            if counter >= n_tn * (1 - f_tn):
                break

        n_tn = 0
        for i in range(len(ary_nn_pred_5mm)):
            if ary_mc_truth_5mm[i, 0] == 0.0 and ary_nn_pred_5mm[i, 0] < theta:
                n_tn += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 0.0 and ary_nn_pred_5mm[ary_idx_5mm[i], 0] < theta:
                ary_nn_pred_5mm[ary_idx_5mm[i], 0] = 1.0
                counter += 1
            if counter >= n_tn * (1 - f_tn):
                break

    if f_fn != 1.0:
        n_fn = 0
        for i in range(len(ary_nn_pred_0mm)):
            if ary_mc_truth_0mm[i, 0] == 1.0 and ary_nn_pred_0mm[i, 0] < theta:
                n_fn += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 1.0 and ary_nn_pred_0mm[ary_idx_0mm[i], 0] < theta:
                ary_nn_pred_0mm[ary_idx_0mm[i], 0] = 1.0
                counter += 1
            if counter >= n_fn * (1 - f_fn):
                break

        n_fn = 0
        for i in range(len(ary_nn_pred_5mm)):
            if ary_mc_truth_5mm[i, 0] == 1.0 and ary_nn_pred_5mm[i, 0] < theta:
                n_fn += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 1.0 and ary_nn_pred_5mm[ary_idx_5mm[i], 0] < theta:
                ary_nn_pred_5mm[ary_idx_5mm[i], 0] = 1.0
                counter += 1
            if counter >= n_fn * (1 - f_fn):
                break

    # --------------------------------------------------------------
    # Control plot generation
    idx_pos_0mm = ary_nn_pred_0mm[:, 0] > 0.5
    idx_pos_5mm = ary_nn_pred_5mm[:, 0] > 0.5

    os.chdir(dir_toy + FILE_NAME + "/")
    if f_reg_ee != 1.0 or f_reg_ep != 1.0:
        Plotter.plot_energy_error(ary_nn_pred_0mm[idx_pos_0mm, 1:3], ary_mc_truth_0mm[idx_pos_0mm, 1:3],
                                  TAG + "_energy")
        Plotter.plot_energy_error(ary_nn_pred_5mm[idx_pos_5mm, 1:3], ary_mc_truth_5mm[idx_pos_5mm, 1:3],
                                  TAG + "_energy")

    if f_reg_xe != 1.0 or f_reg_ye != 1.0 or f_reg_ze != 1.0 or f_reg_xp != 1.0 or f_reg_yp != 1.0 or f_reg_zp != 1.0:
        Plotter.plot_position_error(ary_nn_pred_0mm[idx_pos_0mm, 3:], ary_mc_truth_0mm[idx_pos_0mm, 3:],
                                    TAG + "_position")
        Plotter.plot_position_error(ary_nn_pred_5mm[idx_pos_5mm, 3:], ary_mc_truth_5mm[idx_pos_5mm, 3:],
                                    TAG + "_position")

    if f_fp != 1.0 or f_tp != 1.0 or f_tn != 1.0 or f_fn != 1.0:
        Plotter.plot_score_dist(ary_nn_pred_0mm[:, 0], ary_mc_truth_0mm[:, 0], TAG + "_score")
        Plotter.plot_score_dist(ary_nn_pred_5mm[:, 0], ary_mc_truth_5mm[:, 0], TAG + "_score")

    os.chdir(dir_main)
    # --------------------------------------------------------------
    # export modified toy dataset
    with open(dir_toy + FILE_NAME + "/" + FILE_NAME + "_" + TAG + "_toy.npz", 'wb') as file:
        np.savez_compressed(file, NN_PRED_0MM=ary_nn_pred_0mm, NN_PRED_5MM=ary_nn_pred_5mm)

    print("file saved: ", FILE_NAME + "_" + TAG + "_toy.npz")


# ----------------------------------------------------------------------------------------------------------------------
# Toy set generator for cut-based approach


def create_toy_set_cutbased(FILE_NAME,
                            TAG,
                            PATH_LOOKUP_0MM,
                            PATH_LOOKUP_5MM,
                            f_reg_ee=1.0,
                            f_reg_ep=1.0,
                            f_reg_xe=1.0,
                            f_reg_ye=1.0,
                            f_reg_ze=1.0,
                            f_reg_xp=1.0,
                            f_reg_yp=1.0,
                            f_reg_zp=1.0,
                            f_fp=1.0,
                            f_tp=1.0,
                            mod_bg=False):
    # define directory paths
    dir_main = os.getcwd()
    dir_toy = dir_main + "/toy/"

    if not os.path.isdir(dir_toy + FILE_NAME + "/"):
        os.mkdir(dir_toy + FILE_NAME + "/")

    # ---------------------------------------------------------

    # load Cut-Based reconstruction and Monte Carlo Truth

    npz_lookup_0mm = np.load(PATH_LOOKUP_0MM)
    npz_lookup_5mm = np.load(PATH_LOOKUP_5MM)
    ary_mc_truth_0mm = npz_lookup_0mm["MC_TRUTH"]
    ary_mc_truth_5mm = npz_lookup_5mm["MC_TRUTH"]
    ary_cb_reco_0mm = npz_lookup_0mm["CB_RECO"]
    ary_cb_reco_5mm = npz_lookup_5mm["CB_RECO"]

    # efficiency and purity manipulation
    ary_idx_0mm = np.arange(0.0, len(ary_cb_reco_0mm), 1.0, dtype=int)
    ary_idx_5mm = np.arange(0.0, len(ary_cb_reco_5mm), 1.0, dtype=int)

    rng = np.random.default_rng(42)
    rng.shuffle(ary_idx_0mm)
    rng.shuffle(ary_idx_5mm)

    # ---------------------------------------------------------
    # Main loop

    # exception for efficiency and purity factor
    # only one option should be respected
    # therefore at least one needs to be set to 1.0

    list_f = [f_reg_ee, f_reg_ep, f_reg_xe, f_reg_ye, f_reg_ze, f_reg_xp, f_reg_yp, f_reg_zp]
    for i in range(len(ary_cb_reco_0mm)):
        for j in range(len(list_f)):
            if ary_mc_truth_0mm[i, 0] == 1.0:
                ary_cb_reco_0mm[i, j] = ary_mc_truth_0mm[i, j] + list_f[j] * (
                        ary_cb_reco_0mm[i, j] - ary_mc_truth_0mm[i, j])
            else:
                if mod_bg:
                    ary_cb_reco_0mm[i, j] = ary_mc_truth_0mm[i, j] + list_f[j] * (
                            ary_cb_reco_0mm[i, j] - ary_mc_truth_0mm[i, j])
                continue

    for i in range(len(ary_cb_reco_5mm)):
        for j in range(len(list_f)):
            if ary_mc_truth_5mm[i, 0] == 1.0:
                ary_cb_reco_5mm[i, j] = ary_mc_truth_5mm[i, j] + list_f[j] * (
                        ary_cb_reco_5mm[i, j] - ary_mc_truth_5mm[i, j])
            else:
                if mod_bg:
                    ary_cb_reco_5mm[i, j] = ary_mc_truth_5mm[i, j] + list_f[j] * (
                            ary_cb_reco_5mm[i, j] - ary_mc_truth_5mm[i, j])
                continue

    # -----------------------------------------------------------------------------------
    # TP/FP

    if f_tp != 1.0:
        n_tp = 0
        for i in range(len(ary_cb_reco_0mm)):
            if ary_mc_truth_0mm[i, 0] == 1.0 and ary_cb_reco_0mm[i, 0] != 0.0:
                n_tp += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 1.0 and ary_cb_reco_0mm[ary_idx_0mm[i], 0] != 0.0:
                ary_cb_reco_0mm[ary_idx_0mm[i], 0] = 0.0
                counter += 1
            if counter >= n_tp * (1 - f_tp):
                break

        n_tp = 0
        for i in range(len(ary_cb_reco_5mm)):
            if ary_mc_truth_5mm[i, 0] == 1.0 and ary_cb_reco_5mm[i, 0] != 0.0:
                n_tp += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 1.0 and ary_cb_reco_5mm[ary_idx_5mm[i], 0] != 0.0:
                ary_cb_reco_5mm[ary_idx_5mm[i], 0] = 0.0
                counter += 1
            if counter >= n_tp * (1 - f_tp):
                break

    if f_fp != 1.0:

        n_fp = 0
        for i in range(len(ary_cb_reco_0mm)):
            if ary_mc_truth_0mm[i, 0] == 0.0 and ary_cb_reco_0mm[i, 0] != 0.0:
                n_fp += 1

        counter = 0
        for i in range(len(ary_idx_0mm)):
            if ary_mc_truth_0mm[ary_idx_0mm[i], 0] == 0.0 and ary_cb_reco_0mm[ary_idx_0mm[i], 0] != 0.0:
                ary_cb_reco_0mm[ary_idx_0mm[i], 0] = 0.0
                counter += 1
            if counter >= n_fp * (1 - f_fp):
                break

        n_fp = 0
        for i in range(len(ary_cb_reco_5mm)):
            if ary_mc_truth_5mm[i, 0] == 0.0 and ary_cb_reco_5mm[i, 0] != 0.0:
                n_fp += 1

        counter = 0
        for i in range(len(ary_idx_5mm)):
            if ary_mc_truth_5mm[ary_idx_5mm[i], 0] == 0.0 and ary_cb_reco_5mm[ary_idx_5mm[i], 0] != 0.0:
                ary_cb_reco_5mm[ary_idx_5mm[i], 0] = 0.0
                counter += 1
            if counter >= n_fp * (1 - f_fp):
                break

    os.chdir(dir_main)
    # --------------------------------------------------------------
    # export modified toy dataset
    with open(dir_toy + FILE_NAME + "/" + FILE_NAME + "_" + TAG + "_toy.npz", 'wb') as file:
        np.savez_compressed(file, CB_RECO_0MM=ary_cb_reco_0mm, CB_RECO_5MM=ary_cb_reco_5mm)

    print("file saved: ", FILE_NAME + "_" + TAG + "_toy.npz")
