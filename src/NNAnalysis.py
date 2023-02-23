import numpy as np
from src import Plotter


# ----------------------------------------------------------------------------------------------------------------------
# Classification analysis methods

def efficiency_map_sourceposition(y_pred, y_true, ary_sp, theta=0.5):
    list_sp_true = []
    list_sp_pred = []

    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] > theta:
                list_sp_pred.append(ary_sp[i])
            list_sp_true.append(ary_sp[i])

    Plotter.plot_efficiency_sourceposition(list_sp_pred, list_sp_true, "efficiency_sourceposition")


# ----------------------------------------------------------------------------------------------------------------------
# methods: comparison neural network prediction vs cut-based reconstruction and monte carlo truth

def regression_nn_vs_cb(ary_nn_pred, ary_cb_reco, ary_mc_truth, ary_meta, theta=0.5):
    # grab all neural network positive events and cut-based identified events
    idx_pos = ary_nn_pred[:, 0] > theta
    idx_identified = ary_meta[:, 3] != 0
    idx_ic = ary_meta[:, 2] == 1

    Plotter.plot_reg_vs_cb_energy(ary_nn_pred[idx_pos, 1],
                                  ary_cb_reco[idx_identified, 0],
                                  ary_mc_truth[idx_ic, 0],
                                  ary_nn_pred[idx_pos, 2],
                                  ary_cb_reco[idx_identified, 1],
                                  ary_mc_truth[idx_ic, 1],
                                  ary_nn_pred[idx_pos, 1] - ary_mc_truth[idx_pos, 0],
                                  ary_cb_reco[idx_identified, 0] - ary_mc_truth[idx_identified, 0],
                                  ary_nn_pred[idx_pos, 2] - ary_mc_truth[idx_pos, 1],
                                  ary_cb_reco[idx_identified, 1] - ary_mc_truth[idx_identified, 1],
                                  "energy_nn_vs_cb")

    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 3],
                                       ary_cb_reco[idx_identified, 2],
                                       ary_mc_truth[idx_ic, 2],
                                       ary_nn_pred[idx_pos, 3] - ary_mc_truth[idx_pos, 2],
                                       ary_cb_reco[idx_identified, 2] - ary_mc_truth[idx_identified, 2],
                                       "position_nn_vs_cb",
                                       "x",
                                       "electron", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 4],
                                       ary_cb_reco[idx_identified, 3],
                                       ary_mc_truth[idx_ic, 3],
                                       ary_nn_pred[idx_pos, 4] - ary_mc_truth[idx_pos, 3],
                                       ary_cb_reco[idx_identified, 3] - ary_mc_truth[idx_identified, 3],
                                       "position_nn_vs_cb",
                                       "y",
                                       "electron", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 5],
                                       ary_cb_reco[idx_identified, 4],
                                       ary_mc_truth[idx_ic, 4],
                                       ary_nn_pred[idx_pos, 5] - ary_mc_truth[idx_pos, 4],
                                       ary_cb_reco[idx_identified, 4] - ary_mc_truth[idx_identified, 4],
                                       "position_nn_vs_cb",
                                       "z",
                                       "electron", )
