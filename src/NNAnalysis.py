import numpy as np
from src import Plotter


# ----------------------------------------------------------------------------------------------------------------------
# methods: comparison neural network prediction vs cut-based reconstruction and monte carlo truth

def regression_nn_vs_cb(ary_nn_pred, ary_cb_reco, ary_mc_truth, ary_meta, theta=0.5):
    # energy comparison
    ary_e_nn_err = np.zeros(shape=(ary_nn_pred.shape[0], 2))
    ary_e_cb_err = np.zeros(shape=(ary_cb_reco.shape[0], 2))
    ary_e_nn_err[:, 0] = ary_nn_pred[:, 1] - ary_mc_truth[:, 0]
    ary_e_nn_err[:, 1] = ary_nn_pred[:, 2] - ary_mc_truth[:, 1]
    ary_e_cb_err[:, 0] = ary_cb_reco[:, 0] - ary_mc_truth[:, 0]
    ary_e_cb_err[:, 1] = ary_cb_reco[:, 1] - ary_mc_truth[:, 1]

    # grab all neural network positive events and cut-based identified events
    idx_pos = ary_nn_pred[:, 0] > theta
    idx_idenified = ary_meta[:, 3] != 0

    Plotter.plot_reg_vs_cb_energy(ary_nn_pred[:, 1:3],
                                  ary_cb_reco[:, :2],
                                  ary_mc_truth[:, :2],
                                  ary_e_nn_err[idx_pos, :],
                                  ary_e_cb_err[idx_idenified, :],
                                  "energy_nn_vs_cb")
