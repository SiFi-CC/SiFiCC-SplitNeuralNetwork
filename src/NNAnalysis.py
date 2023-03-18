import numpy as np
from src import Plotter
from src import fastROCAUC


# ----------------------------------------------------------------------------------------------------------------------
# Classification analysis methods


def write_metrics_classifier(y_scores, y_true):
    acc_base, eff_base, pur_base, conf_base = get_classifier_metrics(y_scores, y_true, threshold=0.5)
    acc_weight, _, _, _ = get_classifier_metrics(y_scores, y_true, threshold=0.5, weighted=True)
    print("\nMetrics base threshold: ")
    print("Threshold: {:.3f}".format(0.5))
    print("Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
    print("Accuracy: {:.1f}%".format(acc_base * 100))
    print("Accuracy (weighted): {:.1f}%".format(acc_weight * 100))
    print("Efficiency: {:.1f}%".format(eff_base * 100))
    print("Purity: {:.1f}%".format(pur_base * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_base))

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)
    acc_opt, eff_opt, pur_opt, conf_opt = get_classifier_metrics(y_scores, y_true, threshold=theta)
    acc_opt_weight, _, _, _ = get_classifier_metrics(y_scores, y_true, threshold=theta, weighted=True)
    print("\nMetrics base threshold: ")
    print("AUC Score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
    print("Accuracy: {:.1f}%".format(acc_opt * 100))
    print("Accuracy (weighted): {:.1f}%".format(acc_opt_weight * 100))
    print("Efficiency: {:.1f}%".format(eff_opt * 100))
    print("Purity: {:.1f}%".format(pur_opt * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_opt))

    with open("metrics.txt", 'w') as f:
        f.write("### AnalysisMetric results:\n")
        f.write("\nBaseline accuracy: {:.3f}\n".format(1 - (np.sum(y_true) / len(y_true))))

        f.write("Metrics base threshold:\n")
        f.write("Threshold: {:.3f}\n".format(0.5))
        f.write("Accuracy: {:.1f}%\n".format(acc_base * 100))
        f.write("Accuracy (weighted)%: {:.1f}\n".format(acc_weight * 100))
        f.write("Efficiency: {:.1f}%\n".format(eff_base * 100))
        f.write("Purity: {:.1f}%\n".format(pur_base * 100))
        f.write("TP: {} | FP: {} | TN: {} | FN: {}\n".format(*conf_base))

        f.write("Metrics base threshold\n")
        f.write("AUC Score: {:.3f}\n".format(auc))
        f.write("Threshold: {:.3f}\n".format(theta))
        f.write("Accuracy: {:.1f}%\n".format(acc_opt * 100))
        f.write("Accuracy (weighted): {:.1f}%\n".format(acc_opt_weight * 100))
        f.write("Efficiency: {:.1f}%\n".format(eff_opt * 100))
        f.write("Purity: {:.1f}%\n".format(pur_opt * 100))
        f.write("TP: {} | FP: {} | TN: {} | FN: {}\n".format(*conf_opt))
        f.close()


def efficiency_map_sourceposition(y_pred, y_true, ary_sp, theta=0.5):
    list_sp_true = []
    list_sp_pred = []

    for i in range(len(y_pred)):
        if y_true[i] == 1:
            if y_pred[i] > theta:
                list_sp_pred.append(ary_sp[i])
            list_sp_true.append(ary_sp[i])

    Plotter.plot_efficiency_sourceposition(list_sp_pred, list_sp_true, "efficiency_sourceposition")


def get_classifier_metrics(y_scores, y_true, threshold, weighted=False):
    # pre-define
    y_pred = np.zeros(shape=(len(y_true, )))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_scores[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1

    if (tp + fn) == 0:
        efficiency = 0
    else:
        efficiency = tp / (tp + fn)
    if (tp + fp) == 0:
        purity = 0
    else:
        purity = tp / (tp + fp)

    if weighted:
        # set sample weights to class weights
        _, counts = np.unique(y_pred, return_counts=True)
        class_weights = [len(y_pred) / (2 * counts[0]), len(y_pred) / (2 * counts[1])]

        accuracy = ((tp * class_weights[1]) + (tn * class_weights[0])) / (
                ((tp + fp) * class_weights[1]) + ((tn + fn) * class_weights[0]))
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, efficiency, purity, (tp, fp, tn, fn)


# ----------------------------------------------------------------------------------------------------------------------
# Comparison of score values against features

def dist_primaryenergy(y_scores, y_true, y_primary_energy, theta, figure_name):
    """
    Grab primary energy arrays of all positive, signal and total events
    """
    # neural network positive events
    ary_pe_pos = [y_primary_energy[i] for i in range(len(y_scores))
                  if (y_scores[i] > theta and y_primary_energy[i] != 0.0)]
    # all signal events
    ary_pe_tp = [y_primary_energy[i] for i in range(len(y_true)) if y_true[i] == 1]
    # total events with condition that primary energy cannot be zero
    ary_pe_tot = [y_primary_energy[i] for i in range(len(y_true)) if y_primary_energy[i] != 0.0]
    Plotter.plot_primary_energy_dist(ary_pe_pos,
                                     ary_pe_tp,
                                     ary_pe_tot,
                                     figure_name)


def dist_sourceposition(y_scores, y_true, y_source_pos, theta, figure_name):
    """
    Grab source positions arrays of all positive, signal and total events
    """
    # neural network positive events
    ary_sp_pos = [y_source_pos[i] for i in range(len(y_scores))
                  if (y_scores[i] > theta and y_source_pos[i] != 0.0)]
    # all signal events
    ary_sp_tp = [y_source_pos[i] for i in range(len(y_true)) if y_true[i] == 1]
    # total events with condition that source position cannot be zero
    ary_sp_tot = [y_source_pos[i] for i in range(len(y_true)) if y_source_pos[i] != 0.0]
    Plotter.plot_source_position(ary_sp_pos,
                                 ary_sp_tp,
                                 ary_sp_tot,
                                 figure_name)


# ----------------------------------------------------------------------------------------------------------------------
# methods: comparison neural network prediction vs cut-based reconstruction and monte carlo truth

def regression_nn_vs_cb(ary_nn_pred, ary_cb_reco, ary_mc_truth, ary_meta, theta=0.5):
    # grab all neural network positive events and cut-based identified events
    idx_pos = ary_nn_pred[:, 0] > theta
    idx_identified = ary_meta[:, 3] != 0
    idx_ic = ary_meta[:, 2] == 1

    Plotter.plot_reg_vs_cb_energy(ary_nn_pred[idx_pos, 1],
                                  ary_cb_reco[idx_identified, 1],
                                  ary_mc_truth[idx_ic, 1],
                                  ary_nn_pred[idx_pos, 2],
                                  ary_cb_reco[idx_identified, 2],
                                  ary_mc_truth[idx_ic, 2],
                                  ary_nn_pred[idx_pos, 1] - ary_mc_truth[idx_pos, 1],
                                  ary_cb_reco[idx_identified, 1] - ary_mc_truth[idx_identified, 1],
                                  ary_nn_pred[idx_pos, 2] - ary_mc_truth[idx_pos, 2],
                                  ary_cb_reco[idx_identified, 2] - ary_mc_truth[idx_identified, 2],
                                  "energy_nn_vs_cb")

    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 3],
                                       ary_cb_reco[idx_identified, 3],
                                       ary_mc_truth[idx_ic, 3],
                                       ary_nn_pred[idx_pos, 3] - ary_mc_truth[idx_pos, 3],
                                       ary_cb_reco[idx_identified, 3] - ary_mc_truth[idx_identified, 3],
                                       "position_nn_vs_cb",
                                       "x",
                                       "electron", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 4],
                                       ary_cb_reco[idx_identified, 4],
                                       ary_mc_truth[idx_ic, 4],
                                       ary_nn_pred[idx_pos, 4] - ary_mc_truth[idx_pos, 4],
                                       ary_cb_reco[idx_identified, 4] - ary_mc_truth[idx_identified, 4],
                                       "position_nn_vs_cb",
                                       "y",
                                       "electron", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 5],
                                       ary_cb_reco[idx_identified, 5],
                                       ary_mc_truth[idx_ic, 5],
                                       ary_nn_pred[idx_pos, 5] - ary_mc_truth[idx_pos, 5],
                                       ary_cb_reco[idx_identified, 5] - ary_mc_truth[idx_identified, 5],
                                       "position_nn_vs_cb",
                                       "z",
                                       "electron", )

    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 6],
                                       ary_cb_reco[idx_identified, 6],
                                       ary_mc_truth[idx_ic, 6],
                                       ary_nn_pred[idx_pos, 6] - ary_mc_truth[idx_pos, 6],
                                       ary_cb_reco[idx_identified, 6] - ary_mc_truth[idx_identified, 6],
                                       "position_nn_vs_cb",
                                       "x",
                                       "photon", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 7],
                                       ary_cb_reco[idx_identified, 7],
                                       ary_mc_truth[idx_ic, 7],
                                       ary_nn_pred[idx_pos, 7] - ary_mc_truth[idx_pos, 7],
                                       ary_cb_reco[idx_identified, 7] - ary_mc_truth[idx_identified, 7],
                                       "position_nn_vs_cb",
                                       "y",
                                       "photon", )
    Plotter.plot_reg_nn_vs_cb_position(ary_nn_pred[idx_pos, 8],
                                       ary_cb_reco[idx_identified, 8],
                                       ary_mc_truth[idx_ic, 8],
                                       ary_nn_pred[idx_pos, 8] - ary_mc_truth[idx_pos, 8],
                                       ary_cb_reco[idx_identified, 8] - ary_mc_truth[idx_identified, 8],
                                       "position_nn_vs_cb",
                                       "z",
                                       "photon", )
