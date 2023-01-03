import os
import numpy as np
import matplotlib.pyplot as plt

from src import fastROCAUC
from src import Plotter


def get_metrics(y_scores, y_true, threshold, weighted=False):
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


def write_metrics_classifier(y_scores, y_true):
    print("Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))

    acc_base, eff_base, pur_base, conf_base = get_metrics(y_scores, y_true, threshold=0.5)
    print("\nMetrics base threshold: ")
    print("Threshold: {:.3f}".format(0.5))
    print("Accuracy: {:.1f}".format(acc_base * 100))
    print("Efficiency: {:.1f}%".format(eff_base * 100))
    print("Purity: {:.1f}%".format(pur_base * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_base))

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)
    acc_opt, eff_opt, pur_opt, conf_opt = get_metrics(y_scores, y_true, threshold=theta)
    print("\nMetrics base threshold: ")
    print("AUC Score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format(acc_opt * 100))
    print("Efficiency: {:.1f}%".format(eff_opt * 100))
    print("Purity: {:.1f}%".format(pur_opt * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_opt))

    with open("metrics.txt", 'w') as f:
        f.write("### AnalysisMetric results:\n")
        f.write("\nBaseline accuracy: {:.3f}\n".format(1 - (np.sum(y_true) / len(y_true))))

        f.write("Metrics base threshold:\n")
        f.write("Threshold: {:.3f}\n".format(0.5))
        f.write("Accuracy: {:.1f}\n".format(acc_base * 100))
        f.write("Efficiency: {:.1f}%\n".format(eff_base * 100))
        f.write("Purity: {:.1f}%\n".format(pur_base * 100))
        f.write("TP: {} | FP: {} | TN: {} | FN: {}\n".format(*conf_base))

        f.write("Metrics base threshold\n")
        f.write("AUC Score: {:.3f}\n".format(auc))
        f.write("Threshold: {:.3f}\n".format(theta))
        f.write("Accuracy: {:.1f}\n".format(acc_opt * 100))
        f.write("Efficiency: {:.1f}%\n".format(eff_opt * 100))
        f.write("Purity: {:.1f}%\n".format(pur_opt * 100))
        f.write("TP: {} | FP: {} | TN: {} | FN: {}\n".format(*conf_opt))
        f.close()


########################################################################################################################

def classifier_evaluation(data_cluster, meta_data, classifier):
    y_scores = classifier.predict(data_cluster.x_test())
    y_true = data_cluster.y_test()

    Plotter.plot_history_classifier(classifier, "training_history_classifier")
    write_metrics_classifier(y_scores, y_true)
    Plotter.plot_score_dist(y_scores, y_true, "training_score_dist")
    Plotter.plot_source_position(y_scores,
                                 meta_data.mc_source_position_z()[data_cluster.idx_test()],
                                 meta_data.mc_source_position_z()[meta_data.meta[:, 2] == 1],
                                 "training_source_position")

    counter = 0
    for i in range(len(y_scores)):
        if y_scores[i] < 0.85:
            continue
        if y_true[i] == 1:
            continue

        idx = data_cluster.idx_test()[i]
        print("score: {} | class: {}".format(y_scores[i], y_true[i]))
        print("index: {} | event number: {}".format(idx, meta_data.event_number()[idx]))
        print("")
        counter += 1

        if counter > 10:
            break


def regression_evaluation(data_cluster, regression1, regression2):
    y_pred_energy = regression1.predict(data_cluster.x_test_reg())
    y_true_energy = data_cluster.y_test_reg1()

    y_pred_position = regression2.predict(data_cluster.x_test_reg())
    y_true_position = data_cluster.y_test_reg2()

    Plotter.plot_regression_energy_error(y_pred_energy, y_true_energy, "training_error_energy")
    Plotter.plot_regression_position_error(y_pred_position, y_true_position, "training_error_position")


def export_mlem_cutbased(nn_classifier, data_cluster):
    # set classification threshold
    theta = 0.5

    # get classification results
    y_scores_classifier = nn_classifier.predict(data_cluster.features)

    # pre-define
    y_pred_classifier = np.zeros(shape=(len(y_scores_classifier, )))

    for i in range(len(y_pred_classifier)):
        # apply prediction threshold
        if y_scores_classifier[i] >= theta:
            y_pred_classifier[i] = 1
        else:
            y_pred_classifier[i] = 0

    list_idx_positives = y_pred_classifier == 1

    # denormalize features
    for i in range(data_cluster.features.shape[1]):
        data_cluster.features[:, i] *= data_cluster.list_std[i]
        data_cluster.features[:, i] += data_cluster.list_mean[i]

    # grab event kinematics from feature list
    ary_e = data_cluster.features[list_idx_positives, 1]
    ary_ex = data_cluster.features[list_idx_positives, 2]
    ary_ey = data_cluster.features[list_idx_positives, 3]
    ary_ez = data_cluster.features[list_idx_positives, 4]

    # select only absorber energies
    # select only positive events
    # replace -1. (NaN) values with 0.
    ary_p = data_cluster.features[:, [10, 19, 28, 37, 46]]
    ary_p = ary_p[list_idx_positives, :]
    for i in range(ary_p.shape[0]):
        for j in range(ary_p.shape[1]):
            if ary_p[i, j] == -1.:
                ary_p[i, j] = 0.0
    ary_p = np.sum(ary_p, axis=1)

    ary_px = data_cluster.features[list_idx_positives, 11]
    ary_py = data_cluster.features[list_idx_positives, 12]
    ary_pz = data_cluster.features[list_idx_positives, 13]

    from src import MLEMExportCutBased
    MLEMExportCutBased.export_mlem(ary_e, ary_p, ary_ex, ary_ey, ary_ez, ary_px, ary_py, ary_pz,
                                   "OptimizedGeometry_BP0mm_2e10protons_DNN_S1AX_Mixed")


def export_mlem(nn_classifier, nn_regression1, nn_regression2, data_cluster):
    # update test-sample ratio
    data_cluster.p_train = 0.0
    data_cluster.p_valid = 0.0
    data_cluster.p_test = 1.0

    # set classification threshold
    theta = 0.5

    # get classification results
    y_scores_classifier = nn_classifier.predict(data_cluster.features)

    # pre-define
    y_pred_classifier = np.zeros(shape=(len(y_scores_classifier, )))

    for i in range(len(y_pred_classifier)):
        # apply prediction threshold
        if y_scores_classifier[i] >= theta:
            y_pred_classifier[i] = 1
        else:
            y_pred_classifier[i] = 0

    list_idx_positives = y_pred_classifier == 1
    print("accuracy: {:.1f}".format(np.sum(y_pred_classifier) / len(y_pred_classifier) * 100))
    print("number of positive events: ", np.sum(y_pred_classifier))
    print("input into regression: ", len(data_cluster.features[list_idx_positives, :]))

    # get regression predictions
    y_pred_energies = nn_regression1.predict(data_cluster.features[list_idx_positives, :])
    y_pred_positions = nn_regression2.predict(data_cluster.features[list_idx_positives, :])
    y_true_energy = data_cluster.targets_reg1[list_idx_positives, :]
    y_true_positions = data_cluster.targets_reg2[list_idx_positives, :]

    """
    from src import MLEMExportRegression
    MLEMExport.export_mlem(y_pred_energies[:, 0],
                           y_pred_energies[:, 1],
                           y_pred_positions[:, 0],
                           y_pred_positions[:, 1],
                           y_pred_positions[:, 2],
                           y_pred_positions[:, 3],
                           y_pred_positions[:, 4],
                           y_pred_positions[:, 5],
                           "OptimizedGeometry_BP0mm_2e10protons_DNN_Base")
    """
