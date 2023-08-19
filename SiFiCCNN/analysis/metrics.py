import numpy as np
from SiFiCCNN.analysis.fastROCAUC import fastROCAUC


def get_classifier_metrics(y_scores, y_true, theta=0.5, weighted=False):
    """
    Takes prediction and true labels and calculates True-Positive,
    False-Positive, True-Negative, False-Negative, accuracy, purity and
    efficiency for given sample.

    Args:
        y_scores (List or array): List of prediction scores
        y_true (List or array): List of true labels
        theta (Float): Decision threshold for classification
        weighted (Boolean): If true, all metrics are weighted in calculation by
                            class-weights

    Returns:

    """
    # pre-define
    y_pred = np.zeros(shape=(len(y_true, )))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_scores[i] >= theta:
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
        _, counts = np.unique(y_true, return_counts=True)
        class_weights = [len(y_true) / (2 * counts[0]),
                         len(y_true) / (2 * counts[1])]

        accuracy = ((tp * class_weights[1]) + (tn * class_weights[0])) / (
                ((tp + fp) * class_weights[1]) + ((tn + fn) * class_weights[0]))
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, efficiency, purity, (tp, fp, tn, fn)


def reco_match(y_reco, y_true):
    # energy e
    if abs(y_reco[0] - y_true[0]) > y_true[0] * 0.06 * 2:
        return False
    # energy p
    if abs(y_reco[1] - y_true[1]) > y_true[1] * 0.06 * 2:
        return False
    # position e
    if abs(y_reco[2] - y_true[2]) > 1.3 * 2:
        return False
    if abs(y_reco[3] - y_true[3]) > 10 * 2:
        return False
    if abs(y_reco[4] - y_true[4]) > 1.3 * 2:
        return False
    # position p
    if abs(y_reco[5] - y_true[5]) > 1.3 * 2:
        return False
    if abs(y_reco[6] - y_true[6]) > 10 * 2:
        return False
    if abs(y_reco[7] - y_reco[7]) > 1.3 * 2:
        return False
    return True


def global_metrics(y_reco, y_true, theta=0.5):
    # pre-define
    y_pred = np.zeros(shape=(len(y_true, )))

    n_distcompton = np.sum(y_true)
    n_tag_match = 0
    n_reco_match = 0

    for i in range(len(y_pred)):
        if y_pred[i, 0] > theta and y_true[i, 0] == 1:
            n_tag_match += 1
            if reco_match(y_reco[i, 1:], y_true[i, 1:]):
                n_reco_match += 1

    # the percentage of positively classified
    eff_match = 0

    return accuracy, efficiency, purity


def write_metrics_classifier(y_scores, y_true):
    acc_base, eff_base, pur_base, conf_base = get_classifier_metrics(y_scores,
                                                                     y_true,
                                                                     theta=0.5)
    acc_weight, _, _, _ = get_classifier_metrics(y_scores, y_true, theta=0.5,
                                                 weighted=True)
    print("\nMetrics base threshold: ")
    print("Threshold: {:.3f}".format(0.5))
    print(
        "Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
    print("Accuracy: {:.1f}%".format(acc_base * 100))
    print("Accuracy (weighted): {:.1f}%".format(acc_weight * 100))
    print("Efficiency: {:.1f}%".format(eff_base * 100))
    print("Purity: {:.1f}%".format(pur_base * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_base))

    # run ROC curve and AUC score analysis
    auc, theta, (list_fpr, list_tpr) = fastROCAUC(y_scores, y_true,
                                                  return_score=True)
    acc_opt, eff_opt, pur_opt, conf_opt = get_classifier_metrics(y_scores,
                                                                 y_true,
                                                                 theta=theta)
    acc_opt_weight, _, _, _ = get_classifier_metrics(y_scores, y_true,
                                                     theta=theta, weighted=True)
    print("\nMetrics base threshold: ")
    print("AUC Score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print(
        "Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
    print("Accuracy: {:.1f}%".format(acc_opt * 100))
    print("Accuracy (weighted): {:.1f}%".format(acc_opt_weight * 100))
    print("Efficiency: {:.1f}%".format(eff_opt * 100))
    print("Purity: {:.1f}%".format(pur_opt * 100))
    print("TP: {} | FP: {} | TN: {} | FN: {}".format(*conf_opt))

    with open("metrics.txt", 'w') as f:
        f.write("### AnalysisMetric results:\n")
        f.write("\nBaseline accuracy: {:.3f}\n".format(
            1 - (np.sum(y_true) / len(y_true))))

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
