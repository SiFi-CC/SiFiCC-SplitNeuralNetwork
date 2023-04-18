import numpy as np
import matplotlib.pyplot as plt

def fastROCAUC(y_pred, y_true, weighted=False, return_score=False, save_fig=None):
    """

    :param y_pred: Predicted probabilities of shape [n,1]
    :param y_true: True probabilities of shape [n,1]
    :param weighted: True: use y_pred as weights for AUC Score
    :param return_score: True: returns results: AUC Score
    :param save_fig: str: saves ROC courve plot
    :return:
    """

    # defining variables needed for generating ROC curve
    score = []
    p = []
    weight = []

    # sort input array by possibilities
    full_array = np.zeros((len(y_pred), 2))
    full_array[:, 0] = y_pred[:, 0]
    full_array[:, 1] = y_true
    full_array = full_array[full_array[:, 0].argsort()]

    # bring data into necessary format
    if weighted:
        auc_label = "weightedAUC Score"

        for i in range(len(y_pred)):
            # model predicts X_i in A with probability p_i, contributes with weight wa
            score.append(1)
            p.append(full_array[i][0])
            weight.append(full_array[i][2])

            # model incorrectly predicts X_i in A with probability p_i, contributes with weight wb
            score.append(0)
            p.append(full_array[i][0])
            weight.append(full_array[i][3])

    else:
        auc_label = "AUC Score"

        for i in range(len(y_pred)):
            # simple classifier
            if full_array[i][1] >= 0.5:
                score.append(1)
            else:
                score.append(0)
            p.append(full_array[i][0])
            weight.append(1)

    fpr_list = []  # false-positive-rate
    tpr_list = []  # true-positive-rate
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    # calculate initial fp and fp with the lowest threshold possible (< min(p))
    # events can now only be classified as tp or fp
    for i in range(len(p)):
        if score[i] == 1:
            tp += weight[i]
        elif score[i] == 0:
            fp += weight[i]

    # tpr and fpr are in this case 1.0
    fpr_list.append(1.0)
    tpr_list.append(1.0)

    # iterate through given possibilities using each as ROC threshold
    # if initial result is 1, event changes from tp to fn
    # if initial result is 0, event changes from fp to tn

    # get the most optimal threshold
    acc = (tp + tn) / (tp + tn + fp + fn)
    theta = p[0]
    dot = (0, 0)

    for i in range(len(p)):
        if score[i] == 1:
            tp -= weight[i]
            fn += weight[i]
        elif score[i] == 0:
            fp -= weight[i]
            tn += weight[i]

        # catch exception to tpr,fpr = 0
        # and calculate tpr,fpr
        try:
            fpr = fp / (fp + tn)
        except ZeroDivisionError:
            fpr = 0
        try:
            tpr = tp / (tp + fn)
        except ZeroDivisionError:
            tpr = 0

        if (tp + tn) / (tp + tn + fp + fn) >= acc:
            # update accuracy and threshold
            acc = (tp + tn) / (tp + tn + fp + fn)
            theta = p[i]
            dot = (fpr, tpr)

        fpr_list.append(fpr)
        tpr_list.append(tpr)

    # calc area under ROC
    auc_score = 0
    for i in range(len(fpr_list) - 1):
        # Riemann-sum to calculate area under curve
        area = (fpr_list[i + 1] - fpr_list[i]) * tpr_list[i]
        # multiply result by -1, since x values are ordered from highest to lowest
        auc_score += area * (-1)

    """ 
    # results
    print("### AUC Results: ###")
    print("AUC Score: ", auc_score)
    print("Best threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format(acc * 100)
    """

    if return_score:
        return auc_score, theta
    if save_fig is not None:
        print("Plotting ROC curve and {}...".format(auc_label))
        plt.figure()
        plt.title("ROC Curve | " + auc_label)
        plt.plot(fpr_list, tpr_list, color="red", label="{0:}: {1:.3f}".format(auc_label, auc_score))
        plt.plot([0, 1], [0, 1], color="black", ls="--")
        plt.plot(dot[0], dot[1], 'b+')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.savefig(save_fig)
