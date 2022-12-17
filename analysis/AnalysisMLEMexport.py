def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import numpy as np
    import os
    from fastROCAUC import fastROCAUC

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.features)
    y_true = DataCluster.targets

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC(y_pred, y_true, return_score=True)

    # best optimal threshold
    threshold = theta

    # evaluate important metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_pred[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            FN += 1

    if (TP + FN) == 0:
        efficiency = 0
    else:
        efficiency = TP / (TP + FN)
    if (TP + FP) == 0:
        purity = 0
    else:
        purity = TP / (TP + FP)

    print("\nFull dataset Classification results: ")
    print("AUC score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
    print("Efficiency: {:.1f}%".format(efficiency * 100))
    print("Purity: {:.1f}%".format(purity * 100))
    print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))

    ####################################################################################################################
    # Cut-Based approach based energy and position regression

    # grab sum of absorber energy
    p = DataCluster.meta[:, 2]
    # e = DataCluster.features
