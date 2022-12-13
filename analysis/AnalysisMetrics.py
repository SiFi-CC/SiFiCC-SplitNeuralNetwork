def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    from fastROCAUC import fastROCAUC

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

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

    ### TEMPRORARY
    # grab non identified ideal compton event event-numbers
    list_FN = []
    ary_eventnumber = MetaData.event_number()[DataCluster.idx_test()]
    for i in range(len(ary_eventnumber)):
        if y_pred[i] == 0 and y_true[i] == 1:
            list_FN.append(ary_eventnumber[i])

    print("\nClassification results: ")
    print("AUC score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
    print("Efficiency: {:.1f}%".format(efficiency * 100))
    print("Purity: {:.1f}%".format(purity * 100))
    print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))

    # save results in txt file
    dir_main = os.getcwd()
    dir_results = dir_main + "/results/" + SiFiCCNN.model_name + SiFiCCNN.model_tag
    if not os.path.isdir(dir_results):
        os.mkdir(dir_results)

    with open(dir_results + "/metrics.txt", 'w') as f:
        f.write("### AnalysisMetric results:\n")
        f.write("AUC score: {:.3f}\n".format(auc))
        f.write("Threshold: {:.3f}\n".format(theta))
        f.write("Accuracy: {:.1f}\n".format((TP + TN) / (TP + TN + FP + FN) * 100))
        f.write("Efficiency: {:.1f}%\n".format(efficiency * 100))
        f.write("Purity: {:.1f}%\n".format(purity * 100))
        f.write("TP: {} | TN: {} | FP: {} | FN: {}\n".format(TP, TN, FP, FN))
        f.write("\n")
        for i in range(20):
            f.write(str(list_FN[i]) + "\n")
