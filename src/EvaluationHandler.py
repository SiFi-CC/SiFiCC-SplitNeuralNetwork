import numpy as np
from src import NPZParser
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
    acc_base, eff_base, pur_base, conf_base = get_metrics(y_scores, y_true, threshold=0.5)
    print("\nMetrics base threshold: ")
    print("Threshold: {:.3f}".format(0.5))
    print("Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
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
    print("Baseline accuracy: {:.3f}".format(1 - (np.sum(y_true) / len(y_true))))
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

def get_primary_energy(y_scores, y_true, y_primary_energy, theta=0.5):
    ary_primaryenergy_pos = [y_primary_energy[i] for i in range(len(y_scores))
                             if (y_scores[i] > theta and y_primary_energy[i] != 0.0)]
    ary_primaryenergy_all = [y_primary_energy[i] for i in range(len(y_true)) if y_true[i] == 1]
    return ary_primaryenergy_pos, ary_primaryenergy_all


def get_source_position(y_scores, y_true, y_source_pos, theta=0.5):
    ary_sourcepos_pos = [y_source_pos[i] for i in range(len(y_scores))
                         if (y_scores[i] > theta and y_source_pos[i] != 0.0)]
    ary_sourcepos_all = [y_source_pos[i] for i in range(len(y_true)) if y_true[i] == 1]
    return ary_sourcepos_pos, ary_sourcepos_all


def eval_classifier(NeuralNetwork, data_cluster, theta=0.5):
    # get neural network prediction
    y_scores = NeuralNetwork.predict(data_cluster.x_test())
    y_true = data_cluster.y_test()
    # print(y_true)

    write_metrics_classifier(y_scores, y_true)
    Plotter.plot_score_dist(y_scores, y_true, "score_dist")
    fastROCAUC.fastROCAUC(y_scores, y_true, save_fig="ROCAUC")
    _, theta_opt = fastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)

    # evaluate primary energy spectrum
    for threshold in [theta, 0.7, theta_opt]:
        ary_primE_pos, ary_primE_all = get_primary_energy(y_scores,
                                                          y_true,
                                                          data_cluster.meta[data_cluster.idx_test(), 1],
                                                          threshold)
        Plotter.plot_primary_energy_dist(ary_primE_pos, ary_primE_all, "dist_primE_theta" + str(threshold))

    # evaluate source position spectrum

    for threshold in [theta, 0.7, theta_opt]:
        ary_sourcep_pos, ary_sourcep_all = get_primary_energy(y_scores,
                                                              y_true,
                                                              data_cluster.meta[data_cluster.idx_test(), 2],
                                                              threshold)
        Plotter.plot_source_position(ary_sourcep_pos, ary_sourcep_all, "dist_sourcep_theta" + str(threshold))

    # score distributions as 2dhistorgrams
    y_scores_pos = y_scores[y_true == 1]
    y_sourcepos = data_cluster.meta[data_cluster.idx_test(), 2]
    y_eprimary = data_cluster.meta[data_cluster.idx_test(), 1]
    y_eprimary = y_eprimary[y_true == 1]
    y_sourcepos = y_sourcepos[y_true == 1]

    Plotter.plot_2dhist_score_sourcepos(y_scores_pos, y_sourcepos, "hist2d_score_sourcepos")
    Plotter.plot_2dhist_score_eprimary(y_scores_pos, y_eprimary, "hist2d_score_eprimary")


def eval_regression_energy(NeuralNetwork, DataCluster):
    # set regression
    DataCluster.update_targets_energy()
    DataCluster.update_indexing_positives()

    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    Plotter.plot_regression_energy_error(y_pred, y_true, "error_regression_energy")


def eval_regression_position(NeuralNetwork, DataCluster):
    # set regression
    DataCluster.update_targets_position()
    DataCluster.update_indexing_positives()

    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    Plotter.plot_regression_position_error(y_pred, y_true, "error_regression_position")


def eval_full(NeuralNetwork_clas,
              NeuralNetwork_regE,
              NeuralNetwork_regP,
              npz_file,
              file_name="",
              theta=0.5):
    # load npz file into DataCluster object
    data_cluster = NPZParser.parse(npz_file)

    # standardize input
    data_cluster.standardize()

    # grab all positive identified events by the neural network
    y_scores = NeuralNetwork_clas.predict(data_cluster.features)
    idx_pos = [float(y_scores[i]) > theta for i in range(len(y_scores))]

    # predict energy and position of all positive events
    y_pred_energy = NeuralNetwork_regE.predict(data_cluster.features[idx_pos, :])
    y_pred_position = NeuralNetwork_regP.predict(data_cluster.features[idx_pos, :])

    y_pred_class = (y_scores[idx_pos] > theta) * 1
    y_true_clas = data_cluster.targets_clas[idx_pos]
    y_true_e = data_cluster.targets_reg1[idx_pos, :]
    y_true_p = data_cluster.targets_reg2[idx_pos, :]

    counter_pos = 0
    for i in range(len(y_pred_class)):
        identified = 1
        if not (y_pred_class[i] == 1 and y_true_clas[i] == 1):
            identified = 0
        if np.abs(float(y_pred_energy[i, 0]) - float(y_true_e[i, 0])) > 2 * 0.06 * float(y_true_e[i, 0]):
            identified = 0
        if np.abs(float(y_pred_energy[i, 1]) - float(y_true_e[i, 1])) > 2 * 0.06 * float(y_true_e[i, 1]):
            identified = 0
        if np.abs(y_pred_position[i, 0] - y_true_p[i, 0]) > 1.3 * 2:
            identified = 0
        if np.abs(y_pred_position[i, 1] - y_true_p[i, 1]) > 10.0 * 2:
            identified = 0
        if np.abs(y_pred_position[i, 2] - y_true_p[i, 2]) > 1.3 * 2:
            identified = 0
        if np.abs(y_pred_position[i, 3] - y_true_p[i, 3]) > 1.3 * 2:
            identified = 0
        if np.abs(y_pred_position[i, 4] - y_true_p[i, 4]) > 10.0 * 2:
            identified = 0
        if np.abs(y_pred_position[i, 5] - y_true_p[i, 5]) > 1.3 * 2:
            identified = 0

        if identified == 1:
            counter_pos += 1

    print("# Full evaluation statistics: ")
    print("Efficiency: {:.1f}".format(counter_pos / np.sum(data_cluster.targets_clas) * 100))
    print("Purity: {:.1f}".format(counter_pos / np.sum(y_true_clas) * 100))
    """
    from src import MLEMExport
    MLEMExport.export_mlem(ary_e=y_pred_energy[:, 0],
                           ary_p=y_pred_energy[:, 1],
                           ary_ex=y_pred_position[:, 0],
                           ary_ey=y_pred_position[:, 1],
                           ary_ez=y_pred_position[:, 2],
                           ary_px=y_pred_position[:, 3],
                           ary_py=y_pred_position[:, 4],
                           ary_pz=y_pred_position[:, 5],
                           filename=file_name,
                           verbose=1)
    """

def export_mlem_simpleregression(nn_classifier, npz_file, file_name=""):
    # set classification threshold
    theta = 0.5

    # load npz file into DataCluster object
    data_cluster = NPZParser.parse(npz_file)

    # standardize input
    data_cluster.standardize()

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

    from src import MLEMExport
    MLEMExport.export_mlem(ary_e,
                           ary_p,
                           ary_ex,
                           ary_ey,
                           ary_ez,
                           ary_px,
                           ary_py,
                           ary_pz,
                           filename=file_name,
                           b_comptonkinematics=False,
                           b_dacfilter=False,
                           verbose=1)


def montecarlo_regression(NeuralNetwork_regE,
                          NeuralNetwork_regP,
                          npz_file,
                          file_name):
    """
    This method takes two fully trained Neural Networks for Energy and Position regression and evaluates them on all
    positive events of the dataset. The goal is to investigate how the image reconstruction reacts to a sample with
    perfect classification and DNN regression.

    Args:
        NeuralNetwork_regE:
        NeuralNetwork_regP:
        npz_file:
        file_name:

    Return:
         None
    """
    # load npz file into DataCluster object
    # apply needed preprocessing steps:
    #   - set test set ratio to 1.0 for full evaluation of the data sample
    #   - Standardize evaluation set
    #   - update indexing to take only true positive events
    data_cluster = NPZParser.parse(npz_file)
    data_cluster.p_train = 0.0
    data_cluster.p_valid = 0.0
    data_cluster.p_test = 1.0
    data_cluster.standardize()
    data_cluster.update_targets_position()
    data_cluster.update_indexing_positives()

    # evaluation sample and Neural Network prediction
    y_pred_e = NeuralNetwork_regE.predict(data_cluster.x_test())
    y_pred_p = NeuralNetwork_regP.predict(data_cluster.x_test())

    # export Neural Network prediction to MLEM input
    from src import MLEMExport
    MLEMExport.export_mlem(ary_e=y_pred_e[:, 0],
                           ary_p=y_pred_e[:, 1],
                           ary_ex=y_pred_p[:, 0],
                           ary_ey=y_pred_p[:, 1],
                           ary_ez=y_pred_p[:, 2],
                           ary_px=y_pred_p[:, 3],
                           ary_py=y_pred_p[:, 4],
                           ary_pz=y_pred_p[:, 5],
                           filename=file_name,
                           verbose=1)
