import numpy as np
from src import NPZParser
from src import fastROCAUC
from src import Plotter
from src import SaliencyMap
from src import NNAnalysis
from src import MLEMBackprojection
from src import MLEMExport


def write_metrics_classifier(y_scores, y_true):
    acc_base, eff_base, pur_base, conf_base = NNAnalysis.get_classifier_metrics(y_scores, y_true, threshold=0.5)
    acc_weight, _, _, _ = NNAnalysis.get_classifier_metrics(y_scores, y_true, threshold=0.5, weighted=True)
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
    acc_opt, eff_opt, pur_opt, conf_opt = NNAnalysis.get_classifier_metrics(y_scores, y_true, threshold=theta)
    acc_opt_weight, _, _, _ = NNAnalysis.get_classifier_metrics(y_scores, y_true, threshold=theta, weighted=True)
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


########################################################################################################################

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


########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
# Evaluation of Neural Networks for training set


def training_clas(NeuralNetwork, DataCluster, theta=0.5):
    # Generate Neural Network prediction for test sample on training set
    y_scores = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # Plot training history
    Plotter.plot_history_classifier(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")
    # Generate efficiency map
    NNAnalysis.efficiency_map_sourceposition(y_scores, y_true, DataCluster.meta[DataCluster.idx_test(), 2], theta=theta)
    # classic evaluation of classification
    evaluate_classifier(NeuralNetwork, DataCluster, theta=theta)


def training_regE(NeuralNetwork, DataCluster):
    # Plot training history
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")
    # classic evaluation of energy regression
    evaluate_regression_energy(NeuralNetwork, DataCluster)


def training_regP(NeuralNetwork, DataCluster):
    # Plot training history
    Plotter.plot_history_regression(NeuralNetwork,
                                    NeuralNetwork.model_name + "_" + NeuralNetwork.model_tag + "_history_training")
    evaluate_regression_position(NeuralNetwork, DataCluster)


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation of Neural Network for evaluation sets

def evaluate_classifier(NeuralNetwork, data_cluster, theta=0.5):
    """
    Standard evaluation script for neural network classifier.
    """

    # grab neural network predictions for test dataset
    y_scores = NeuralNetwork.predict(data_cluster.x_test())
    y_true = data_cluster.y_test()

    # write general binary classifier metrics into console and .txt file
    write_metrics_classifier(y_scores, y_true)
    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    Plotter.plot_score_dist(y_scores, y_true, "score_dist")
    fastROCAUC.fastROCAUC(y_scores, y_true, save_fig="ROCAUC")
    _, theta_opt = fastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)

    # evaluate source position spectrum for baseline and optimal threshold
    dist_sourceposition(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 2], 0.3,
                        "dist_sourceposition_theta03")
    dist_sourceposition(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 2], 0.5,
                        "dist_sourceposition_theta05")
    dist_sourceposition(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 2], theta_opt,
                        "dist_sourceposition_thetaOPT")

    # evaluate primary energy spectrum
    dist_primaryenergy(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 1], 0.3,
                       "dist_primaryenergy_theta03")
    dist_primaryenergy(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 1], 0.5,
                       "dist_primaryenergy_theta05")
    dist_primaryenergy(y_scores, y_true, data_cluster.meta[data_cluster.idx_test(), 1], theta_opt,
                       "dist_primaryenergy_thetaOPT")

    # score distributions as 2d-historgrams
    y_scores_pos = y_scores[y_true == 1]
    y_sourcepos = data_cluster.meta[data_cluster.idx_test(), 2]
    y_eprimary = data_cluster.meta[data_cluster.idx_test(), 1]
    y_eprimary = y_eprimary[y_true == 1]
    y_sourcepos = y_sourcepos[y_true == 1]
    Plotter.plot_2dhist_score_sourcepos(y_scores_pos, y_sourcepos, "hist2d_score_sourcepos")
    Plotter.plot_2dhist_score_eprimary(y_scores_pos, y_eprimary, "hist2d_score_eprimary")

    # saliency maps for the first 10 entries of the data sample
    """
    # TODO: prob needs an update
    for i in range(10):
        x_feat = np.array([data_cluster.features[i], ])
        score_true = data_cluster.targets_clas[i]
        score_pred = float(NeuralNetwork.predict(x_feat))
        print("True class: {:.1f} | Predicted class: {:.2f}".format(score_true, score_pred))

        smap = SaliencyMap.get_smap(NeuralNetwork.model, x_feat)
        smap = np.reshape(smap, (8, 9))
        x_feat = np.reshape(x_feat, (8, 9))
        str_title = "Event ID: {}\nTrue class: {:.1f}\nPred class: {:.2f}".format(data_cluster.meta[i, 0], score_true,
                                                                                  score_pred)
        SaliencyMap.smap_plot(smap, x_feat, str_title, "SMAP_sample_" + str(i))
    """


def evaluate_regression_energy(NeuralNetwork, DataCluster):
    # set regression
    DataCluster.update_targets_energy()
    DataCluster.update_indexing_positives()

    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # energy regression
    Plotter.plot_energy_error(y_pred, y_true, "error_regression_energy")

    # predicting theta
    y_true_theta = DataCluster.theta[DataCluster.idx_test()]
    y_pred_theta = [MLEMBackprojection.calculate_theta(y_pred[i, 0], y_pred[i, 1]) for i in range(len(y_pred))]
    y_pred_theta = np.array(y_pred_theta)
    Plotter.plot_theta_error(y_pred_theta, y_true_theta, "error_regression_energy_theta")


def evaluate_regression_position(NeuralNetwork, DataCluster):
    # set regression
    DataCluster.update_targets_position()
    DataCluster.update_indexing_positives()

    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    Plotter.plot_position_error(y_pred, y_true, "error_regression_position")


"""
def eval_regression_theta(NeuralNetwork, DataCluster):
    # set regression
    DataCluster.update_targets_theta()
    DataCluster.update_indexing_positives()

    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    Plotter.plot_theta_error(y_pred, y_true, "error_regression_theta")
"""


# ----------------------------------------------------------------------------------------------------------------------
# Full Evaluation of Neural Network for evaluation sets


def eval_full(NeuralNetwork_clas,
              NeuralNetwork_regE,
              NeuralNetwork_regP,
              DataCluster,
              lookup_file,
              file_name="",
              mlem_export=None,
              theta=0.5):
    # load lookup file for monte carlo truth and cut-based reconstruction
    npz_lookup = np.load(lookup_file)
    ary_mc_truth = npz_lookup["MC_TRUTH"]
    ary_cb_reco = npz_lookup["CB_RECO"]
    ary_meta = npz_lookup["META"]

    # grab all positive identified events by the neural network
    y_scores = NeuralNetwork_clas.predict(DataCluster.features)
    y_pred_energy = NeuralNetwork_regE.predict(DataCluster.features)
    y_pred_position = NeuralNetwork_regP.predict(DataCluster.features)

    # This is done this way cause y_scores gets a really dumb shape from tensorflow
    idx_clas_pos = [float(y_scores[i]) > theta for i in range(len(y_scores))]

    # create an array containing full neural network prediction
    ary_nn_pred = np.zeros(shape=(DataCluster.entries, 9))
    ary_nn_pred[:, 0] = np.reshape(y_scores, newshape=(len(y_scores),))
    ary_nn_pred[:, 1:3] = np.reshape(y_pred_energy, newshape=(y_pred_energy.shape[0], y_pred_energy.shape[1]))
    ary_nn_pred[:, 3:] = np.reshape(y_pred_position, newshape=(y_pred_position.shape[0], y_pred_position.shape[1]))

    # plot regression neural network vs cut-based approach
    NNAnalysis.regression_nn_vs_cb(ary_nn_pred, ary_cb_reco, ary_mc_truth, ary_meta)

    # export prediction to a usable npz file
    with open(file_name + ".npz", 'wb') as f_output:
        np.savez_compressed(f_output, NN_PRED=ary_nn_pred)

    if mlem_export is not None:
        if mlem_export == "PRED":
            MLEMExport.export_mlem(ary_e=y_pred_energy[idx_clas_pos, 0],
                                   ary_p=y_pred_energy[idx_clas_pos, 1],
                                   ary_ex=y_pred_position[idx_clas_pos, 0],
                                   ary_ey=y_pred_position[idx_clas_pos, 1],
                                   ary_ez=y_pred_position[idx_clas_pos, 2],
                                   ary_px=y_pred_position[idx_clas_pos, 3],
                                   ary_py=y_pred_position[idx_clas_pos, 4],
                                   ary_pz=y_pred_position[idx_clas_pos, 5],
                                   filename=file_name,
                                   b_comptonkinematics=True,
                                   b_dacfilter=True,
                                   verbose=1)

        if mlem_export == "RECO":
            # denormalize features
            for i in range(DataCluster.features.shape[1]):
                DataCluster.features[:, i] *= DataCluster.list_std[i]
                DataCluster.features[:, i] += DataCluster.list_mean[i]

            # grab event kinematics from feature list
            ary_e = DataCluster.features[idx_clas_pos, 2]
            ary_ex = DataCluster.features[idx_clas_pos, 3]
            ary_ey = DataCluster.features[idx_clas_pos, 4]
            ary_ez = DataCluster.features[idx_clas_pos, 5]

            # select only absorber energies
            # select only positive events
            # replace -1. (NaN) values with 0.
            ary_p = DataCluster.features[:, [12, 22, 32, 42, 52]]
            ary_p = ary_p[idx_clas_pos, :]
            for i in range(ary_p.shape[0]):
                for j in range(ary_p.shape[1]):
                    if ary_p[i, j] == -1.:
                        ary_p[i, j] = 0.0
            ary_p = np.sum(ary_p, axis=1)

            ary_px = DataCluster.features[idx_clas_pos, 13]
            ary_py = DataCluster.features[idx_clas_pos, 14]
            ary_pz = DataCluster.features[idx_clas_pos, 15]

            MLEMExport.export_mlem(ary_e,
                                   ary_p,
                                   ary_ex,
                                   ary_ey,
                                   ary_ez,
                                   ary_px,
                                   ary_py,
                                   ary_pz,
                                   filename=file_name,
                                   b_comptonkinematics=True,
                                   b_dacfilter=True,
                                   verbose=1)

    """
    # plotting score distribution vs neural network error
    idx_clas_tp = []
    for i in range(len(y_scores)):
        if y_scores[i] > theta and DataCluster.targets_clas[i] == 1.0:
            idx_clas_tp.append(True)
        else:
            idx_clas_tp.append(False)
    Plotter.plot_2dhist_score_regE_error(y_scores[idx_clas_tp],
                                         y_pred_energy[idx_clas_tp, 0] - DataCluster.targets_reg1[idx_clas_tp, 0],
                                         "hist2d_score_error_energy_e")
    Plotter.plot_2dhist_score_regE_error(y_scores[idx_clas_tp],
                                         y_pred_energy[idx_clas_tp, 1] - DataCluster.targets_reg1[idx_clas_tp, 1],
                                         "hist2d_score_error_energy_p")
    
    # plot angle distribution for different subsets
    list_angle_pos = []
    list_angle_inpeak = []
    list_angle_offpeak = []
    list_angle_ic = []
    list_angle_tot = []
    for i in range(len(y_scores)):
        if DataCluster.targets_clas[i] == 1:
            list_angle_ic.append(calculate_theta(*y_pred_energy[i, :]))
        if y_scores[i] < 0.5:
            continue
        list_angle_pos.append(calculate_theta(*y_pred_energy[i, :]))
        if -8.0 < DataCluster.meta[i, 2] < 2.0:
            list_angle_inpeak.append(calculate_theta(*y_pred_energy[i, :]))
        if -20.0 < DataCluster.meta[i, 2] < -8.0:
            list_angle_offpeak.append(calculate_theta(*y_pred_energy[i, :]))

    Plotter.plot_angle_dist([list_angle_pos, list_angle_inpeak, list_angle_offpeak, list_angle_ic],
                            ["Positives", "Bragg Peak", "Tail", "IdealCompton"],
                            "dist_angle")

    # plot scattering cluster distance distribution for different subsets
    list_r_pos = []
    list_r_inpeak = []
    list_r_offpeak = []
    list_r_ic = []
    for i in range(len(y_scores)):
        r = np.sqrt(
            (y_pred_position[i, 1] - y_pred_position[i, 4]) ** 2 + (y_pred_position[i, 2] - y_pred_position[i, 5]) ** 2)
        r *= np.sign(y_pred_position[i, 2])
        if DataCluster.targets_clas[i] == 1:
            list_r_ic.append(r)
        if y_scores[i] < 0.5:
            continue
        list_r_pos.append(r)
        if -8.0 < DataCluster.meta[i, 2] < 2.0:
            list_r_inpeak.append(r)
        if -20.0 < DataCluster.meta[i, 2] < -8.0:
            list_r_offpeak.append(r)

    Plotter.plot_eucldist_dist([list_r_pos, list_r_inpeak, list_r_offpeak, list_r_ic],
                               ["Positives", "Bragg Peak", "Tail", "IdealCompton"],
                               "eucldist_angle")

    # source position plot heatmap
    list_sp_z = []
    list_sp_y = []
    for i in range(len(idx_clas_p)):
        if not idx_clas_p[i]:
            continue

        if Metrics.is_event_correct(int((y_scores[i] > theta) * 1),
                                    y_pred_energy[i],
                                    y_pred_position[i],
                                    int(DataCluster.targets_clas[i]),
                                    DataCluster.targets_reg1[i, :],
                                    DataCluster.targets_reg2[i, :]):
            list_sp_y.append(0)
            list_sp_z.append(DataCluster.meta[i, 2])
    Plotter.plot_sourceposition_heatmap(list_sp_z, list_sp_y, "heatmap_sourcepos")
    

    # collect full prediction and true values of test dataset
    y_pred_class = (y_scores[idx_clas_pos] > theta) * 1
    y_true_clas = DataCluster.targets_clas[idx_clas_pos]
    y_true_energy = DataCluster.targets_reg1[idx_clas_pos, :]
    y_true_position = DataCluster.targets_reg2[idx_clas_pos, :]
    y_pred_energy = y_pred_energy[idx_clas_pos, :]
    y_pred_position = y_pred_position[idx_clas_pos, :]

    efficiency, purity = Metrics.get_global_effpur(np.sum(DataCluster.targets_clas),
                                                   y_pred_class,
                                                   y_pred_energy,
                                                   y_pred_position,
                                                   y_true_clas,
                                                   y_true_energy,
                                                   y_true_position)
    
    print("# Full evaluation statistics: ")
    print("Efficiency: {:.1f}".format(efficiency * 100))
    print("Purity: {:.1f}".format(purity * 100))
    

    if mlem_export:
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


def export_mlem_simpleregression(NeuralNetwork_clas,
                                 DataCluster,
                                 file_name="",
                                 theta=0.5):
    # grab all positive identified events by the neural network
    y_scores = NeuralNetwork_clas.predict(DataCluster.features)
    # This is done this way cause y_scores gets a really dumb shape from tensorflow
    idx_clas_pos = [float(y_scores[i]) > theta for i in range(len(y_scores))]

    # denormalize features
    for i in range(DataCluster.features.shape[1]):
        DataCluster.features[:, i] *= DataCluster.list_std[i]
        DataCluster.features[:, i] += DataCluster.list_mean[i]

    # grab event kinematics from feature list
    ary_e = DataCluster.features[idx_clas_pos, 2]
    ary_ex = DataCluster.features[idx_clas_pos, 3]
    ary_ey = DataCluster.features[idx_clas_pos, 4]
    ary_ez = DataCluster.features[idx_clas_pos, 5]

    # select only absorber energies
    # select only positive events
    # replace -1. (NaN) values with 0.
    ary_p = DataCluster.features[:, [12, 22, 32, 42, 52, 63, 72]]
    ary_p = ary_p[idx_clas_pos, :]
    for i in range(ary_p.shape[0]):
        for j in range(ary_p.shape[1]):
            if ary_p[i, j] == -1.:
                ary_p[i, j] = 0.0
    ary_p = np.sum(ary_p, axis=1)

    ary_px = DataCluster.features[idx_clas_pos, 13]
    ary_py = DataCluster.features[idx_clas_pos, 14]
    ary_pz = DataCluster.features[idx_clas_pos, 15]

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
                           b_comptonkinematics=True,
                           b_dacfilter=True,
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


def eval_full_mod(NeuralNetwork_clas,
                  NeuralNetwork_regE,
                  NeuralNetwork_regP,
                  DataCluster,
                  file_name="",
                  theta=0.5):
    # grab all positive identified events by the neural network
    y_scores = NeuralNetwork_clas.predict(DataCluster.features)
    # This is done this way cause y_scores gets a really dumb shape from tensorflow
    idx_clas_pos = [float(y_scores[i]) > theta for i in range(len(y_scores))]

    # predict energy and position of all positive events
    y_pred_energy = NeuralNetwork_regE.predict(DataCluster.features)
    y_pred_position = NeuralNetwork_regP.predict(DataCluster.features)
