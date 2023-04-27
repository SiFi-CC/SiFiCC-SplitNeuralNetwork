import numpy as np

from src.SiFiCCNN import Utility

from src.SiFiCCNN.NeuralNetwork import NNExport, NNAnalysis
from src.SiFiCCNN.NeuralNetwork import FastROCAUC

from src.SiFiCCNN.Plotter import PTNetworkHistory, PTClassifier, PTRegression, PTCompare


def evaluate_classifier(NeuralNetwork,
                        DataCluster,
                        theta=0.5):
    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork.norm_mean, NeuralNetwork.norm_std)

    # grab neural network predictions for test dataset
    y_scores = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # ROC-AUC Analysis
    FastROCAUC.fastROCAUC(y_scores, y_true, save_fig="ROCAUC")
    _, theta_opt = FastROCAUC.fastROCAUC(y_scores, y_true, return_score=True)

    # Plot Neural Network training history
    PTNetworkHistory.plot_history_classifier(NeuralNetwork,
                                             (NeuralNetwork.model_name +
                                              "_" +
                                              NeuralNetwork.model_tag +
                                              "_history"))

    # write general binary classifier metrics into console and .txt file
    NNAnalysis.write_metrics_classifier(y_scores, y_true)

    # Generate efficiency map
    PTClassifier.plot_efficiencymap(y_scores,
                                    y_true,
                                    DataCluster.meta[DataCluster.idx_test(), 2],
                                    figure_name="efficiency_map",
                                    theta=theta)

    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    PTClassifier.plot_score_distribution(y_scores, y_true, "score_dist")

    # Saliency map examples
    NNAnalysis.get_saliency_examples(y_scores, y_true, NeuralNetwork, DataCluster)


def evaluate_regression_energy(NeuralNetwork,
                               DataCluster):
    # set regression
    DataCluster.update_targets_energy()
    DataCluster.update_indexing_positives()

    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork.norm_mean, NeuralNetwork.norm_std)

    # grab neural network predictions for test dataset
    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # Plot training history
    PTNetworkHistory.plot_history_regression(NeuralNetwork,
                                             (NeuralNetwork.model_name +
                                              "_" +
                                              NeuralNetwork.model_tag +
                                              "_history_training"))

    # Plot energy regression error
    PTRegression.plot_energy_error(y_pred, y_true, "error_regression_energy")

    # Plot theta error
    # Predicted theta is based on energy calculation from Compton scattering equation
    # True theta is based on true classification of theta regression (DotVec)
    y_true_theta = DataCluster.targets_reg3[DataCluster.idx_test()]
    y_pred_theta = np.array(
        [Utility.get_scattering_angle_energy(y_pred[i, 0], y_pred[i, 1]) for i in range(len(y_pred))])
    PTRegression.plot_theta_error(y_pred_theta, y_true_theta, "error_scatteringangle_energy")


def evaluate_regression_position(NeuralNetwork,
                                 DataCluster):
    # set regression
    DataCluster.update_targets_position()
    DataCluster.update_indexing_positives()

    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork.norm_mean, NeuralNetwork.norm_std)

    # grab neural network predictions for test dataset
    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # Plot training history
    PTNetworkHistory.plot_history_regression(NeuralNetwork,
                                             (NeuralNetwork.model_name +
                                              "_" +
                                              NeuralNetwork.model_tag +
                                              "_history_training"))

    # Plot position error
    PTRegression.plot_position_error(y_pred, y_true, "error_regression_position")


def evaluate_regression_theta(NeuralNetwork,
                              DataCluster):
    # set regression
    DataCluster.update_targets_theta()
    DataCluster.update_indexing_positives()

    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork.norm_mean, NeuralNetwork.norm_std)

    # grab neural network predictions for test dataset
    y_pred = NeuralNetwork.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # Plot training history
    PTNetworkHistory.plot_history_regression(NeuralNetwork,
                                             (NeuralNetwork.model_name +
                                              "_" +
                                              NeuralNetwork.model_tag +
                                              "_history_training"))

    # Plot scattering angle error
    PTRegression.plot_theta_error(y_pred, y_true, "error_regression_theta")


def eval_complete(NeuralNetwork_clas,
                  NeuralNetwork_regE,
                  NeuralNetwork_regP,
                  NeuralNetwork_regT,
                  DataCluster,
                  theta=0.5,
                  file_name="",
                  export_npz=False,
                  export_CC6=False):
    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork_clas.norm_mean, NeuralNetwork_clas.norm_std)

    # grab all positive identified events by the neural network
    y_scores = NeuralNetwork_clas.predict(DataCluster.features)
    y_pred_energy = NeuralNetwork_regE.predict(DataCluster.features)
    y_pred_position = NeuralNetwork_regP.predict(DataCluster.features)
    y_pred_theta = NeuralNetwork_regT.predict(DataCluster.features)

    # This is done this way cause y_scores gets a really dumb shape from tensorflow
    idx_clas_pos = [float(y_scores[i]) > theta for i in range(len(y_scores))]

    # create an array containing full neural network prediction
    ary_nn_pred = np.zeros(shape=(DataCluster.entries, 10))
    ary_nn_pred[:, 0] = np.reshape(y_scores, newshape=(len(y_scores),))
    ary_nn_pred[:, 1:3] = np.reshape(y_pred_energy, newshape=(y_pred_energy.shape[0], y_pred_energy.shape[1]))
    ary_nn_pred[:, 3:9] = np.reshape(y_pred_position, newshape=(y_pred_position.shape[0], y_pred_position.shape[1]))
    ary_nn_pred[:, 9] = np.reshape(y_pred_theta, newshape=(y_pred_position.shape[0],))

    if export_npz:
        NNExport.export_prediction_npz(ary_nn_pred=ary_nn_pred,
                                       file_name=file_name)

    if export_CC6:
        NNExport.export_prediction_cc6(ary_nn_pred=ary_nn_pred,
                                       file_name=file_name)


def eval_reco_compare(NeuralNetwork_regE,
                      NeuralNetwork_regP,
                      NeuralNetwork_regT,
                      DataCluster,
                      reco_file,
                      file_name=""):
    # load CB-Reconstruction results from npz file
    reco_npz_data = np.load(reco_file)
    ary_reco = reco_npz_data["CB_RECO"]

    # Normalize the evaluation data
    DataCluster.standardize(NeuralNetwork_regE.norm_mean, NeuralNetwork_regE.norm_std)
    DataCluster.update_indexing_positives()
    idx_pos = DataCluster.ary_idx

    # grab all positive identified events by the neural network
    y_pred_energy = NeuralNetwork_regE.predict(DataCluster.features[idx_pos])
    y_pred_position = NeuralNetwork_regP.predict(DataCluster.features[idx_pos])
    y_pred_theta = NeuralNetwork_regT.predict(DataCluster.features[idx_pos])

    y_reco_energy = ary_reco[idx_pos, 1:3]
    y_reco_position = ary_reco[idx_pos, 3:9]
    y_reco_theta = ary_reco[idx_pos, 9]

    y_true_energy = DataCluster.targets_reg1[idx_pos, :]
    y_true_position = DataCluster.targets_reg2[idx_pos, :]
    y_true_theta = DataCluster.targets_reg3[idx_pos]

    PTCompare.plot_compare_energy(y_pred_energy,
                                  y_reco_energy,
                                  y_true_energy,
                                  ["DNN", "Reco"],
                                  "reco_compare_energyregression")

    PTCompare.plot_compare_position(y_pred_position,
                                    y_reco_position,
                                    y_true_position,
                                    ["DNN", "Reco"],
                                    "reco_compare_positionregression")

    PTCompare.plot_compare_theta(y_pred_theta,
                                 y_reco_theta,
                                 y_true_theta,
                                 ["DNN", "Reco"],
                                 "reco_compare_thetaregression")
