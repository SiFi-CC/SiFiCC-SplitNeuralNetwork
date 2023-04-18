import numpy as np

from src.SiFiCCNN import Utility

from src.SiFiCCNN.NeuralNetwork import NNExport, NNAnalysis
from src.SiFiCCNN.NeuralNetwork import FastROCAUC

from src.SiFiCCNN.Plotter import PTNetworkHistory, PTClassifier, PTRegression


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

    # Plotting of score distributions and ROC-analysis
    # grab optimal threshold from ROC-analysis
    PTClassifier.plot_score_distribution(y_scores, y_true, "score_dist")


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
