import numpy as np
import os
import pickle as pkl

from src.SiFiCCNN.Plotter import PTNetworkCompare
from src.SiFiCCNN.NeuralNetwork import NeuralNetwork
from src.SiFiCCNN.Data import DFParser

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

dir_final = dir_plots + "NNCompare/"

# plot results in specific folder
if not os.path.isdir(dir_final):
    os.mkdir(dir_final)


# ----------------------------------------------------------------------------------------------------------------------

def compare_neuralnetworks(run_name1,
                           run_name2):
    # load classifier histories
    with open(dir_results + run_name1 + "/" + run_name1 + "_clas" + ".hst", 'rb') as f_hist:
        history_clas1 = pkl.load(f_hist)
    with open(dir_results + run_name2 + "/" + run_name2 + "_clas" + ".hst", 'rb') as f_hist:
        history_clas2 = pkl.load(f_hist)

    nn1_loss = history_clas1["loss"]
    nn1_val_loss = history_clas1["val_loss"]
    nn1_eff = history_clas1["recall"]
    nn1_val_eff = history_clas1["val_recall"]
    nn1_pur = history_clas1["precision"]
    nn1_val_pur = history_clas1["val_precision"]
    nn2_loss = history_clas2["loss"]
    nn2_val_loss = history_clas2["val_loss"]
    nn2_eff = history_clas2["recall"]
    nn2_val_eff = history_clas2["val_recall"]
    nn2_pur = history_clas2["precision"]
    nn2_val_pur = history_clas2["val_precision"]

    PTNetworkCompare.plot_compare_classifier(nn1_loss,
                                             nn1_val_loss,
                                             nn1_eff,
                                             nn1_val_eff,
                                             nn1_pur,
                                             nn1_val_pur,
                                             nn2_loss,
                                             nn2_val_loss,
                                             nn2_eff,
                                             nn2_val_eff,
                                             nn2_pur,
                                             nn2_val_pur,
                                             ["DNN", "RNN"],
                                             dir_final + "DNNS4A6_RNNS4A6_classifier")

    # load regression energy history
    # load classifier histories
    with open(dir_results + run_name1 + "/" + run_name1 + "_regE" + ".hst", 'rb') as f_hist:
        history_regE1 = pkl.load(f_hist)
    with open(dir_results + run_name2 + "/" + run_name2 + "_regE" + ".hst", 'rb') as f_hist:
        history_regE2 = pkl.load(f_hist)

    nn1_loss = history_regE1["loss"]
    nn1_val_loss = history_regE1["val_loss"]
    nn2_loss = history_regE2["loss"]
    nn2_val_loss = history_regE2["val_loss"]

    PTNetworkCompare.plot_compare_regression_loss(nn1_loss,
                                                  nn1_val_loss,
                                                  nn2_loss,
                                                  nn2_val_loss,
                                                  ["DNN", "RNN"],
                                                  dir_final + "DNNS4A6_RNNS4A6_regE")

    # load regression position history
    with open(dir_results + run_name1 + "/" + run_name1 + "_regP" + ".hst", 'rb') as f_hist:
        history_regP1 = pkl.load(f_hist)
    with open(dir_results + run_name2 + "/" + run_name2 + "_regP" + ".hst", 'rb') as f_hist:
        history_regP2 = pkl.load(f_hist)

    nn1_loss = history_regP1["loss"]
    nn1_val_loss = history_regP1["val_loss"]
    nn2_loss = history_regP2["loss"]
    nn2_val_loss = history_regP2["val_loss"]

    PTNetworkCompare.plot_compare_regression_position(nn1_loss,
                                                      nn1_val_loss,
                                                      nn2_loss,
                                                      nn2_val_loss,
                                                      ["DNN", "RNN"],
                                                      dir_final + "DNNS4A6_RNNS4A6_regP")

    # load regression theta history
    with open(dir_results + run_name1 + "/" + run_name1 + "_regT" + ".hst", 'rb') as f_hist:
        history_regT1 = pkl.load(f_hist)
    with open(dir_results + run_name2 + "/" + run_name2 + "_regT" + ".hst", 'rb') as f_hist:
        history_regT2 = pkl.load(f_hist)

    nn1_loss = history_regT1["loss"]
    nn1_val_loss = history_regT1["val_loss"]
    nn2_loss = history_regT2["loss"]
    nn2_val_loss = history_regT2["val_loss"]

    PTNetworkCompare.plot_compare_regression_loss(nn1_loss,
                                                  nn1_val_loss,
                                                  nn2_loss,
                                                  nn2_val_loss,
                                                  ["DNN", "RNN"],
                                                  dir_final + "DNNS4A6_RNNS4A6_regT")


# compare_neuralnetworks("DNN_S4A6_master", "RNN_S4A6_master")


# ----------------------------------------------------------------------------------------------------------------------

def compare_prediction(DNN_name,
                       RNN_name):
    from src.SiFiCCNN.Model import DNN_SXAX_classifier
    from src.SiFiCCNN.Model import DNN_SXAX_regression_energy
    from src.SiFiCCNN.Model import DNN_SXAX_regression_position
    from src.SiFiCCNN.Model import DNN_SXAX_regression_theta

    from src.SiFiCCNN.Model import RNN_SXAX_classifier
    from src.SiFiCCNN.Model import RNN_SXAX_regression_energy
    from src.SiFiCCNN.Model import RNN_SXAX_regression_position
    from src.SiFiCCNN.Model import RNN_SXAX_regression_theta

    NPZ_FILE_TRAIN = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_RNN_S4A6.npz"

    # get prediction DNN
    tfmodel_clas = DNN_SXAX_classifier.return_model(10, 10)
    tfmodel_regE = DNN_SXAX_regression_energy.return_model(10, 10)
    tfmodel_regP = DNN_SXAX_regression_position.return_model(10, 10)
    tfmodel_regT = DNN_SXAX_regression_theta.return_model(10, 10)

    neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                     model_name=DNN_name,
                                                     model_tag="clas")
    neuralnetwork_regE = NeuralNetwork.NeuralNetwork(model=tfmodel_regE,
                                                     model_name=DNN_name,
                                                     model_tag="regE")
    neuralnetwork_regP = NeuralNetwork.NeuralNetwork(model=tfmodel_regP,
                                                     model_name=DNN_name,
                                                     model_tag="regP")
    neuralnetwork_regT = NeuralNetwork.NeuralNetwork(model=tfmodel_regT,
                                                     model_name=DNN_name,
                                                     model_tag="regT")

    os.chdir(dir_results + DNN_name + "/")
    neuralnetwork_clas.load()
    neuralnetwork_regE.load()
    neuralnetwork_regP.load()
    neuralnetwork_regT.load()

    # data cluster
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN, n_frac=1.0)
    data_cluster.standardize(neuralnetwork_clas.norm_mean, neuralnetwork_clas.norm_std)
    idx_pos = data_cluster.targets_clas == 1
    n_pos = int(np.sum(data_cluster.targets_clas))

    # grab all positive identified events by the neural network
    y_scores = neuralnetwork_clas.predict(data_cluster.features[idx_pos, :])
    y_pred_energy = neuralnetwork_regE.predict(data_cluster.features[idx_pos, :])
    y_pred_position = neuralnetwork_regP.predict(data_cluster.features[idx_pos, :])
    y_pred_theta = neuralnetwork_regT.predict(data_cluster.features[idx_pos, :])

    # create an array containing full neural network prediction
    ary_dnn_pred = np.zeros(shape=(n_pos, 10))
    ary_dnn_pred[:, 0] = np.reshape(y_scores, newshape=(len(y_scores),))
    ary_dnn_pred[:, 1:3] = np.reshape(y_pred_energy, newshape=(y_pred_energy.shape[0], y_pred_energy.shape[1]))
    ary_dnn_pred[:, 3:9] = np.reshape(y_pred_position, newshape=(y_pred_position.shape[0], y_pred_position.shape[1]))
    ary_dnn_pred[:, 9] = np.reshape(y_pred_theta, newshape=(y_pred_position.shape[0],))

    # get prediction RNN
    tfmodel_clas = RNN_SXAX_classifier.return_model(10, 10)
    tfmodel_regE = RNN_SXAX_regression_energy.return_model(10, 10)
    tfmodel_regP = RNN_SXAX_regression_position.return_model(10, 10)
    tfmodel_regT = RNN_SXAX_regression_theta.return_model(10, 10)

    neuralnetwork_clas = NeuralNetwork.NeuralNetwork(model=tfmodel_clas,
                                                     model_name=RNN_name,
                                                     model_tag="clas")
    neuralnetwork_regE = NeuralNetwork.NeuralNetwork(model=tfmodel_regE,
                                                     model_name=RNN_name,
                                                     model_tag="regE")
    neuralnetwork_regP = NeuralNetwork.NeuralNetwork(model=tfmodel_regP,
                                                     model_name=RNN_name,
                                                     model_tag="regP")
    neuralnetwork_regT = NeuralNetwork.NeuralNetwork(model=tfmodel_regT,
                                                     model_name=RNN_name,
                                                     model_tag="regT")

    os.chdir(dir_results + RNN_name + "/")
    neuralnetwork_clas.load()
    neuralnetwork_regE.load()
    neuralnetwork_regP.load()
    neuralnetwork_regT.load()

    # data cluster
    data_cluster = DFParser.parse_cluster(dir_npz + NPZ_FILE_TRAIN, n_frac=1.0)
    data_cluster.standardize(neuralnetwork_clas.norm_mean, neuralnetwork_clas.norm_std)


    # grab all positive identified events by the neural network
    y_scores = neuralnetwork_clas.predict(data_cluster.features[idx_pos, :])
    y_pred_energy = neuralnetwork_regE.predict(data_cluster.features[idx_pos, :])
    y_pred_position = neuralnetwork_regP.predict(data_cluster.features[idx_pos, :])
    y_pred_theta = neuralnetwork_regT.predict(data_cluster.features[idx_pos, :])

    # create an array containing full neural network prediction
    ary_rnn_pred = np.zeros(shape=(n_pos, 10))
    ary_rnn_pred[:, 0] = np.reshape(y_scores, newshape=(len(y_scores),))
    ary_rnn_pred[:, 1:3] = np.reshape(y_pred_energy, newshape=(y_pred_energy.shape[0], y_pred_energy.shape[1]))
    ary_rnn_pred[:, 3:9] = np.reshape(y_pred_position, newshape=(y_pred_position.shape[0], y_pred_position.shape[1]))
    ary_rnn_pred[:, 9] = np.reshape(y_pred_theta, newshape=(y_pred_position.shape[0],))

    # comparison plots for network predictions
    PTNetworkCompare.plot_compare_energy(ary_dnn_pred[:, 1:3],
                                         ary_rnn_pred[:, 1:3],
                                         data_cluster.targets_reg1,
                                         dir_final + "DNNS4A6_RNNS4A6_energyregression")
    PTNetworkCompare.plot_compare_position(ary_dnn_pred[:, 3:9],
                                           ary_rnn_pred[:, 3:9],
                                           data_cluster.targets_reg2,
                                           dir_final + "DNNS4A6_RNNS4A6_positionregression")
    PTNetworkCompare.plot_compare_theta(ary_dnn_pred[:, 9],
                                        ary_rnn_pred[:, 9],
                                        data_cluster.targets_reg3,
                                        dir_final + "DNNS4A6_RNNS4A6_thetaregression")


compare_prediction("DNN_S4A6_master", "RNN_S4A6_master")
