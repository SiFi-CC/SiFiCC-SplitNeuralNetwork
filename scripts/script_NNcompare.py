import numpy as np
import os
import pickle as pkl

from src.SiFiCCNN.Plotter import PTNetworkCompare

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


compare_neuralnetworks("DNN_S4X6", "RNN_S4X6")
