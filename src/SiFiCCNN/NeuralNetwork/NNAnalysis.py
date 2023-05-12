import numpy as np

from src.SiFiCCNN.NeuralNetwork import FastROCAUC

from src.SiFiCCNN.plotting import PTSaliencyMap

# ----------------------------------------------------------------------------------------------------------------------
# Saliency mapping

def get_saliency_examples(y_pred, y_true, NeuralNetwork, DataFrame):
    # grab TP, FP, TN, FN example from network predictions
    # TP example
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5 and y_true[i] == 1.0:
            smap = PTSaliencyMap.get_smap(NeuralNetwork.model, DataFrame.features[i, :])
            PTSaliencyMap.smap_plot(smap, DataFrame.features[i, :], "Event " + str(i), "smap_example_TP")
            break
    for i in range(len(y_pred)):
        if y_pred[i] > 0.5 and y_true[i] == 0.0:
            smap = PTSaliencyMap.get_smap(NeuralNetwork.model, DataFrame.features[i, :])
            PTSaliencyMap.smap_plot(smap, DataFrame.features[i, :], "Event " + str(i), "smap_example_FP")
            break
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5 and y_true[i] == 0.0:
            smap = PTSaliencyMap.get_smap(NeuralNetwork.model, DataFrame.features[i, :])
            PTSaliencyMap.smap_plot(smap, DataFrame.features[i, :], "Event " + str(i), "smap_example_TN")
            break
    for i in range(len(y_pred)):
        if y_pred[i] < 0.5 and y_true[i] == 0.0:
            smap = PTSaliencyMap.get_smap(NeuralNetwork.model, DataFrame.features[i, :])
            PTSaliencyMap.smap_plot(smap, DataFrame.features[i, :], "Event " + str(i), "smap_example_FN")
            break
