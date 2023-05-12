import numpy as np

from fastROCAUC import fastROCAUC

from src.SiFiCCNN.plotting.mpl_classifier import roc_curve, score_distribution
from src.SiFiCCNN.analysis.utils import get_classifier_metrics, \
    write_metrics_classifier


def classifier_metrics(y_scores,
                       y_true,
                       theta=0.5):
    # ROC-AUC Analysis
    _, theta_opt, (list_fpr, list_tpr) = fastROCAUC(y_scores,
                                                    y_true,
                                                    return_score=True)
    roc_curve(list_fpr, list_tpr, "rocauc_curve")

    # Plotting of score distributions
    score_distribution(y_scores, y_true, "score_dist")

    # write general binary classifier metrics into console and .txt file
    write_metrics_classifier(y_scores, y_true)
