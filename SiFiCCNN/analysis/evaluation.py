from SiFiCCNN.analysis.fastROCAUC import fastROCAUC
from SiFiCCNN.analysis import metrics, fastROCAUC
from SiFiCCNN.plotting import plt_models
from SiFiCCNN.utils import physics


def eval_classifier(y_scores,
                    y_true,
                    theta=0.5):
    # ROC-AUC Analysis
    _, theta_opt, (list_fpr, list_tpr) = fastROCAUC(y_scores,
                                                    y_true,
                                                    return_score=True)
    plt_models.roc_curve(list_fpr, list_tpr, "rocauc_curve")

    # Plotting of score distributions
    plt_models.score_distribution(y_scores, y_true, "score_dist")

    # write general binary classifier metrics into console and .txt file
    metrics.write_metrics_classifier(y_scores, y_true)


def eval_regression_energy(y_pred,
                           y_true):
    plt_models.plot_energy_error(y_pred, y_true, "error_regression_energy")


def eval_regression_position(y_pred,
                             y_true):
    # Plot position error
    plt_models.plot_position_error(y_pred, y_true,
                                   "error_regression_position")


def eval_regression_theta(y_pred,
                          y_true):
    # Plot scattering angle error
    plt_models.plot_theta_error(y_pred, y_true, "error_regression_theta")
