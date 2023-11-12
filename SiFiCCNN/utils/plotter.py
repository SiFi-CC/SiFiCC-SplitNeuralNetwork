####################################################################################################
#
# ### plotter.py
#
# This file contains all plotting methods used to evaluate neural network results. Some specialized
# plots are not included here and used in their corresponding analysis scripts. These methods are
# used to generate automated results plots and specialized plots for theses and papers should be
# created in their own stand-alone scripts. Most methods are tuned to  be presentable in 4:3
# Power-Point slides.
#
####################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


####################################################################################################
# Classifier plots
####################################################################################################


def plot_score_distribution(y_scores, y_true, figure_name):
    """
    Plots the score distribution of the neural network score for true background and true signal
    events. A marker is placed on the decision boundary.

    Args:
        y_scores:
        y_true:
        figure_name:

    Returns:

    """
    # update matplotlib parameter for bigger font size
    plt.rcParams.update({'font.size': 20})

    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.02, 0.02)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 0]

    plt.figure(figsize=(14, 8))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Signal score")
    plt.ylabel("Counts")
    plt.xlim(0.0, 1.0)
    plt.hist(np.array(ary_scores_pos), bins=bins, color="deeppink",
             label="True positives", alpha=0.25)
    plt.hist(np.array(ary_scores_neg), bins=bins, color="blue",
             label="True negatives", alpha=0.25)
    h0, _, _ = plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step",
                        color="deeppink")
    h1, _, _ = plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step",
                        color="blue")
    plt.vlines(x=0.5, ymin=0.0, ymax=max([max(h0), max(h1)]), color="red",
               label="Decision boundary")
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_efficiencymap(y_pred, y_true, y_sp, figure_name, theta=0.5, sr=100):
    """
    Plots the efficiency against the true source position.

    Args:
        y_pred:
        y_true:
        y_sp:
        figure_name:
        theta:
        sr:

    Returns:

    """
    # determine contribution from each prediction
    # Done by giving correct prediction weight 1, else weight 0
    ary_w = np.zeros(shape=(len(y_pred),))
    for i in range(len(ary_w)):
        if y_true[i] == 1:
            if y_pred[i] > theta:
                ary_w[i] = 1

    # determine binning from source position array
    bin_min = int(min(y_sp))
    bin_max = int(max(y_sp))
    ary_bin = np.linspace(bin_min, bin_max, sr)
    width = abs(ary_bin[1] - ary_bin[0])
    ary_bin_err = np.ones(shape=(len(ary_bin) - 1,)) * width / 2

    # Create histogram from prediction and truth
    hist0, _ = np.histogram(y_sp, bins=ary_bin, weights=y_true)
    hist1, _ = np.histogram(y_sp, bins=ary_bin, weights=ary_w)

    # determine efficiency from histogram
    ary_eff = np.zeros(shape=(len(hist0),))
    ary_eff_err = np.zeros(shape=(len(hist0),))
    for i in range(len(ary_eff)):
        if hist1[i] < 10:
            continue
        else:
            ary_eff[i] = hist1[i] / hist0[i]
            ary_eff_err[i] = (np.sqrt((np.sqrt(hist1[i]) / hist0[i]) ** 2 +
                                      (hist1[i] * np.sqrt(hist0[i]) / hist0[
                                          i] ** 2) ** 2))

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Source position efficiency")
    axs[0].set_ylabel("Counts")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=y_true,
                color="black", alpha=0.5, label="Dist. Compton")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=ary_w,
                color="red", alpha=0.5, linestyle="--",
                label="Prediction")
    axs[0].legend(loc="upper right")
    axs[1].set_ylim(0, 1)
    axs[1].grid()
    axs[1].set_xlabel("Source Position z-axis [mm]")
    axs[1].set_ylabel("Efficiency")
    axs[1].errorbar(ary_bin[:-1] + width, ary_eff, ary_eff_err, ary_bin_err,
                    fmt=".", color="blue")
    # axs[1].plot(bins[:-1] + 0.5, ary_eff, ".", color="darkblue")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_roc_curve(list_fpr,
                   list_tpr,
                   figure_name,
                   weighted=False):
    if weighted:
        auc_label = "weightedAUC"
    else:
        auc_label = "AUC"

    # calc area under ROC
    auc_score = 0
    for i in range(len(list_fpr) - 1):
        # Riemann-sum to calculate area under curve
        area = (list_fpr[i + 1] - list_fpr[i]) * list_tpr[i]
        # multiply result by -1, since x values are ordered from highest to
        # lowest
        auc_score += area * (-1)

    print("Plotting ROC curve and {}...".format(auc_label))
    plt.figure()
    plt.title("ROC Curve | " + auc_label)
    plt.plot(list_fpr, list_tpr, color="red",
             label="{0:}: {1:.3f}".format(auc_label, auc_score))
    plt.plot([0, 1], [0, 1], color="black", ls="--")
    # plt.plot(dot[0], dot[1], 'b+')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_sp_distribution(ary_sp,
                         ary_score,
                         ary_true,
                         figure_name):
    # plot MC Source Position z-direction
    width = 1.0
    bins = np.arange(int(min(ary_sp)), int(max(ary_sp)), width)

    idx_tp = []
    for i in range(len(ary_sp)):
        if ary_score[i] > 0.5 and ary_true[i] == 1.0:
            idx_tp.append(True)
        else:
            idx_tp.append(False)
    idx_tp = np.array(idx_tp)

    hist0, _ = np.histogram(ary_sp[ary_sp != 0.0], bins=bins)
    hist1, _ = np.histogram(ary_sp[idx_tp], bins=bins)
    hist2, _ = np.histogram(ary_sp[ary_true == 1.0], bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(ary_sp[ary_sp != 0.0], bins=bins, color="orange", alpha=0.5,
             label="All events")
    plt.errorbar(bins[1:] - width / 2, hist0, np.sqrt(hist0), color="orange",
                 fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black",
                 fmt=".", label="Dist. Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red",
                 fmt=".", label="True Positive events")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_pe_distribution(ary_pe,
                         ary_score,
                         ary_true,
                         figure_name):
    # plot MC Source Position z-direction
    width = 0.1
    bins = np.arange(0.0, 10.0, width)

    idx_tp = []
    for i in range(len(ary_pe)):
        if ary_score[i] > 0.5 and ary_true[i] == 1.0:
            idx_tp.append(True)
        else:
            idx_tp.append(False)
    idx_tp = np.array(idx_tp)

    hist0, _ = np.histogram(ary_pe, bins=bins)
    hist1, _ = np.histogram(ary_pe[idx_tp], bins=bins)
    hist2, _ = np.histogram(ary_pe[ary_true == 1.0], bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy Primary")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    plt.xlim(0.0, 10.0)
    # total event histogram
    plt.hist(ary_pe, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist0, np.sqrt(hist0), color="orange",
                 fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black",
                 fmt=".", label="Dist. Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red",
                 fmt=".", label="True Positive events")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


################################################################################
# Regression plots
################################################################################


def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


def max_super_function(x):
    return (0.1 + np.exp((0.5 * (x + 3)) / 2)) / (
            1 + np.exp((8 * x + 5) / 3)) / 6


def plot_energy_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})
    width = 0.1
    bins_err = np.arange(-2.0, 2.0, width)
    bins_energy = np.arange(0.0, 10.0, width)

    bins_err_center = bins_err[:-1] + (width / 2)

    hist0, _ = np.histogram((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0],
                            bins=bins_err)
    hist1, _ = np.histogram((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1],
                            bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0,
                             p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_center, hist1,
                             p0=[0.0, 1.0, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$(E_{Pred} - E_{True}) / E_{True}$")
    plt.ylabel("counts")
    plt.hist((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err,
             histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f} ".format(
                 popt0[0], np.sqrt(pcov0[0, 0]), popt0[1], np.sqrt(pcov0[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$(E_{Pred} - E_{True}) / E_{True}$")
    plt.ylabel("counts")
    plt.hist((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err,
             histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt1), color="green",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt1[0], np.sqrt(pcov1[0, 0]), popt1[1], np.sqrt(pcov1[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()

    plt.figure()
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 0], y=(y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0],
               bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(figure_name + "_electron_relative.png")
    plt.close()

    plt.figure()
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 1], y=(y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1],
               bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.savefig(figure_name + "_photon_relative.png")
    plt.close()


def plot_position_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    width = 0.1
    bins_err_x = np.arange(-5.5, 5.5, width)
    bins_err_y = np.arange(-60.5, 60.5, width)
    bins_err_z = np.arange(-5.5, 5.5, width)

    bins_x = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)
    bins_y = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
    bins_z = np.arange(150.0 - 20.8 / 2.0, 270.0 + 46.8 / 2.0, width)

    hist0, _ = np.histogram(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x)
    hist1, _ = np.histogram(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y)
    hist2, _ = np.histogram(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z)
    hist3, _ = np.histogram(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x)
    hist4, _ = np.histogram(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y)
    hist5, _ = np.histogram(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z)

    # fitting position resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist0,
                             p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist1,
                             p0=[0.0, 20.0, np.sum(hist1) * width])
    popt2, pcov2 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist2,
                             p0=[0.0, 1.0, np.sum(hist2) * width])
    popt3, pcov3 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist3,
                             p0=[0.0, 1.0, np.sum(hist3) * width])
    popt4, pcov4 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist4,
                             p0=[0.0, 20.0, np.sum(hist4) * width])
    popt5, pcov5 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist5,
                             p0=[0.0, 1.0, np.sum(hist5) * width])

    ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
    ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
    ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{x}$ - $r^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step",
             color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt0[0], np.sqrt(pcov0[0, 0]), popt0[1], np.sqrt(pcov0[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{x}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{x}$ - $r^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0],
               bins=[bins_x, bins_err_x], norm=LogNorm())
    plt.hlines(xmin=min(bins_x), xmax=max(bins_x), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{y}$ - $r^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step",
             color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt1), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt1[0], np.sqrt(pcov1[0, 0]), popt1[1], np.sqrt(pcov1[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{y}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{y}$ - $r^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1],
               bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{z}$ - $r^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step",
             color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt2), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt2[0], np.sqrt(pcov2[0, 0]), popt2[1], np.sqrt(pcov2[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{z}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{z}$ - $r^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 2], y=y_pred[:, 2] - y_true[:, 2],
               bins=[bins_z[:209], bins_err_z], norm=LogNorm())
    plt.xlim(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0)
    plt.hlines(xmin=150.0 - 20.8 / 2.0, xmax=150.0 + 20.8 / 2.0, y=0,
               color="red", linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z_relative.png")
    plt.close()

    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{x}$ - $r^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step",
             color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt3), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt3[0], np.sqrt(pcov3[0, 0]), popt3[1], np.sqrt(pcov3[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{x}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{x}$ - $r^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 3], y=y_pred[:, 3] - y_true[:, 3],
               bins=[bins_x, bins_err_x], norm=LogNorm())
    plt.hlines(xmin=min(bins_x), xmax=max(bins_x), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{y}$ - $r^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step",
             color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt4), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt4[0], np.sqrt(pcov4[0, 0]), popt4[1], np.sqrt(pcov4[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{y}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{y}$ - $r^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 4], y=y_pred[:, 4] - y_true[:, 4],
               bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel(r"$r^{Pred}_{z}$ - $r^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step",
             color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt5), color="deeppink",
             label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                 popt5[0], np.sqrt(pcov5[0, 0]), popt5[1], np.sqrt(pcov5[1, 1])))
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()

    plt.figure()
    plt.xlabel("$r^{True}_{z}$ [mm]")
    plt.ylabel(r"$r^{Pred}_{z}$ - $r^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 5], y=y_pred[:, 5] - y_true[:, 5],
               bins=[bins_z[467:], bins_err_z], norm=LogNorm())
    plt.xlim(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0)
    plt.hlines(xmin=270.0 - 46.8 / 2.0, xmax=270.0 + 46.8 / 2.0, y=0,
               color="red", linestyles="--")
    plt.colorbar()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z_relative.png")
    plt.close()


def plot_theta_error(y_pred, y_true, figure_name):
    y_pred = np.reshape(y_pred, newshape=(len(y_pred),))
    y_true = np.reshape(y_true, newshape=(len(y_true),))

    width = 0.01
    bins_err = np.arange(-np.pi, np.pi, width)
    bins_theta = np.arange(0.0, np.pi, width)

    bins_err_center = bins_err[:-1] + (width / 2)
    hist0, _ = np.histogram(y_pred - y_true, bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0,
                             p0=[0.0, 1.0, np.sum(hist0) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure()
    plt.title("Error scattering angle")
    plt.xlabel(r"$\theta_{Pred}$ - $\theta_{True}$ [rad]")
    plt.ylabel("counts")
    plt.hist(y_pred - y_true, bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt0[0],
                                                                  popt0[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()

    plt.figure()
    plt.title("Error scattering angle")
    plt.xlabel(r"$\theta_{True}$ [rad]")
    plt.ylabel(r"$\theta_{Pred}$ - $\theta_{True}$")
    plt.hist2d(x=y_true, y=y_pred - y_true, bins=[bins_theta, bins_err],
               norm=LogNorm(vmin=1.0, vmax=800))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_relative.png")
    plt.close()


################################################################################
# Training history
################################################################################


def plot_history_classifier(history, figure_name):
    # update matplotlib parameter for bigger font size
    plt.rcParams.update({'font.size': 20})

    loss = history['loss']
    val_loss = history['val_loss']
    # mse = nn_classifier.history["accuracy"]
    # val_mse = nn_classifier.history["val_accuracy"]
    eff = history["recall"]
    val_eff = history["val_recall"]
    pur = history["precision"]
    val_pur = history["val_precision"]

    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(loss, label="Loss", linestyle='-', color="blue")
    ax1.plot(val_loss, label="Validation", linestyle='--', color="blue")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper center")
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax2 = ax1.twinx()
    ax2.plot(eff, label="Efficiency", linestyle='-', color="deeppink")
    ax2.plot(val_eff, label="Validation", linestyle='--', color="deeppink")
    ax2.plot(pur, label="Purity", linestyle="-", color="green")
    ax2.plot(val_pur, label="Validation", linestyle="--", color="green")
    ax2.set_ylabel("%")
    ax2.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_classifier_fancy(history, figure_name):
    # Fancy means that this plot is generally made for presentations and paper
    plt.rcParams.update({'font.size': 16})
    # plot model performance
    loss = history['loss']
    val_loss = history['val_loss']
    # mse = nn_classifier.history["accuracy"]
    # val_mse = nn_classifier.history["val_accuracy"]

    eff = history["recall"]
    val_eff = history["val_recall"]
    pur = history["precision"]
    val_pur = history["val_precision"]

    fig, ax1 = plt.subplots(figsize=(9, 8))

    ax1.plot(loss, label="Loss", linestyle='-', color="blue")
    ax1.plot(val_loss, label="Validation", linestyle='--', color="blue")
    ax1.set_ylim(0.38, 0.8)
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="center left")
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    ax1.minorticks_on()
    ax2 = ax1.twinx()
    ax2.plot(eff, label="Efficiency", linestyle='-.', color="deeppink")
    ax2.plot(val_eff, label="Validation", linestyle='--', color="deeppink")
    ax2.plot(pur, label="Purity", linestyle=":", color="green")
    ax2.plot(val_pur, label="Validation", linestyle="--", color="green")
    ax2.set_ylabel("Efficiency and Purity in %")
    ax2.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_regression(history, figure_name):
    plt.rcParams.update({'font.size': 16})

    loss = history['loss']
    val_loss = history['val_loss']
    mse = history["mean_absolute_error"]
    val_mse = history["val_mean_absolute_error"]

    plt.figure(figsize=(7, 6))
    plt.plot(loss, label="Loss", linestyle='-', color="blue")
    plt.plot(val_loss, label="Validation", linestyle='--', color="blue")
    # plt.plot(mse, label="MAE", linestyle='-', color="red")
    # plt.plot(val_mse, linestyle='--', color="red")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_regression_fancy(historyE, historyP, figure_name):
    plt.rcParams.update({'font.size': 16})

    lossE = historyE['loss']
    val_lossE = historyE['val_loss']
    mseE = historyE["mean_absolute_error"]
    val_mseE = historyE["val_mean_absolute_error"]

    lossP = historyP['loss']
    val_lossP = historyP['val_loss']
    mseP = historyP["mean_absolute_error"]
    val_mseP = historyP["val_mean_absolute_error"]

    fig, axs = plt.subplots(figsize=(8, 8), nrows=2, sharex=True)
    axs[0].plot(lossE, label="Energy loss", linestyle='-', color="blue")
    axs[0].plot(val_lossE, label="Validation", linestyle='--', color="blue")
    # axs[0].plot(mseE, label="MAE", linestyle='-', color="pink")
    # axs[0].plot(val_mseE, label="Validation", linestyle='--', color="pink")
    # axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("loss")
    axs[0].legend()
    axs[0].grid(which='major', color='#CCCCCC', linewidth=0.8)
    axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    axs[0].minorticks_on()

    axs[1].plot(lossP, label="Position loss", linestyle='-', color="blue")
    axs[1].plot(val_lossP, label="Validation", linestyle='--', color="blue")
    # axs[1].plot(mseP, label="MAE", linestyle='-', color="pink")
    # axs[1].plot(val_mseP, label="Validation", linestyle='--', color="pink")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("loss")
    axs[1].legend()
    axs[1].grid(which='major', color='#CCCCCC', linewidth=0.8)
    axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    axs[1].minorticks_on()

    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


####################################################################################################
# 2d histograms
####################################################################################################


def plot_2dhist_sp_score(sp, y_score, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_sp = np.arange(int(min(sp)), 50, 1.0)

    plt.figure(figsize=(7, 6))
    plt.xlabel("True source position z-axis [mm]")
    plt.ylabel("Signal score")
    h0 = plt.hist2d(sp[y_true == 1], y_score[y_true == 1], bins=[bin_sp, bin_score])
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.grid(which='major', color='#CCCCCC', linewidth=0.5, alpha=0.5)
    plt.grid(which='minor', color='#DDDDDD', linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_2dhist_ep_score(pe, y_score, y_true, figure_name):
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_pe = np.arange(0.0, 16.0, 0.1)

    plt.figure()
    plt.xlabel("MC Primary Energy [MeV]")
    plt.ylabel("Score")
    h0 = plt.hist2d(pe[y_true == 1], y_score[y_true == 1], bins=[bin_pe, bin_score], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.grid(which='major', color='#DDDDDD', linewidth=0.5, alpha=0.5)
    plt.grid(which='minor', color='#EEEEEE', linestyle=":", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_energy_resolution(y_pred, y_true, figure_name):
    # general settings
    plt.rcParams.update({'font.size': 16})

    # method for determining an estimation of the full-width-half-maximum for fitting
    def get_fit_range(hist, a=5, b=3):
        # determine maximum of histogram
        idx_max = np.argmax(hist)
        val_max = hist[idx_max]
        # grab all entries above maximum
        ary_tmp = np.zeros(shape=(len(hist)))
        for i in range(len(ary_tmp)):
            if hist[i] > val_max / 2:
                ary_tmp[i] = 1
        idx_left = 0
        idx_right = 0
        for i in range(len(ary_tmp)):
            if ary_tmp[i] == 1:
                idx_left = i
                idx_right = abs(idx_max - idx_left) * 2 + i
                break
        return max(idx_left - a, 0), min(len(hist), idx_right + b)

    def gaussian_lin_bg(x, mu, sigma, A, m, b):
        return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2) + \
               (m * x + b)

    # binning energy error
    bins_err_1 = np.arange(-0.5, 0.5 + 0.005, 0.005)
    bins_err_2 = np.arange(-3.0, 3.0 + 0.05, 0.02)
    bins_err_3 = np.arange(-6.0, 6.0 + 0.1, 0.1)
    bins_err_4 = np.arange(-8.0, 8.0 + 0.2, 0.2)

    # binning true energy
    # bin width increases for higher energies to increase statistics per bin-range
    bins_energy_e = np.concatenate([[0.0, 0.2],
                                    np.arange(0.3, 1.0, 0.1),
                                    np.arange(1.0, 4.6, 0.2),
                                    [4.6, 5.0, 6.0, 30.0]])
    bin_energy_e_center = np.array(
        [bins_energy_e[i] + (bins_energy_e[i + 1] - bins_energy_e[i]) / 2 for i in
         range(len(bins_energy_e) - 1)])
    bin_energy_e_center_err = np.array(
        [(bins_energy_e[i + 1] - bins_energy_e[i]) / 2 for i in range(len(bins_energy_e) - 1)])

    bins_energy_p = np.concatenate([[0.0, 0.3, 0.5],
                                    np.arange(0.6, 1.0, 0.1),
                                    np.arange(1.0, 4.6, 0.2),
                                    [4.6, 5.0, 6.0, 7.0, 30.0]])
    bin_energy_p_center = np.array(
        [bins_energy_p[i] + (bins_energy_p[i + 1] - bins_energy_p[i]) / 2 for i in
         range(len(bins_energy_p) - 1)])
    bin_energy_p_center_err = np.array(
        [(bins_energy_p[i + 1] - bins_energy_p[i]) / 2 for i in range(len(bins_energy_p) - 1)])

    ary_res_e = np.zeros(shape=(len(bin_energy_e_center),))
    ary_res_e_err = np.zeros(shape=(len(bin_energy_e_center),))
    ary_res_p = np.zeros(shape=(len(bin_energy_p_center),))
    ary_res_p_err = np.zeros(shape=(len(bin_energy_p_center),))

    # main iteration over energy bins for electron energy
    for i in range(len(bins_energy_e) - 1):
        ary_ee_cut = np.where(
            (bins_energy_e[i] < y_true[:, 0]) & (bins_energy_e[i + 1] > y_true[:, 0]),
            y_true[:, 0], y_true[:, 0] * 0)

        # initialization
        bins = bins_err_4
        a = 3
        b = 5

        if i in [0, 1, 2, 3]:
            bins = bins_err_1
            a = 20
            b = 20

        if i in [4, 5, 6, 7, 8]:
            bins = bins_err_1
            a = 5
            b = 5

        if i in [30]:
            bins = bins_err_4
            a = -5
            b = -5

        bins_center = np.array(
            [bins[i] + (bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)])
        width = bins[1] - bins[0]

        # electron energy resolution fitting plots
        plt.figure()
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Counts")
        plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
        plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
        plt.minorticks_on()
        hist_ee, _, _ = plt.hist((y_pred[:, 0] - y_true[:, 0])[ary_ee_cut != 0], bins=bins,
                                 histtype=u"step", color="blue",
                                 label=r"{:.1f} < $E_e$ <= {:.1f}".format(bins_energy_e[i],
                                                                          bins_energy_e[i + 1]))

        # get optimized fitting range
        # (frl = fit range left, frr = fit range right)
        e_frl, e_frr = get_fit_range(hist_ee, a=a, b=b)

        # calc additional quantities for better fitting estimation
        y0 = hist_ee[e_frl]
        y1 = hist_ee[e_frr]
        p0_m = (y1 - y0) / (bins_center[e_frr] - bins_center[e_frl])
        p0_b = (y1 - y0) / 2

        print("fit iteration {} [{}, {}]".format(i, e_frl, e_frr))
        # fit gaussian to histogram
        popt_e, pcov_e = curve_fit(gaussian_lin_bg, bins_center[e_frl:e_frr],
                                   hist_ee[e_frl:e_frr],
                                   p0=[0.0, 1.0, np.sum(hist_ee[e_frl:e_frr]) * width, p0_m, p0_b])
        plt.plot(bins_center[e_frl:e_frr],
                 gaussian_lin_bg(bins_center[e_frl:e_frr], *popt_e), color="deeppink",
                 label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                     popt_e[0], np.sqrt(pcov_e[0, 0]), popt_e[1], np.sqrt(pcov_e[1, 1])))

        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(figure_name + "_ee_{}".format(i))
        plt.close()

        ary_res_e[i] = abs(popt_e[1])
        ary_res_e_err[i] = abs(np.sqrt(pcov_e[1, 1]))

    # main iteration over energy bins for electron energy
    for i in range(len(bins_energy_p) - 1):
        ary_ep_cut = np.where(
            (bins_energy_p[i] < y_true[:, 1]) & (bins_energy_p[i + 1] > y_true[:, 1]),
            y_true[:, 1], y_true[:, 1] * 0)

        # initialization
        bins = bins_err_4
        a = 2
        b = 5

        if i in [0, 1, 2, 3]:
            bins = bins_err_4
            a = 4

        if i in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
            bins = bins_err_3
            a = 2

        if i in [31]:
            bins = bins_err_4
            a = -20
            b = -20

        bins_center = np.array(
            [bins[i] + (bins[i + 1] - bins[i]) / 2 for i in range(len(bins) - 1)])
        width = bins[1] - bins[0]

        # photon energy resolution fitting plots
        plt.figure()
        plt.xlabel("Energy [MeV]")
        plt.ylabel("Counts")
        plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
        plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
        plt.minorticks_on()
        hist_ep, _, _ = plt.hist((y_pred[:, 1] - y_true[:, 1])[ary_ep_cut != 0], bins=bins,
                                 histtype=u"step", color="blue",
                                 label="{:.1f} < {} <= {:.1f}".format(bins_energy_p[i],
                                                                      r"$E_{\gamma}$",
                                                                      bins_energy_p[i + 1]))

        # get optimized fitting range
        # (frl = fit range left, frr = fit range right)
        p_frl, p_frr = get_fit_range(hist_ep, a=a, b=b)

        # calc additional quantities for better fitting estimation
        y0 = hist_ep[p_frl]
        y1 = hist_ep[p_frr]
        p0_m = (y1 - y0) / (bins_center[p_frr] - bins_center[p_frl])
        p0_b = (y1 - y0) / 2

        print("fit iteration {} [{}, {}]".format(i, p_frl, p_frr))
        # fit gaussian to histogram
        popt_p, pcov_p = curve_fit(gaussian_lin_bg, bins_center[p_frl:p_frr],
                                   hist_ep[p_frl:p_frr],
                                   p0=[0.0, 1.0, np.sum(hist_ep[p_frl:p_frr]) * width, p0_m, p0_b])
        plt.plot(bins_center[p_frl:p_frr],
                 gaussian_lin_bg(bins_center[p_frl:p_frr], *popt_p), color="deeppink",
                 label=r"$\mu$ = {:.2f} $\pm$ {:.2f}""\n"r"$\sigma$ = {:.2f} $\pm$ {:.2f}".format(
                     popt_p[0], np.sqrt(pcov_p[0, 0]), popt_p[1], np.sqrt(pcov_p[1, 1])))

        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(figure_name + "_ep_{}".format(i))
        plt.close()

        ary_res_p[i] = abs(popt_p[1])
        ary_res_p_err[i] = abs(np.sqrt(pcov_p[1, 1]))

    plt.figure(figsize=(12, 6))
    plt.xlabel("Energy [MeV]")
    plt.ylabel("Energy Resolution [%]")
    plt.xlim(0, 8)
    plt.ylim(0, 25)
    plt.errorbar(bin_energy_e_center,
                 ary_res_e / bin_energy_e_center * 100, ary_res_e_err / bin_energy_e_center * 100,
                 bin_energy_e_center_err, fmt=".", color="deeppink", label=r"$E_e$")
    plt.errorbar(bin_energy_p_center,
                 ary_res_p / bin_energy_p_center * 100, ary_res_p_err / bin_energy_p_center * 100,
                 bin_energy_p_center_err, fmt=".", color="blue", label=r"$E_{\gamma}}$")
    plt.legend(loc="upper right")
    plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
    plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
    plt.minorticks_on()
    plt.tight_layout()
    plt.savefig(figure_name)
