import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


################################################################################
# Classifier plots
################################################################################


def plot_score_distribution(y_scores, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 0]

    plt.figure(figsize=(8, 6))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Signal score")
    plt.ylabel("counts")
    plt.hist(np.array(ary_scores_pos), bins=bins, color="orange",
             label="True positives", alpha=0.25)
    plt.hist(np.array(ary_scores_neg), bins=bins, color="blue",
             label="True negatives", alpha=0.25)
    h0, _, _ = plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step",
                        color="orange")
    h1, _, _ = plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step",
                        color="blue")
    plt.vlines(x=0.5, ymin=0.0, ymax=max([max(h0), max(h1)]), color="red",
               label="Decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_efficiencymap(y_pred, y_true, y_sp, figure_name, theta=0.5, sr=100):
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
    popt1, pcov1 = curve_fit(lorentzian, bins_err_center, hist1,
                             p0=[0.0, 0.5, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron energy resolution")
    plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.ylabel("counts")
    plt.hist((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err,
             histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0],
                                                                    popt0[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon energy resolution")
    plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.ylabel("counts")
    plt.hist((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err,
             histtype=u"step", color="blue")
    plt.plot(ary_x, lorentzian(ary_x, *popt1), color="green",
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt1[0],
                                                                  popt1[1] / 2))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 0], y=(y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0],
               bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_relative.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 1], y=(y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1],
               bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_relative.png")
    plt.close()

    # added new combined plot for better visibility
    plt.figure(figsize=(8, 5))
    plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.ylabel("counts")
    plt.xlim(-2, 2)
    plt.hist((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err,
             histtype=u"step", color="blue", label=r"$e^{-}$")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="blue", linestyle="--",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0],
                                                                    popt0[1]))
    plt.hist((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 0], bins=bins_err,
             histtype=u"step", color="green", label=r"$\gamma$")
    plt.plot(ary_x, gaussian(ary_x, *popt1), color="green", linestyle="--",
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt1[0],
                                                                  popt1[1] / 2))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_position_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    width = 0.1
    bins_err_x = np.arange(-5.5, 5.5, width)
    bins_err_y = np.arange(-60.5, 60.5, width)
    bins_err_z = np.arange(-5.5, 5.5, width)

    bins_x = np.arange(150.0 - 20.8 / 2.0, 270.0 + 46.8 / 2.0, width)
    bins_y = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
    bins_z = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)

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
    plt.title("Electron position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step",
             color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0],
                                                                    popt0[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Electron")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0],
               bins=[bins_x[:209], bins_err_x], norm=LogNorm())
    plt.xlim(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0)
    plt.hlines(xmin=150.0 - 20.8 / 2.0, xmax=150.0 + 20.8 / 2.0, y=0,
               color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step",
             color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt1), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt1[0],
                                                                    popt1[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Electron")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1],
               bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step",
             color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt2), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt2[0],
                                                                    popt2[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Electron")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 2], y=y_pred[:, 2] - y_true[:, 2],
               bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z_relative.png")
    plt.close()

    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step",
             color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt3), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt3[0],
                                                                    popt3[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Photon")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 3], y=y_pred[:, 3] - y_true[:, 3],
               bins=[bins_x[467:], bins_err_x], norm=LogNorm())
    plt.xlim(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0)
    plt.hlines(xmin=270.0 - 46.8 / 2.0, xmax=270.0 + 46.8 / 2.0, y=0,
               color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step",
             color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt4), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt4[0],
                                                                    popt4[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Photon")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 4], y=y_pred[:, 4] - y_true[:, 4],
               bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red",
               linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step",
             color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt5), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt5[0],
                                                                    popt5[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Photon")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 5], y=y_pred[:, 5] - y_true[:, 5],
               bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red",
               linestyles="--")
    plt.colorbar()
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
    # TODO: make this one plot
    plt.rcParams.update({'font.size': 16})
    # plot model performance
    loss = history['loss']
    val_loss = history['val_loss']
    # mse = nn_classifier.history["accuracy"]
    # val_mse = nn_classifier.history["val_accuracy"]

    eff = history["recall"]
    val_eff = history["val_recall"]
    pur = history["precision"]
    val_pur = history["precision"]

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(221)

    ax1.plot(loss, label="Loss", linestyle='-', color="blue")
    ax1.plot(val_loss, label="Validation", linestyle='--', color="blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.grid(which="both")

    ax2.plot(eff, label="Efficiency", linestyle='-', color="red")
    ax2.plot(val_eff, label="Validation", linestyle='--', color="red")
    ax2.plot(pur, label="Purity", linestyle="-", color="green")
    ax2.plot(val_pur, label="Validation", linestyle="--", color="green")
    ax2.set_ylabel("%")
    ax2.legend(loc="lower right")
    ax2.grid(which="both")

    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_regression(history, figure_name):
    plt.rcParams.update({'font.size': 16})

    loss = history['loss']
    val_loss = history['val_loss']
    mse = history["mean_absolute_error"]
    val_mse = history["val_mean_absolute_error"]

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Loss", linestyle='-', color="blue")
    plt.plot(val_loss, label="Validation", linestyle='--', color="blue")
    # plt.plot(mse, label="MAE", linestyle='-', color="red")
    # plt.plot(val_mse, linestyle='--', color="red")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(-5, 100)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


####################################################################################################
# 2d histograms
####################################################################################################


def plot_2dhist_sp_score(sp, y_score, y_true, figure_name):
    plot_bragg = True

    plt.rcParams.update({'font.size': 16})
    bin_score = np.arange(0.0, 1.0, 0.01)
    bin_sp = np.arange(int(min(sp)), int(max(sp)), 1.0)

    plt.figure(figsize=(8, 6))
    plt.xlabel("True source position z-axis [mm]")
    plt.ylabel("Signal score")
    h0 = plt.hist2d(sp[y_true == 1], y_score[y_true == 1], bins=[bin_sp, bin_score])
    if plot_bragg:
        x = np.linspace(-80.0, 5.0, 1000)
        mu = -2.5
        sigma = 1.5
        A = 0.8
        B = 0.2
        plt.plot(x, max_super_function(x), color="red", linestyle="--")

    plt.colorbar(h0[3])
    plt.tight_layout()
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
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()
