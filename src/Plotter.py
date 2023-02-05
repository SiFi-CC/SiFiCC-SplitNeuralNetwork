import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


# --------------------------------------------------------
# fitting functions
def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


def plot_score_dist(y_scores, y_true, figure_name):
    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 0]

    plt.figure()
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step", color="orange", label="true positives")
    plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step", color="blue", label="true negatives")
    # plt.vlines(x=theta, ymin=0.0, ymax=len(ary_scores_neg)/2, color="red", linestyles="--", label="optimal threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_classifier(nn_classifier, figure_name):
    # plot model performance
    loss = nn_classifier.history['loss']
    val_loss = nn_classifier.history['val_loss']
    mse = nn_classifier.history["accuracy"]
    val_mse = nn_classifier.history["val_accuracy"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_regression(nn_regression, figure_name):
    loss = nn_regression.history['loss']
    val_loss = nn_regression.history['val_loss']
    mse = nn_regression.history["mean_absolute_error"]
    val_mse = nn_regression.history["val_mean_absolute_error"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("mean_absolute_error")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_energy_error(y_pred, y_true, figure_name):
    width = 0.01
    bins_err = np.arange(-2.0, 2.0, width)
    bins_energy = np.arange(0.0, 10.0, width)

    bins_err_center = bins_err[:-1] + (width / 2)

    hist0, _ = np.histogram(y_pred[:, 0] - y_true[:, 0], bins=bins_err)
    hist1, _ = np.histogram(y_pred[:, 1] - y_true[:, 1], bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_center, hist1, p0=[0.0, 1.0, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure()
    plt.title("Electron Energy resolution")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt1), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt1[0], popt1[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$E_{Pred}$ - $E_{True}$")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0], bins=[bins_energy, bins_err], norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_relative.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$E_{Pred}$ - $E_{True}$")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1], bins=[bins_energy, bins_err], norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_relative.png")
    plt.close()


def plot_position_error(y_pred, y_true, figure_name):
    width = 0.1
    bins_err_x = np.arange(-20.5, 20.5, width)
    bins_err_y = np.arange(-60.5, 60.5, width)
    bins_err_z = np.arange(-20.5, 20.5, width)

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
    popt0, pcov0 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist1, p0=[0.0, 20.0, np.sum(hist1) * width])
    popt2, pcov2 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist2, p0=[0.0, 1.0, np.sum(hist2) * width])
    popt3, pcov3 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist3, p0=[0.0, 1.0, np.sum(hist3) * width])
    popt4, pcov4 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist4, p0=[0.0, 20.0, np.sum(hist4) * width])
    popt5, pcov5 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist5, p0=[0.0, 1.0, np.sum(hist5) * width])

    ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
    ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
    ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

    plt.figure()
    plt.title("Electron position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Electron")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0], bins=[bins_x[:209], bins_err_x], norm=LogNorm())
    plt.xlim(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0)
    plt.hlines(xmin=150.0 - 20.8 / 2.0, xmax=150.0 + 20.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x_relative.png")
    plt.close()

    plt.figure()
    plt.title("Electron position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt1), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt1[0], popt1[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Electron")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y_relative.png")
    plt.close()

    plt.figure()
    plt.title("Electron position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt2), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt2[0], popt2[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Electron")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$")
    plt.hist2d(x=y_true[:, 2], y=y_pred[:, 2] - y_true[:, 2], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z_relative.png")
    plt.close()

    # ----------------------------------------------------------

    plt.figure()
    plt.title("Photon position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt3), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt3[0], popt3[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Photon")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$")
    plt.hist2d(x=y_true[:, 3], y=y_pred[:, 3] - y_true[:, 3], bins=[bins_x[467:], bins_err_x], norm=LogNorm())
    plt.xlim(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0)
    plt.hlines(xmin=270.0 - 46.8 / 2.0, xmax=270.0 + 46.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x_relative.png")
    plt.close()

    plt.figure()
    plt.title("Photon position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt4), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt4[0], popt4[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Photon")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$")
    plt.hist2d(x=y_true[:, 4], y=y_pred[:, 4] - y_true[:, 4], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y_relative.png")
    plt.close()

    plt.figure()
    plt.title("Photon position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt5), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt5[0], popt5[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Photon")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$")
    plt.hist2d(x=y_true[:, 5], y=y_pred[:, 5] - y_true[:, 5], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z_relative.png")
    plt.close()


def plot_source_position(ary_sp_pos, ary_sp_tp, ary_sp_tot, figure_name):
    # plot MC Source Position z-direction
    bins = np.arange(-80, 20, 1.0)
    width = abs(bins[0] - bins[1])
    hist1, _ = np.histogram(ary_sp_pos, bins=bins)
    hist2, _ = np.histogram(ary_sp_tp, bins=bins)
    hist3, _ = np.histogram(ary_sp_tot, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(ary_sp_tot, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist3, np.sqrt(hist3), color="orange", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black", fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red", fmt=".", label="NN positive\nevents")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_primary_energy_dist(ary_pe_pos, ary_pe_tp, ary_pe_tot, figure_name):
    # plot MC Source Position z-direction
    bins = np.arange(0.0, 10.0, 0.1)
    width = abs(bins[0] - bins[1])
    hist1, _ = np.histogram(ary_pe_pos, bins=bins)
    hist2, _ = np.histogram(ary_pe_tp, bins=bins)
    hist3, _ = np.histogram(ary_pe_tot, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy Primary")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    plt.xlim(0.0, 10.0)
    # total event histogram
    plt.hist(ary_pe_tot, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist3, np.sqrt(hist3), color="orange", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black", fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red", fmt=".", label="NN positive\nevents")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_2dhist_score_sourcepos(ary_score, ary_sp, figure_name):
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_sp = np.arange(-80.0, 20.0, 1.0)

    list_score = []
    list_sp = []
    for i in range(len(ary_score)):
        if not ary_sp[i] == 0.0:
            list_score.append(ary_score[i])
            list_sp.append(ary_sp[i])

    plt.figure()
    plt.xlabel("MC Source Position z [mm]")
    plt.ylabel("Score")
    plt.hist2d(list_sp, list_score, bins=[bin_sp, bin_score])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_2dhist_score_eprimary(ary_score, ary_ep, figure_name):
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_sp = np.arange(0.0, 16.0, 0.1)

    list_score = []
    list_sp = []
    for i in range(len(ary_score)):
        if not ary_ep[i] == 0.0:
            list_score.append(ary_score[i])
            list_sp.append(ary_ep[i])

    plt.figure()
    plt.xlabel("MC Primary Energy [MeV]")
    plt.ylabel("Score")
    h0 = plt.hist2d(list_sp, list_score, bins=[bin_sp, bin_score], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_2dhist_score_regE_error(ary_score, ary_regE_err, figure_name):
    bin_score = np.arange(0.4, 1.0, 0.05)
    bin_sp = np.arange(-2.0, 2.0, 0.05)

    ary_score = np.reshape(ary_score, (len(ary_score),))
    ary_regE_err = np.reshape(ary_regE_err, (len(ary_regE_err),))

    plt.figure()
    plt.xlabel(r" $E_{Reco} - E_{True}$ [MeV]")
    plt.ylabel("Score")
    h0 = plt.hist2d(ary_regE_err, ary_score, bins=[bin_sp, bin_score], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_angle_dist(list_angles, list_labels, figure_name):
    bins = np.arange(0.0, 3/4*np.pi, 0.01)

    plt.figure()
    plt.xlabel(r"$\theta$ [rad]")
    plt.ylabel("counts (normalized)")
    for i in range(len(list_angles)):
        plt.hist(list_angles[i], bins=bins, histtype=u"step", density=True, label=list_labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()

def plot_eucldist_dist(list_r, list_labels, figure_name):
    bins = np.arange(-100.0, 100.0, 1.0)

    plt.figure()
    plt.xlabel(r"$r_{eucl} \cdot sign(e_z)$ [mm]")
    plt.ylabel("counts (normalized)")
    for i in range(len(list_r)):
        plt.hist(list_r[i], bins=bins, histtype=u"step", density=True, label=list_labels[i])
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_sourceposition_heatmap(ary_sp_z, ary_sp_y, figure_name):
    bins_z = np.arange(-80, 20, 1.0)
    bins_y = np.arange(-20, 20, 1.0)

    hist0, _, _ = np.histogram2d(x=ary_sp_z, y=ary_sp_y, bins=[bins_z, bins_y])
    hist1, _ = np.histogram(ary_sp_z, bins=bins_z)

    # normalize histogram to range [0.0, 1.0]
    min_val, max_val = np.min(hist0), np.max(hist0)
    hmap = (hist0 - min_val) / (max_val - min_val + 1e-10)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].set_xlabel("MC Source Position z [mm]")
    axs[0].set_ylabel("MC Source Position y [mm]")
    im0 = axs[0].imshow(hmap)

    axs[1].set_xlabel("MC Source Position z [mm]")
    axs[1].set_ylabel("Counts")
    axs[1].hist(ary_sp_z, bins=bins_z, histtype=u"step", color="black", linestyle="--", alpha=0.7)
    axs[1].errorbar(bins_z[:-1] - 0.5, hist1, np.sqrt(hist1), fmt=".", color="black")

    # color-bars
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_backprojection(image, figure_name):
    plt.figure()
    plt.imshow(image)
    # plt.savefig(figure_name + ".png")
    plt.show()
