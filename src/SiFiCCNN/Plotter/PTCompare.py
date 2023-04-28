import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


# ----------------------------------------------------------------------------------------------------------------------
# Fit functions needed for regression fits

def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


# ----------------------------------------------------------------------------------------------------------------------

def plot_compare_classifier(nn1_loss,
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
                            labels,
                            figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})
    y_upperlim = max(np.max(nn1_val_eff), np.max(nn2_val_eff)) + 0.1
    y_lowerlim = min(np.min(nn1_val_pur), np.min(nn2_val_pur)) - 0.1

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_loss, label=labels[0] + " Loss", linestyle='-', color="blue")
    plt.plot(nn1_val_loss, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_loss, label=labels[1] + " Loss", linestyle='-', color="orange")
    plt.plot(nn2_val_loss, label="Validation", linestyle='--', color="orange")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()

    plt.fig, axs1 = plt.subplots(figsize=(8, 6))
    axs1.set_xlabel("Epochs")
    axs1.set_ylabel("Efficiency", color="red")
    axs1.tick_params(axis='y', labelcolor="red")
    axs1.set_ylim(0.2, 0.9)
    axs1.plot(nn1_eff, label=labels[0] + " Efficiency", linestyle="-", color="black")
    axs1.plot(nn1_val_eff, linestyle="--", color="black")
    axs1.plot(nn2_eff, label=labels[1] + " Efficiency", linestyle="-", color="red")
    axs1.plot(nn2_val_eff, linestyle="--", color="red")

    axs2 = axs1.twinx()
    axs2.set_ylabel("Purity", color="blue")
    axs2.tick_params(axis='y', labelcolor="blue")
    axs2.set_ylim(0.2, 0.9)
    axs2.plot(nn1_pur, label=labels[0] + " Purity", linestyle="-", color="black")
    axs2.plot(nn1_val_pur, linestyle="--", color="black")
    axs2.plot(nn2_pur, label=labels[1] + " Purity", linestyle="-", color="blue")
    axs2.plot(nn2_val_pur, linestyle="--", color="blue")

    axs1.grid()
    plt.tight_layout()
    axs1.legend(loc="lower left")
    axs2.legend(loc="lower right")
    plt.savefig(figure_name + "_metrics" + ".png")
    plt.close()


def plot_compare_regression_loss(nn1_mae,
                                 nn1_val_mae,
                                 nn2_mae,
                                 nn2_val_mae,
                                 labels,
                                 figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_mae, label=labels[0] + " MAE", linestyle='-', color="blue")
    plt.plot(nn1_val_mae, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_mae, label=labels[1] + " MAE", linestyle='-', color="orange")
    plt.plot(nn2_val_mae, label="Validation", linestyle='--', color="orange")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()


def plot_compare_regression_position(nn1_mae,
                                     nn1_val_mae,
                                     nn2_mae,
                                     nn2_val_mae,
                                     labels,
                                     figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_mae, label=labels[0] + " MAE", linestyle='-', color="blue")
    plt.plot(nn1_val_mae, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_mae, label=labels[1] + " MAE", linestyle='-', color="orange")
    plt.plot(nn2_val_mae, label="Validation", linestyle='--', color="orange")
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()


# ----------------------------------------------------------------------------------------------------------------------
#

def plot_compare_energy(y_pred0,
                        y_pred1,
                        y_true,
                        labels,
                        figure_name):
    plt.rcParams.update({'font.size': 16})

    # reco positives
    idx_pos = y_pred1[:, 0] != 0.0

    width = 0.01
    bins_err = np.arange(-1.0 + width, 1.0 - width, width)
    bins_err_center = bins_err[:-1] + (width / 2)

    hist00, _ = np.histogram((y_pred0[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err)
    hist01, _ = np.histogram((y_pred0[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err)
    hist10, _ = np.histogram((y_pred1[idx_pos, 0] - y_true[idx_pos, 0]) / y_true[idx_pos, 0], bins=bins_err)
    hist11, _ = np.histogram((y_pred1[idx_pos, 1] - y_true[idx_pos, 1]) / y_true[idx_pos, 1], bins=bins_err)

    # fitting energy resolution
    popt00, pcov00 = curve_fit(gaussian, bins_err_center, hist00, p0=[0.0, 1.0, np.sum(hist00) * width])
    popt01, pcov01 = curve_fit(lorentzian, bins_err_center, hist01, p0=[0.0, 0.5, np.sum(hist01) * width])
    popt10, pcov10 = curve_fit(gaussian, bins_err_center, hist10, p0=[0.0, 1.0, np.sum(hist10) * width])
    popt11, pcov11 = curve_fit(lorentzian, bins_err_center, hist11, p0=[0.0, 0.5, np.sum(hist11) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron energy resolution")
    plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.ylabel("counts")
    plt.hist((y_pred0[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err, histtype=u"step", color="red",
             label=labels[0])
    plt.plot(ary_x, gaussian(ary_x, *popt00), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt00[0], popt00[1]))
    plt.hist((y_pred1[idx_pos, 0] - y_true[idx_pos, 0]) / y_true[idx_pos, 0], bins=bins_err, histtype=u"step",
             color="black",
             label=labels[1], alpha=0.5)
    plt.plot(ary_x, gaussian(ary_x, *popt10), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt10[0], popt10[1]))
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
    plt.hist((y_pred0[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err, histtype=u"step", color="red",
             label=labels[0])
    plt.plot(ary_x, lorentzian(ary_x, *popt01), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt01[0], popt01[1] / 2))
    plt.hist((y_pred1[idx_pos, 1] - y_true[idx_pos, 1]) / y_true[idx_pos, 1], bins=bins_err, histtype=u"step",
             color="black",
             label=labels[1], alpha=0.5)
    plt.plot(ary_x, lorentzian(ary_x, *popt11), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt11[0], popt11[1] / 2))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()


def plot_compare_position(y_pred0,
                          y_pred1,
                          y_true,
                          labels,
                          figure_name):
    plt.rcParams.update({'font.size': 16})

    # reco positives
    idx_pos = y_pred1[:, 0] != 0.0

    width = 0.1
    bins_err_x = np.arange(-5.5, 5.5, width)
    bins_err_y = np.arange(-60.5, 60.5, width)
    bins_err_z = np.arange(-5.5, 5.5, width)

    bins_x = np.arange(150.0 - 20.8 / 2.0, 270.0 + 46.8 / 2.0, width)
    bins_y = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
    bins_z = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)

    hist00, _ = np.histogram(y_pred0[:, 0] - y_true[:, 0], bins=bins_err_x)
    hist01, _ = np.histogram(y_pred0[:, 1] - y_true[:, 1], bins=bins_err_y)
    hist02, _ = np.histogram(y_pred0[:, 2] - y_true[:, 2], bins=bins_err_z)
    hist03, _ = np.histogram(y_pred0[:, 3] - y_true[:, 3], bins=bins_err_x)
    hist04, _ = np.histogram(y_pred0[:, 4] - y_true[:, 4], bins=bins_err_y)
    hist05, _ = np.histogram(y_pred0[:, 5] - y_true[:, 5], bins=bins_err_z)
    hist10, _ = np.histogram(y_pred1[idx_pos, 0] - y_true[idx_pos, 0], bins=bins_err_x)
    hist11, _ = np.histogram(y_pred1[idx_pos, 1] - y_true[idx_pos, 1], bins=bins_err_y)
    hist12, _ = np.histogram(y_pred1[idx_pos, 2] - y_true[idx_pos, 2], bins=bins_err_z)
    hist13, _ = np.histogram(y_pred1[idx_pos, 3] - y_true[idx_pos, 3], bins=bins_err_x)
    hist14, _ = np.histogram(y_pred1[idx_pos, 4] - y_true[idx_pos, 4], bins=bins_err_y)
    hist15, _ = np.histogram(y_pred1[idx_pos, 5] - y_true[idx_pos, 5], bins=bins_err_z)

    # fitting position resolution
    popt00, pcov00 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist00, p0=[0.0, 1.0, np.sum(hist00) * width])
    popt01, pcov01 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist01, p0=[0.0, 20.0, np.sum(hist01) * width])
    popt02, pcov02 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist02, p0=[0.0, 1.0, np.sum(hist02) * width])
    popt03, pcov03 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist03, p0=[0.0, 1.0, np.sum(hist03) * width])
    popt04, pcov04 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist04, p0=[0.0, 20.0, np.sum(hist04) * width])
    popt05, pcov05 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist05, p0=[0.0, 1.0, np.sum(hist05) * width])
    popt10, pcov10 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist10, p0=[0.0, 1.0, np.sum(hist10) * width])
    popt11, pcov11 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist11, p0=[0.0, 20.0, np.sum(hist11) * width])
    popt12, pcov12 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist12, p0=[0.0, 1.0, np.sum(hist12) * width])
    popt13, pcov13 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist13, p0=[0.0, 1.0, np.sum(hist13) * width])
    popt14, pcov14 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist14, p0=[0.0, 20.0, np.sum(hist14) * width])
    popt15, pcov15 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist15, p0=[0.0, 1.0, np.sum(hist15) * width])

    ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
    ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
    ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_x, gaussian(ary_x, *popt00), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt00[0], popt00[1]))
    plt.hist(y_pred1[idx_pos, 0] - y_true[idx_pos, 0], bins=bins_err_x, histtype=u"step", color="black", alpha=0.5,
             label=labels[1])
    plt.plot(ary_x, gaussian(ary_x, *popt10), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt10[0], popt10[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_y, gaussian(ary_y, *popt01), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt01[0], popt01[1]))
    plt.hist(y_pred1[idx_pos, 1] - y_true[idx_pos, 1], bins=bins_err_y, histtype=u"step", color="black", alpha=0.5,
             label=labels[1])
    plt.plot(ary_y, gaussian(ary_y, *popt11), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt11[0], popt11[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_z, gaussian(ary_z, *popt02), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt02[0], popt02[1]))
    plt.hist(y_pred1[idx_pos, 2] - y_true[idx_pos, 2], bins=bins_err_z, histtype=u"step", color="black",
             label=labels[1], alpha=0.5)
    plt.plot(ary_z, gaussian(ary_z, *popt12), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt12[0], popt12[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    # ----------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_x, gaussian(ary_x, *popt03), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt03[0], popt03[1]))
    plt.hist(y_pred1[idx_pos, 3] - y_true[idx_pos, 3], bins=bins_err_x, histtype=u"step", color="black", alpha=0.5,
             label=labels[1])
    plt.plot(ary_x, gaussian(ary_x, *popt13), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt13[0], popt13[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_y, gaussian(ary_y, *popt04), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt04[0], popt04[1]))
    plt.hist(y_pred1[idx_pos, 4] - y_true[idx_pos, 4], bins=bins_err_y, histtype=u"step", color="black", alpha=0.5,
             label=labels[1])
    plt.plot(ary_y, gaussian(ary_y, *popt14), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt14[0], popt14[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred0[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_z, gaussian(ary_z, *popt05), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt05[0], popt05[1]))
    plt.hist(y_pred1[idx_pos, 5] - y_true[idx_pos, 5], bins=bins_err_z, histtype=u"step", color="black", alpha=0.5,
             label=labels[1])
    plt.plot(ary_z, gaussian(ary_z, *popt15), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt15[0], popt15[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()


def plot_compare_theta(y_pred0,
                       y_pred1,
                       y_true,
                       labels,
                       figure_name):
    y_pred0 = np.reshape(y_pred0, newshape=(len(y_pred0),))
    y_pred1 = np.reshape(y_pred1, newshape=(len(y_pred1),))
    y_true = np.reshape(y_true, newshape=(len(y_true),))

    # reco positives
    idx_pos = y_pred1 != 0.0

    width = 0.01
    bins_err = np.arange(-np.pi, np.pi, width)
    bins_err_center = bins_err[:-1] + (width / 2)
    hist0, _ = np.histogram(y_pred0 - y_true, bins=bins_err)
    hist1, _ = np.histogram(y_pred1[idx_pos] - y_true[idx_pos], bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_center, hist1, p0=[0.0, 1.0, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure()
    plt.title("Error scattering angle")
    plt.xlabel(r"$\theta_{Pred}$ - $\theta_{True}$ [rad]")
    plt.ylabel("counts")
    plt.hist(y_pred0 - y_true, bins=bins_err, histtype=u"step", color="red", label=labels[0])
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="red",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt0[0], popt0[1]))
    plt.hist(y_pred1[idx_pos] - y_true[idx_pos], bins=bins_err, histtype=u"step", color="black", label=labels[1],
             alpha=0.5)
    plt.plot(ary_x, gaussian(ary_x, *popt1), color="black", alpha=0.5,
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt1[0], popt1[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()
