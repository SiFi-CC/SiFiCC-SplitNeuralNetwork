import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------------------------------------------------
# Fit functions needed for regression fits

def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


def max_super_function(x):
    return (0.1 + np.exp((0.5 * (x + 3)) / 2)) / (1 + np.exp((8 * x + 5) / 3)) / 6


# ----------------------------------------------------------------------------------------------------------------------


def plot_energy_error(y_pred, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})
    width = 0.1
    bins_err = np.arange(-1.0, 1.0, width)
    bins_energy = np.arange(0.0, 10.0, width)

    bins_err_center = bins_err[:-1] + (width / 2)

    hist0, _ = np.histogram((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err)
    hist1, _ = np.histogram((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err)

    # fitting energy resolution
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(lorentzian, bins_err_center, hist1, p0=[0.0, 0.5, np.sum(hist1) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron energy resolution")
    plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.ylabel("counts")
    plt.hist((y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]))
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
    plt.hist((y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, lorentzian(ary_x, *popt1), color="green",
             label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt1[0], popt1[1] / 2))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 0], y=(y_pred[:, 0] - y_true[:, 0]) / y_true[:, 0], bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_relative.png")
    plt.close()

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel("$E_{True}$ [MeV]")
    plt.ylabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
    plt.hist2d(x=y_true[:, 1], y=(y_pred[:, 1] - y_true[:, 1]) / y_true[:, 1], bins=[bins_energy, bins_err],
               norm=LogNorm())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_relative.png")
    plt.close()


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
    popt0, pcov0 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    popt1, pcov1 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist1, p0=[0.0, 20.0, np.sum(hist1) * width])
    popt2, pcov2 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist2, p0=[0.0, 1.0, np.sum(hist2) * width])
    popt3, pcov3 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist3, p0=[0.0, 1.0, np.sum(hist3) * width])
    popt4, pcov4 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist4, p0=[0.0, 20.0, np.sum(hist4) * width])
    popt5, pcov5 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist5, p0=[0.0, 1.0, np.sum(hist5) * width])

    ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
    ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
    ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-x resolution")
    plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Electron")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 0], y=y_pred[:, 0] - y_true[:, 0], bins=[bins_x[:209], bins_err_x], norm=LogNorm())
    plt.xlim(150.0 - 20.8 / 2.0, 150.0 + 20.8 / 2.0)
    plt.hlines(xmin=150.0 - 20.8 / 2.0, xmax=150.0 + 20.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt1), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt1[0], popt1[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Electron")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 1], y=y_pred[:, 1] - y_true[:, 1], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Electron position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt2), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt2[0], popt2[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_electron_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Electron")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 2], y=y_pred[:, 2] - y_true[:, 2], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
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
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_err_x, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt3), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt3[0], popt3[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x.png")
    plt.close()

    plt.figure()
    plt.title("Error position-x Photon")
    plt.xlabel("$e^{True}_{x}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
    plt.hist2d(x=y_true[:, 3], y=y_pred[:, 3] - y_true[:, 3], bins=[bins_x[467:], bins_err_x], norm=LogNorm())
    plt.xlim(270.0 - 46.8 / 2.0, 270.0 + 46.8 / 2.0)
    plt.hlines(xmin=270.0 - 46.8 / 2.0, xmax=270.0 + 46.8 / 2.0, y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_x_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-y resolution")
    plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_err_y, histtype=u"step", color="blue")
    plt.plot(ary_y, gaussian(ary_y, *popt4), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt4[0], popt4[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y.png")
    plt.close()

    plt.figure()
    plt.title("Error position-y Photon")
    plt.xlabel("$e^{True}_{y}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
    plt.hist2d(x=y_true[:, 4], y=y_pred[:, 4] - y_true[:, 4], bins=[bins_y, bins_err_y], norm=LogNorm())
    plt.hlines(xmin=min(bins_y), xmax=max(bins_y), y=0, color="red", linestyles="--")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_y_relative.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.title("Photon position-z resolution")
    plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_err_z, histtype=u"step", color="blue")
    plt.plot(ary_z, gaussian(ary_z, *popt5), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt5[0], popt5[1]))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + "_photon_z.png")
    plt.close()

    plt.figure()
    plt.title("Error position-z Photon")
    plt.xlabel("$e^{True}_{z}$ [mm]")
    plt.ylabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
    plt.hist2d(x=y_true[:, 5], y=y_pred[:, 5] - y_true[:, 5], bins=[bins_z, bins_err_z], norm=LogNorm())
    plt.hlines(xmin=min(bins_z), xmax=max(bins_z), y=0, color="red", linestyles="--")
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
    popt0, pcov0 = curve_fit(gaussian, bins_err_center, hist0, p0=[0.0, 1.0, np.sum(hist0) * width])
    ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

    plt.figure()
    plt.title("Error scattering angle")
    plt.xlabel(r"$\theta_{Pred}$ - $\theta_{True}$ [rad]")
    plt.ylabel("counts")
    plt.hist(y_pred - y_true, bins=bins_err, histtype=u"step", color="blue")
    plt.plot(ary_x, gaussian(ary_x, *popt0), color="orange",
             label=r"$\mu$ = {:.2f}""\n"r"$\sigma$={:.2f}".format(popt0[0], popt0[1]))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()

    plt.figure()
    plt.title("Error scattering angle")
    plt.xlabel(r"$\theta_{True}$ [rad]")
    plt.ylabel(r"$\theta_{Pred}$ - $\theta_{True}$")
    plt.hist2d(x=y_true, y=y_pred - y_true, bins=[bins_theta, bins_err], norm=LogNorm(vmin=1.0, vmax=800))
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + "_relative.png")
    plt.close()
