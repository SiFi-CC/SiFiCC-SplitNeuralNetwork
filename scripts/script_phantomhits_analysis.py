####################################################################################################
# Script missing primary analysis
#
# Script for analysing for the phenomenon of phantom compton events
# Phantom hits are described by having a valid compton event structure but the absorber
# interaction of a primary gamma is missing, instead a secondary fills the role with a valid
# position
#
####################################################################################################

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 12})

from SiFiCCNN.root import RootParser, RootFiles
from SiFiCCNN.utils.physics import vector_angle

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    path = os.path.abspath(os.path.join(path, os.pardir))
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
path_main = path
path_root = path + "/root_files/"

# load root files
# As a comparison the old BP0mm with taggingv1 will be loaded as well
root_parser_old = RootParser.RootParser(path_root + RootFiles.onetoone_BP5mm_taggingv2)
root_parser_new = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_taggingv2)

# --------------------------------------------------------------------------------------------------
# Main loop over both root files

# number of entries considered for analysis (either integer or None for all entries available)
# n1 <=> old tagging file, n2 <=> new tagging file
n = 100000

if n is None:
    n1 = root_parser_old.events_entries
    n2 = root_parser_new.events_entries
else:
    n1 = n
    n2 = n

# predefine all needed variables
# monte carlo truths
mask_compton = np.zeros(shape=(n1,))
mask_ph = np.zeros(shape=(n1,))
mc_ee = np.zeros(shape=(n1,))
mc_ep = np.zeros(shape=(n1,))
mc_ex = np.zeros(shape=(n1,))
mc_ey = np.zeros(shape=(n1,))
mc_ez = np.zeros(shape=(n1,))
mc_px = np.zeros(shape=(n1,))
mc_py = np.zeros(shape=(n1,))
mc_pz = np.zeros(shape=(n1,))

# counting statistics
n1_compton = 0
n1_compton_ph = 0

# distance measurements cluster/interaction
# distance of the nearest cluster/interaction to scattered gamma track
list_clustdist = []
list_clustdist_ph = []
list_clustdist_all = []
list_intdist = []
list_intdist_ph = []
list_intdist_all = []

list1_tdot = []
list1_ph_tdot = []

list1_tdot_diff = []
list1_tdot_ph_diff = []

list1_pe = []
list1_ph_pe = []

list1_ep = []
list1_eabs = []
list1_ph_ep = []
list1_ph_eabs = []

# main iteration over root file. All needed quantities for the analysis are collected at once to
# save on iterations over the root file
for i, event in enumerate(root_parser_old.iterate_events(n=n1)):
    # grab MC-Truth values and collect them
    target_energy_e, target_energy_p = event.get_target_energy()
    target_position_e, target_position_p = event.get_target_position()
    mc_ee[i] = target_energy_e
    mc_ep[i] = target_energy_p
    mc_ex[i] = target_position_e.x
    mc_ey[i] = target_position_e.y
    mc_ez[i] = target_position_e.z
    mc_px[i] = target_position_p.x
    mc_py[i] = target_position_p.y
    mc_pz[i] = target_position_p.z

    # at this point skip events that are meaningless for phantom hit analysis
    if not len(event.MCPosition_p) > 1:
        continue

    # collect quantities from phantom hits and distributed compton events
    idx_scat, idx_abs = event.sort_clusters_by_module()
    tmp_list_dist = []
    for j in idx_abs:
        tmp_vector_cluster = event.RecoClusterPosition[j] - event.MCComptonPosition
        tmp_angle_cluster = vector_angle(event.MCDirection_scatter,
                                         tmp_vector_cluster)
        dist_cluster = np.sin(tmp_angle_cluster) * tmp_vector_cluster.mag
        tmp_list_dist.append(dist_cluster)
    min_dist_cluster = min(tmp_list_dist)  # distance of the closest cluster to scattering direction

    tmp_list_dist = []
    for j in range(1, len(event.MCPosition_p)):
        tmp_vector_int = event.MCPosition_p[j] - event.MCComptonPosition
        tmp_angle_int = vector_angle(event.MCDirection_scatter,
                                     tmp_vector_int)
        dist_int = np.sin(tmp_angle_int) * tmp_vector_int.mag
        tmp_list_dist.append(dist_int)
    min_dist_int = min(tmp_list_dist)  # distance of the closest cluster to scattering direction

    # for all events
    list_clustdist_all.append(min_dist_cluster)
    list_intdist_all.append(min_dist_int)

    # collection from only distributed compton events (phantom hits included)
    if event.get_distcompton_tag():
        n1_compton += 1

        if not event.b_phantom_hit:
            mask_compton[i] = 1
            list1_tdot.append(event.scatter_angle_dotvec)
            list1_pe.append(event.MCEnergy_Primary)
            list1_ep.append(event.MCEnergy_p)
            list1_eabs.append(np.sum(event.RecoClusterEnergies_values[idx_abs]))

            list_clustdist.append(min_dist_cluster)
            list_intdist.append(min_dist_int)

        else:
            mask_ph[i] = 1
            n1_compton_ph += 1
            list1_ph_tdot.append(event.scatter_angle_dotvec)
            list1_ph_pe.append(event.MCEnergy_Primary)

            list1_ph_ep.append(event.MCEnergy_p)
            list1_ph_eabs.append(np.sum(event.RecoClusterEnergies_values[idx_abs]))

            list_clustdist_ph.append(min_dist_cluster)
            list_intdist_ph.append(min_dist_int)

# --------------------------------------------------------------------------------------------------
# Analysis output
# print general statistics of root file for control
print("LOADED " + root_parser_old.file_name)
print("Phantom hits: {:.1f} % total | {:.1f} % Compton".format(n1_compton_ph / n1 * 100,
                                                               n1_compton_ph / n1_compton * 100))

# Plot histogram scattering angle (DotVec)
plt.figure(figsize=(12, 6))
plt.xlabel(r"Compton scattering angle $\theta_{DotVec}$ [rad]")
plt.ylabel("Counts (Normalized)")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
bins = np.linspace(0, np.pi, 100)
plt.hist(list1_tdot, bins=bins, histtype=u"step", linestyle="-", color="black", density=True,
         label="Distributed Compton", linewidth=1.5)
plt.hist(list1_ph_tdot, bins=bins, histtype=u"step", linestyle="--", color="blue", density=True,
         label="Phantom hits", linewidth=1.5)
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.show()

# plot distance of the closest cluster to scatter trajectory
plt.figure(figsize=(12,6))
plt.xlabel("Distance closest cluster to trajectory [mm]")
plt.ylabel("Counts normalized(a.u.)")
# plt.xscale("log")
plt.yscale("log")
# bins = np.concatenate([[0.0], np.logspace(-8, -1, 60, endpoint=True)])
bins = np.arange(0.0, 100.0, 0.1)
plt.hist(list_clustdist, bins=bins, histtype=u"step", linestyle="--", linewidth=1.5,
         color="blue", label="Distributed Compton")
plt.hist(list_clustdist_all, bins=bins, histtype=u"step", linestyle="-", linewidth=1.5,
         color="black", label="All")
plt.hist(list_clustdist_ph, bins=bins, histtype=u"step", linestyle="--", linewidth=1.5,
         color="green", label="Phantom hits")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# plot distance of the closest interaction to scatter trajectory
plt.figure(figsize=(12,6))
plt.xlabel("Distance closest interaction to trajectory [mm]")
plt.ylabel("Counts normalized(a.u.)")
plt.xscale("log")
plt.yscale("log")
bins = np.concatenate([[0.0], np.logspace(-11, 1, 120, endpoint=True)])
# bins = np.arange(0.0, 10.0, 0.1)
plt.hist(list_intdist, bins=bins, histtype=u"step", linestyle="--", linewidth=1.5,
         color="blue", label="Distributed Compton")
plt.hist(list_intdist_all, bins=bins, histtype=u"step", linestyle="-", linewidth=1.5,
         color="black", label="All")
plt.hist(list_intdist_ph, bins=bins, histtype=u"step", linestyle="--", linewidth=1.5,
         color="green", label="Phantom hits")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


"""
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
bins = np.arange(0.0, 10.0, 0.1)
axs[0].set_xlabel(r"$E_{Absorber}^{Reco}$ [Mev]")
axs[0].set_ylabel(r"$E_{\gamma}$ [Mev]")
axs[0].set_title("Distributed Compton")
axs[1].set_xlabel(r"$E_{Absorber}^{Reco}$ [Mev]")
axs[1].set_ylabel(r"$E_{\gamma}$ [Mev]")
axs[1].set_title("Phantom hits")
h0 = axs[0].hist2d(list1_eabs, list1_ep, bins=bins, norm=LogNorm())
h1 = axs[1].hist2d(list1_ph_eabs, list1_ph_ep, bins=bins, norm=LogNorm())
axs[0].plot([0, 10], [0, 10], linestyle="--", color="red", alpha=0.5)
axs[1].plot([0, 10], [0, 10], linestyle="--", color="red", alpha=0.5)
axs[0].plot([0, 10 - 0.511], [0 + 0.511, 10], linestyle="--", color="red", alpha=0.5)
axs[1].plot([0, 10 - 0.511], [0 + 0.511, 10], linestyle="--", color="red", alpha=0.5)
axs[0].plot([0, 10 - 2 * 0.511], [0 + 2 * 0.511, 10], linestyle="--", color="red", alpha=0.5)
axs[1].plot([0, 10 - 2 * 0.511], [0 + 2 * 0.511, 10], linestyle="--", color="red", alpha=0.5)
fig.colorbar(h0[3], ax=axs[0])
fig.colorbar(h1[3], ax=axs[1])
plt.show()

# Plot histogram primary energy
plt.figure(figsize=(12, 6))
plt.xlabel("Primary Energy [MeV]")
plt.ylabel("Counts (Normalized)")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
bins = np.arange(0.0, 10.0, 0.1)
plt.hist(list1_pe, bins=bins, histtype=u"step", linestyle="-", color="black", density=True,
         label="Distributed Compton", linewidth=1.5)
plt.hist(list1_ph_pe, bins=bins, histtype=u"step", linestyle="--", color="blue", density=True,
         label="Phantom hits", linewidth=1.5)
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Plot identification angle distribution
plt.figure(figsize=(12, 6))
plt.xlabel(r"$\theta_{control}$ [rad]")
plt.ylabel("Counts")
plt.xscale("log")
plt.yscale("log")
bins = np.concatenate([[0.0], np.logspace(-8, -2, 60, endpoint=True)])
plt.hist(list1_tdot_diff, bins=bins, histtype=u"step", linestyle="-", linewidth=1.5,
         color="black", label="Distributed Compton")
plt.hist(list1_tdot_ph_diff, bins=bins, histtype=u"step", linestyle="--", linewidth=1.5,
         color="blue", label="Phantom hits")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
"""
"""
# --------------------------------------------------------------------------------------------------
# comparison in cut-based reco

# fill up Cut-Based reconstruction values manually due to
# them being stored in branches

reco_ee = root_parser_old.events["RecoEnergy_e"]["value"].array()[:n1]
reco_ep = root_parser_old.events["RecoEnergy_p"]["value"].array()[:n1]
reco_ex = root_parser_old.events["RecoPosition_e"]["position"].array().x[:n1]
reco_ey = root_parser_old.events["RecoPosition_e"]["position"].array().y[:n1]
reco_ez = root_parser_old.events["RecoPosition_e"]["position"].array().z[:n1]
reco_px = root_parser_old.events["RecoPosition_p"]["position"].array().x[:n1]
reco_py = root_parser_old.events["RecoPosition_p"]["position"].array().y[:n1]
reco_pz = root_parser_old.events["RecoPosition_p"]["position"].array().z[:n1]
print("LOADED: Cut-Based reconstruction")


def gaussian(x, mu, sigma, A):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)


def lorentzian(x, mu, sigma, A):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2)


mask_compton = mask_compton == 1
mask_ph = mask_ph == 1
width = 0.01
bins_err = np.arange(-1.0 + width, 1.0 - width, width)
bins_err_center = bins_err[:-1] + (width / 2)

hist00, _ = np.histogram((reco_ee[mask_compton] - mc_ee[mask_compton]) / mc_ee[mask_compton],
                         bins=bins_err)
hist01, _ = np.histogram((reco_ep[mask_compton] - mc_ep[mask_compton]) / mc_ep[mask_compton],
                         bins=bins_err)
hist10, _ = np.histogram((reco_ee[mask_ph] - mc_ee[mask_ph]) / mc_ee[mask_ph],
                         bins=bins_err)
hist11, _ = np.histogram((reco_ep[mask_ph] - mc_ep[mask_ph]) / mc_ep[mask_ph],
                         bins=bins_err)

# fitting energy resolution
popt00, pcov00 = curve_fit(gaussian, bins_err_center, hist00, p0=[0.0, 1.0, np.sum(hist00) * width])
popt01, pcov01 = curve_fit(lorentzian, bins_err_center, hist01,
                           p0=[0.0, 0.5, np.sum(hist01) * width])
popt10, pcov10 = curve_fit(gaussian, bins_err_center, hist10, p0=[0.0, 1.0, np.sum(hist10) * width])
popt11, pcov11 = curve_fit(lorentzian, bins_err_center, hist11,
                           p0=[0.0, 0.5, np.sum(hist11) * width])
ary_x = np.linspace(min(bins_err), max(bins_err), 1000)

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Electron energy resolution")
plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
plt.ylabel("counts")
plt.hist((reco_ee[mask_compton] - mc_ee[mask_compton]) / mc_ee[mask_compton], bins=bins_err,
         histtype=u"step", color="red", label="Distributed Compton")
plt.plot(ary_x, gaussian(ary_x, *popt00), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt00[0], popt00[1]))
plt.hist((reco_ee[mask_ph] - mc_ee[mask_ph]) / mc_ee[mask_ph], bins=bins_err,
         histtype=u"step",
         color="black",
         label="Phantom hits", alpha=0.5)
plt.plot(ary_x, gaussian(ary_x, *popt10), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt10[0], popt10[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Photon energy resolution")
plt.xlabel(r"$\frac{E_{Pred} - E_{True}}{E_{True}}$")
plt.ylabel("counts")
plt.hist((reco_ep[mask_compton] - mc_ep[mask_compton]) / mc_ep[mask_compton], bins=bins_err,
         histtype=u"step", color="red", label="Distributed Compton")
plt.plot(ary_x, lorentzian(ary_x, *popt01), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt01[0], popt01[1] / 2))
plt.hist((reco_ep[mask_ph] - mc_ep[mask_ph]) / mc_ep[mask_ph], bins=bins_err,
         histtype=u"step", color="black", label="Phantom hits", alpha=0.5)
plt.plot(ary_x, lorentzian(ary_x, *popt11), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$FWHM$ = {:.2f}".format(popt11[0], popt11[1] / 2))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

width = 0.1
bins_err_x = np.arange(-5.5, 5.5, width)
bins_err_y = np.arange(-60.5, 60.5, width)
bins_err_z = np.arange(-5.5, 5.5, width)

bins_x = np.arange(150.0 - 20.8 / 2.0, 270.0 + 46.8 / 2.0, width)
bins_y = np.arange(-100.0 / 2.0, 100.0 / 2.0, width)
bins_z = np.arange(-98.8 / 2.0, 98.8 / 2.0, width)

hist00, _ = np.histogram(reco_ex[mask_compton] - mc_ex[mask_compton], bins=bins_err_x)
hist01, _ = np.histogram(reco_ey[mask_compton] - mc_ey[mask_compton], bins=bins_err_y)
hist02, _ = np.histogram(reco_ez[mask_compton] - mc_ez[mask_compton], bins=bins_err_z)
hist03, _ = np.histogram(reco_px[mask_compton] - mc_px[mask_compton], bins=bins_err_x)
hist04, _ = np.histogram(reco_py[mask_compton] - mc_py[mask_compton], bins=bins_err_y)
hist05, _ = np.histogram(reco_pz[mask_compton] - mc_pz[mask_compton], bins=bins_err_z)
hist10, _ = np.histogram(reco_ex[mask_ph] - mc_ex[mask_ph], bins=bins_err_x)
hist11, _ = np.histogram(reco_ey[mask_ph] - mc_ey[mask_ph], bins=bins_err_y)
hist12, _ = np.histogram(reco_ez[mask_ph] - mc_ez[mask_ph], bins=bins_err_z)
hist13, _ = np.histogram(reco_px[mask_ph] - mc_px[mask_ph], bins=bins_err_x)
hist14, _ = np.histogram(reco_py[mask_ph] - mc_py[mask_ph], bins=bins_err_y)
hist15, _ = np.histogram(reco_pz[mask_ph] - mc_pz[mask_ph], bins=bins_err_z)

# fitting position resolution
popt00, pcov00 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist00,
                           p0=[0.0, 1.0, np.sum(hist00) * width])
popt01, pcov01 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist01,
                           p0=[0.0, 20.0, np.sum(hist01) * width])
popt02, pcov02 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist02,
                           p0=[0.0, 1.0, np.sum(hist02) * width])
popt03, pcov03 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist03,
                           p0=[0.0, 1.0, np.sum(hist03) * width])
popt04, pcov04 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist04,
                           p0=[0.0, 20.0, np.sum(hist04) * width])
popt05, pcov05 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist05,
                           p0=[0.0, 1.0, np.sum(hist05) * width])
popt10, pcov10 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist10,
                           p0=[0.0, 1.0, np.sum(hist10) * width])
popt11, pcov11 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist11,
                           p0=[0.0, 20.0, np.sum(hist11) * width])
popt12, pcov12 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist12,
                           p0=[0.0, 1.0, np.sum(hist12) * width])
popt13, pcov13 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist13,
                           p0=[0.0, 1.0, np.sum(hist13) * width])
popt14, pcov14 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist14,
                           p0=[0.0, 20.0, np.sum(hist14) * width])
popt15, pcov15 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist15,
                           p0=[0.0, 1.0, np.sum(hist15) * width])

ary_x = np.linspace(min(bins_err_x), max(bins_err_x), 1000)
ary_y = np.linspace(min(bins_err_y), max(bins_err_y), 1000)
ary_z = np.linspace(min(bins_err_z), max(bins_err_z), 1000)

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Electron position-x resolution")
plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_ex[mask_compton] - mc_ex[mask_compton], bins=bins_err_x, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_x, gaussian(ary_x, *popt00), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt00[0], popt00[1]))
plt.hist(reco_ex[mask_ph] - mc_ex[mask_ph], bins=bins_err_x, histtype=u"step", color="black",
         alpha=0.5,
         label="Phantom hits")
plt.plot(ary_x, gaussian(ary_x, *popt10), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt10[0], popt10[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Electron position-y resolution")
plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_ey[mask_compton] - mc_ey[mask_compton], bins=bins_err_y, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_y, gaussian(ary_y, *popt01), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt01[0], popt01[1]))
plt.hist(reco_ey[mask_ph] - mc_ey[mask_ph], bins=bins_err_y, histtype=u"step", color="black",
         alpha=0.5,
         label="Phantom hits")
plt.plot(ary_y, gaussian(ary_y, *popt11), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt11[0], popt11[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Electron position-z resolution")
plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_ez[mask_compton] - mc_ez[mask_compton], bins=bins_err_z, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_z, gaussian(ary_z, *popt02), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt02[0], popt02[1]))
plt.hist(reco_ez[mask_ph] - mc_ez[mask_ph], bins=bins_err_z, histtype=u"step", color="black",
         label="Phantom hits", alpha=0.5)
plt.plot(ary_z, gaussian(ary_z, *popt12), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt12[0], popt12[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Photon position-x resolution")
plt.xlabel(r"$e^{Pred}_{x}$ - $e^{True}_{x}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_px[mask_compton] - mc_px[mask_compton], bins=bins_err_x, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_x, gaussian(ary_x, *popt03), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt03[0], popt03[1]))
plt.hist(reco_px[mask_ph] - mc_px[mask_ph], bins=bins_err_x, histtype=u"step", color="black",
         alpha=0.5, label="Phantom hits")
plt.plot(ary_x, gaussian(ary_x, *popt13), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt13[0], popt13[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Photon position-y resolution")
plt.xlabel(r"$e^{Pred}_{y}$ - $e^{True}_{y}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_py[mask_compton] - mc_py[mask_compton], bins=bins_err_y, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_y, gaussian(ary_y, *popt04), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt04[0], popt04[1]))
plt.hist(reco_py[mask_ph] - mc_py[mask_ph], bins=bins_err_y, histtype=u"step", color="black",
         alpha=0.5,
         label="Phantom hits")
plt.plot(ary_y, gaussian(ary_y, *popt14), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt14[0], popt14[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.title("Photon position-z resolution")
plt.xlabel(r"$e^{Pred}_{z}$ - $e^{True}_{z}$ [mm]")
plt.ylabel("counts")
plt.hist(reco_pz[mask_compton] - mc_pz[mask_compton], bins=bins_err_z, histtype=u"step",
         color="red", label="Distributed Compton")
plt.plot(ary_z, gaussian(ary_z, *popt05), color="red",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt05[0], popt05[1]))
plt.hist(reco_pz[mask_ph] - mc_pz[mask_ph], bins=bins_err_z, histtype=u"step", color="black",
         alpha=0.5, label="Phantom hits")
plt.plot(ary_z, gaussian(ary_z, *popt15), color="black", alpha=0.5,
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt15[0], popt15[1]))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
"""
