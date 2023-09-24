####################################################################################################
# Script Analysis phantom-hits advanced
#
# Script for analysing for the phenomenon of phantom compton events
# Phantom hits are described by having a valid compton event structure but the absorber
# interaction of a primary gamma is missing, instead a secondary fills the role with a valid
# position. In the advanced script only root files containing energy depositions of interactions
# can be used as these are scanned to determine pair-production processes
#
####################################################################################################

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})

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

# --------------------------------------------------------------------------------------------------
# Main loop over both root files
"""
# number of entries considered for analysis (either integer or None for all entries available)
n = 100000
root_parser = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_simv4)

if n is None:
    n_entries = root_parser.events_entries
else:
    n_entries = n


def get_pp_tag(event):
    n_e_miss = np.sum((event.MCEnergyDeps_p == 0.0) * 1)
    if n_e_miss > 0.0:
        # iterate over new interaction list
        for j in range(len(event.MCInteractions_p)):
            if int(str(event.MCInteractions_p[j])[2]) == 3:
                return True
    return False


# predefine all needed variables
# pp: Pair-production
# ph: Phantom-hits
n_pp = 0
n_pp_ph = 0
n_ph = 0

# main iteration over root file
for i, event in enumerate(root_parser.iterate_events(n=n)):
    tag = event.get_distcompton_tag()
    if event.b_phantom_hit:
        n_ph += 1

    if get_pp_tag(event):
        n_pp += 1

    if get_pp_tag(event) and event.b_phantom_hit:
        n_pp_ph += 1

print("LOADED " + root_parser.file_name)
print("Pair-production      : {:.1f} % total".format(n_pp / n * 100))
print("    - phantom hits   : {:.1f} % (of PP)".format(n_pp_ph / n_pp * 100))
print("    - phantom hits   : {:.1f} % (of PH total)".format(n_pp_ph / n_ph * 100))
"""
# ##################################################################################################
# Determining the correct acceptance threshold
"""
# load root files
root_parser = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_taggingv2)

def get_pp_tag(event):
    n_e_miss = np.sum((event.MCEnergyDeps_p == 0.0) * 1)
    if n_e_miss > 0.0:
        # iterate over new interaction list
        for j in range(len(event.MCInteractions_p)):
            if int(str(event.MCInteractions_p[j])[2]) == 3:
                return True
    return False


log_steps = np.linspace(0.1, 1, 10, endpoint=True)
steps = np.concatenate(
    [log_steps / 1e3, log_steps / 1e2, log_steps / 1e1, log_steps, log_steps * 10])
n = 100000

list_tp = []
list_fp = []
list_tn = []
list_fn = []  # bg = events tagged as phantom hit but no pair production

for theta in steps:
    n_tp = 0
    n_fp = 0
    n_tn = 0
    n_fn = 0

    for i, event in enumerate(root_parser.iterate_events(n=n)):
        ph_tag = event.get_phantomhit_tag(acceptance=theta)
        pp_tag = get_pp_tag(event)

        if pp_tag and ph_tag:
            n_tp += 1
        if not pp_tag and ph_tag:
            n_fp += 1
        if not pp_tag and not ph_tag:
            n_tn += 1
        if pp_tag and not ph_tag:
            n_fn += 1

    list_tp.append(n_tp)
    list_fp.append(n_fp)
    list_tn.append(n_tn)
    list_fn.append(n_fn)

ary_tp = np.array(list_tp)
ary_fp = np.array(list_fp)
ary_tn = np.array(list_tn)
ary_fn = np.array(list_fn)

# efficiency and purity plot
fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.set_xscale("log")
ax1.set_xlabel(r"$d_{acceptance} [mm]$")
ax1.set_ylabel("%")
ax1.tick_params(axis='y', labelcolor="red")
ax1.plot(steps, ary_tp / (ary_tp + ary_fn) * 100, "o", color="red", label="PP Efficiency")
ax1.plot(steps, ary_tp / (ary_tp + ary_fp) * 100, "x", color="red", label="PP Purity")
ax1.legend(loc="upper left")
ax1.grid(which="both")
ax2 = ax1.twinx()
ax2.set_ylabel("%")
ax2.plot(steps, (ary_tp + ary_fp) / n * 100, ".", color="blue", label="PH Efficiency")
# ax2.plot(steps, (ary_tn + ary_fn)/n*100, "s", color="blue", label="BG Efficiency")
ax2.tick_params(axis='y', labelcolor="blue")
ax2.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""
####################################################################################################
"""
# predefine all needed variables
# monte carlo truths
n_entries = 1000000
# load root files
root_parser = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_simV2)

mask_compton = np.zeros(shape=(n_entries,))
mask_ph = np.zeros(shape=(n_entries,))
mc_ee = np.zeros(shape=(n_entries,))
mc_ep = np.zeros(shape=(n_entries,))
mc_ex = np.zeros(shape=(n_entries,))
mc_ey = np.zeros(shape=(n_entries,))
mc_ez = np.zeros(shape=(n_entries,))
mc_px = np.zeros(shape=(n_entries,))
mc_py = np.zeros(shape=(n_entries,))
mc_pz = np.zeros(shape=(n_entries,))

# main iteration over root file. All needed quantities for the analysis are collected at once to
# save on iterations over the root file
for i, event in enumerate(root_parser.iterate_events(n=n_entries)):
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

    # collection from only distributed compton events (phantom hits included)
    tag = event.get_distcompton_tag()

    if tag:
        if not event.b_phantom_hit:
            mask_compton[i] = 1
        else:
            mask_ph[i] = 1

reco_ee = root_parser.events["RecoEnergy_e"]["value"].array()[:n_entries]
reco_ep = root_parser.events["RecoEnergy_p"]["value"].array()[:n_entries]
reco_ex = root_parser.events["RecoPosition_e"]["position"].array().x[:n_entries]
reco_ey = root_parser.events["RecoPosition_e"]["position"].array().y[:n_entries]
reco_ez = root_parser.events["RecoPosition_e"]["position"].array().z[:n_entries]
reco_px = root_parser.events["RecoPosition_p"]["position"].array().x[:n_entries]
reco_py = root_parser.events["RecoPosition_p"]["position"].array().y[:n_entries]
reco_pz = root_parser.events["RecoPosition_p"]["position"].array().z[:n_entries]
print("LOADED: Cut-Based reconstruction")


def gaussian(x, mu, sigma, A, c):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 / 2 * ((x - mu) / sigma) ** 2) + c


def lorentzian(x, mu, sigma, A, c):
    return A / np.pi * (1 / 2 * sigma) / ((x - mu) ** 2 + (1 / 2 * sigma) ** 2) + c


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
popt00, pcov00 = curve_fit(gaussian, bins_err_center, hist00,
                           p0=[0.0, 1.0, np.sum(hist00) * width, 0])
popt01, pcov01 = curve_fit(lorentzian, bins_err_center, hist01,
                           p0=[0.0, 0.5, np.sum(hist01) * width, 0])
popt10, pcov10 = curve_fit(gaussian, bins_err_center, hist10,
                           p0=[0.0, 1.0, np.sum(hist10) * width, 0])
popt11, pcov11 = curve_fit(lorentzian, bins_err_center, hist11,
                           p0=[0.0, 0.5, np.sum(hist11) * width, 0])
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
                           p0=[0.0, 1.0, np.sum(hist00) * width, 0])
popt01, pcov01 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist01,
                           p0=[0.0, 20.0, np.sum(hist01) * width, 0])
popt02, pcov02 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist02,
                           p0=[0.0, 1.0, np.sum(hist02) * width, 0])
popt03, pcov03 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist03,
                           p0=[0.0, 1.0, np.sum(hist03) * width, 0])
popt04, pcov04 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist04,
                           p0=[0.0, 20.0, np.sum(hist04) * width, 0])
popt05, pcov05 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist05,
                           p0=[0.0, 1.0, np.sum(hist05) * width, 0])
popt10, pcov10 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist10,
                           p0=[0.0, 1.0, np.sum(hist10) * width, 0])
popt11, pcov11 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist11,
                           p0=[0.0, 20.0, np.sum(hist11) * width, 0])
popt12, pcov12 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist12,
                           p0=[0.0, 1.0, np.sum(hist12) * width, 0])
popt13, pcov13 = curve_fit(gaussian, bins_err_x[:-1] + width / 2, hist13,
                           p0=[0.0, 1.0, np.sum(hist13) * width, 0])
popt14, pcov14 = curve_fit(gaussian, bins_err_y[:-1] + width / 2, hist14,
                           p0=[0.0, 20.0, np.sum(hist14) * width, 0])
popt15, pcov15 = curve_fit(gaussian, bins_err_z[:-1] + width / 2, hist15,
                           p0=[0.0, 1.0, np.sum(hist15) * width, 0])

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

####################################################################################################
# Control plots for distance d_a distribution

# number of entries considered for analysis (either integer or None for all entries available)
n = 10000
root_parser = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_simv4)

if n is None:
    n_entries = root_parser.events_entries
else:
    n_entries = n


def get_pp_tag(event):
    n_e_miss = np.sum((event.MCEnergyDeps_p == 0.0) * 1)
    if n_e_miss > 0.0:
        # iterate over new interaction list
        for j in range(len(event.MCInteractions_p)):
            if int(str(event.MCInteractions_p[j])[2]) == 3:
                return True
    return False


# predefine all needed variables
# pp: Pair-production
# ph: Phantom-hits
list_da_pp = []
list_da_bg = []

# main iteration over root file
for k, event in enumerate(root_parser.iterate_events(n=n)):
    tag_pp = get_pp_tag(event)

    # determine distance d_a
    # for pp events the first interaction after compton scattering is used to determine d_a
    # for background the interaction with the minimal distance is used for d_a
    if tag_pp:
        tmp_angle = vector_angle(event.MCPosition_p[2] - event.MCComptonPosition,
                                 event.MCDirection_scatter)
        r = (event.MCPosition_p[1] - event.MCComptonPosition).mag
        d_a = np.sin(tmp_angle) * r
        list_da_pp.append(d_a)
    else:
        temp_list_da = []
        for i in range(1, len(event.MCInteractions_p_uni)):
            # skip zero energy deposition interactions
            if event.mask_interactions_p[i] == 0:
                continue
            if (event.MCInteractions_p_uni[i, 1] <= 2 and
                    event.absorber.is_vec_in_module(event.MCPosition_p[i])):
                # check additionally if the interaction is in the scattering direction
                tmp_angle = vector_angle(event.MCPosition_p[i] - event.MCComptonPosition,
                                         event.MCDirection_scatter)
                r = (event.MCPosition_p[i] - event.MCComptonPosition).mag
                d_a = np.sin(tmp_angle) * r
                temp_list_da.append(d_a)
        if len(temp_list_da) > 0:
            list_da_bg.append(min(temp_list_da))

log_steps = np.linspace(0.1, 1, 10, endpoint=True)
steps = np.concatenate([log_steps / 1e3,
                        log_steps / 1e2,
                        log_steps / 1e1,
                        log_steps,
                        log_steps * 10,
                        log_steps * 100])

plt.figure(figsize=(10, 6))
plt.xlabel(r"$d_a in [mm]$")
plt.xscale("log")
plt.ylabel("Counts")
plt.hist(list_da_pp, bins=steps, histtype=u"step", color="blue", label="Pair-Production")
plt.hist(list_da_bg, bins=steps, histtype=u"step", color="grey", linestyle="--",
         label="Background")
plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()
