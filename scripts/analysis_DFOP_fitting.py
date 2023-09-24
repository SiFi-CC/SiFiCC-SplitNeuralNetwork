# ##################################################################################################
# Script Analysis DFOP Fitting
#
# This script analysis the phenomena of a constant background in the final image reconstruction
# if neural network event reconstruction was used.
#
# ##################################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import uproot
from scipy.optimize import curve_fit

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 14})

# interpolation: cubic smoothing spline method
from scipy import interpolate

# define root file path
path = os.getcwd() + "/dfop_fitting_files/"
"""
####################################################################################################
# Template fits

# Define root files used
# The root file should be in the same folder as the script
ROOTFILE_SG_0MM = "CC6_ECRNSiPM_4to1_simv4_0mm_4e9protons_SIGNAL.root"
ROOTFILE_SG_5MM = "CC6_ECRNSiPM_4to1_simv4_5mm_4e9protons_SIGNAL.root"
ROOTFILE_SG_m5MM = "CC6_ECRNSiPM_4to1_simv4_m5mm_4e9protons_SIGNAL.root"
ROOTFILE_SG_10MM = "CC6_ECRNSiPM_4to1_simv4_10mm_4e9protons_SIGNAL.root"


def get_proj_from_root(root_file):
    file = uproot.open(root_file)
    ary_img = file[b'image_iter5_smooth_2.00;1'].values
    ary_proj = np.sum(ary_img, axis=1)
    return ary_proj[1:-1]


# get projections for all files
ary_proj_sg_0mm = get_proj_from_root(path + ROOTFILE_SG_0MM)
ary_proj_sg_5mm = get_proj_from_root(path + ROOTFILE_SG_5MM)
ary_proj_sg_m5mm = get_proj_from_root(path + ROOTFILE_SG_m5MM)
ary_proj_sg_10mm = get_proj_from_root(path + ROOTFILE_SG_10MM)

# define bin center
bins = np.linspace(-150, 150, 300)
width = bins[1] - bins[0]
bin_center = bins[1:] - width

# Control plot for signal and background projections
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].set_xlabel("Position along beam axis [mm]")
axs[0].set_ylabel("Counts")
axs[0].set_xlim(-70, 70)
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].errorbar(bin_center, ary_proj_sg_m5mm, np.sqrt(ary_proj_sg_m5mm), fmt=".", color="blue",
                label=r"-5mm")
axs[0].errorbar(bin_center, ary_proj_sg_0mm, np.sqrt(ary_proj_sg_0mm), fmt=".", color="grey",
                label=r"0mm")
axs[0].errorbar(bin_center, ary_proj_sg_5mm, np.sqrt(ary_proj_sg_5mm), fmt=".", color="deeppink",
                label=r"+5mm")
axs[0].errorbar(bin_center, ary_proj_sg_10mm, np.sqrt(ary_proj_sg_10mm), fmt=".", color="limegreen",
                label=r"+10mm")
axs[0].legend(loc="lower left")
axs[0].grid(which='major', color='#CCCCCC', linewidth=0.8)
axs[0].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
axs[0].minorticks_on()
axs[1].set_xlabel("Position along beam axis [mm]")
axs[1].set_xlim(-70, 70)
axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[1].errorbar(bin_center, ary_proj_sg_m5mm / 0.835,
                np.sqrt(ary_proj_sg_m5mm) / 0.835, fmt=".", color="blue",
                label=r"-5mm, $4x10^9$ protons")
axs[1].errorbar(bin_center, ary_proj_sg_0mm / 1.0,
                np.sqrt(ary_proj_sg_0mm) / 1.0, fmt=".", color="grey",
                label=r"0mm, $4x10^9$ protons")
axs[1].errorbar(bin_center, ary_proj_sg_5mm / 1.020,
                np.sqrt(ary_proj_sg_5mm) / 1.020, fmt=".", color="deeppink",
                label=r"+5mm, $4x10^9$ protons")
axs[1].errorbar(bin_center, ary_proj_sg_10mm / 0.980,
                np.sqrt(ary_proj_sg_10mm) / 0.980, fmt=".", color="limegreen",
                label=r"+10mm, $4x10^9$ protons")
axs[1].grid(which='major', color='#CCCCCC', linewidth=0.8)
axs[1].grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
axs[1].minorticks_on()

plt.tight_layout()
plt.show()
"""
"""
####################################################################################################
# fit range scanning

def find_minimum(ary):
    min_i, min_j = 0, 0
    min_chi = 1e3
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            if ary[i, j] == 0:
                continue
            if ary[i, j] < min_chi:
                min_chi = ary[i, j]
                min_i = i
                min_j = j
    return min_chi, min_i, min_j


ary_fitrange = np.load(path + "fit_range_scan.npy")

# fixed ticks
xticks = np.linspace(0, 75, 4, endpoint=True) + 0.5
xlabels = ["0", "5", "10", "15"]
yticks = np.linspace(0, 75, 4, endpoint=True) + 0.5
ylabels = ["-5", "-0", "5", "10"]

# custom color map
BuPu = cm.get_cmap("viridis", 512)
newcolor = BuPu(np.linspace(0, 1, 512))
red = np.array([0, 0, 0, 0])
newcolor[:1, :] = red
newcmp = ListedColormap(newcolor)

# get minimum of array
min_chi, min_i, min_j = find_minimum(ary_fitrange[:-25, 25:])

# plot
fig, axs = plt.subplots(figsize=(10, 6))
axs.set_xlabel("upper limit [mm]")
axs.set_ylabel("lower limit [mm]")
axs.set_xticks(xticks, xlabels)
axs.set_yticks(yticks, ylabels)

heatmap = axs.pcolor(ary_fitrange[:-25, 25:], cmap=newcmp)
cbar = plt.colorbar(heatmap)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel(r"$\chi^2/ndof$", rotation=270)
axs.plot(min_j + 0.5, min_i + 0.5, "x", color="red",
         label="Fitting range;\n[{:.1f}, {:.1f}]".format(min_j*0.2-5, min_i*0.2))
plt.legend(loc="upper left")
plt.show()
"""

####################################################################################################
# calibration plot

y1 = np.array([[-0.062, -0.182, -0.272, -0.302],
               [3.838, 3.792, 3.900, 3.940],
               [7.190, 7.376, 7.558, 7.542],
               [-3.486, -3.444, -3.106, -3.172]])
y1_err = np.array([[1.867, 1.132, 0.743, 0.558],
                   [1.991, 1.435, 0.811, 0.520],
                   [1.755, 1.473, 0.798, 0.495],
                   [2.517, 1.465, 0.925, 0.497]])


def linear(x, m, b):
    return x * m + b


shifts = [0, 5, 10, -5]
popt, pconv = curve_fit(linear, shifts, y1[:, 3], sigma=y1_err[:, 3], absolute_sigma=True,
                        p0=[1, 0])
chi2 = 0
for i in range(len(y1[:, 3])):
    chi2 += (linear(shifts[i], *popt) - y1[i, 3]) ** 2 / y1_err[i, 3] * +2
chi2_ndof = chi2 / (len(y1) - 2)

x = np.linspace(-6, 11, 100)
plt.figure(figsize=(8, 6))
plt.xlabel("Simulated shift [mm]")
plt.ylabel("Reconstructed shift [mm]")
plt.errorbar([0, 5, 10, -5], y1[:, 3], y1_err[:, 3], fmt=".", color="black")
plt.plot(x, linear(x, 1, 0), color="grey", linestyle="--")
plt.plot(x, linear(x, *popt), color="red", linestyle="-",
         label=r"$\chi^2/ndof$ = {:.2f}".format(chi2_ndof) + "\n" +
               r"p0 = {:.2f} $\pm$ {:.2f}".format(popt[0], np.sqrt(pconv[0, 0])) +
               "\n" + "p1 = {:.2f} $\pm$ {:.2f}".format(popt[1], np.sqrt(pconv[1, 1])))
plt.legend(loc="upper left")
plt.grid(which='major', color='#CCCCCC', linewidth=0.8)
plt.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.8)
plt.minorticks_on()
plt.show()
