# ##################################################################################################
# Script Background Fit Analysis
#
# This script analysis the phenomena of a constant background in the final image reconstruction
# if neural network event reconstruction was used.
#
# ##################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import uproot
from scipy.optimize import curve_fit

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})

# interpolation: cubic smoothing spline method
from scipy import interpolate

# Define root files used
# The root file should be in the same folder as the script
ROOTFILE_FP_0MM = "FIT_NNRECO_FPONLY_DenseClusterS4A6_4e9protons_BP0mm_theta05.root"
ROOTFILE_FP_5MM = "FIT_NNRECO_FPONLY_DenseClusterS4A6_4e9protons_BP5mm_theta05.root"
ROOTFILE_FP_CONT = "FIT_NNRECO_FPONLY_DenseClusterS4A6_4e9protons_BP0mm_theta05.root"

ROOTFILE_S4A6_0MM = "FIT_NNRECO_DenseClusterS4A6_4e9protons_BP0mm_theta05.root"
ROOTFILE_S4A6_5MM = "FIT_NNRECO_DenseClusterS4A6_4e9protons_BP5mm_theta05.root"

# open root files with uproot
file_fp_0mm = uproot.open(ROOTFILE_FP_0MM)
file_fp_5mm = uproot.open(ROOTFILE_FP_5MM)
file_fp_cont = uproot.open(ROOTFILE_FP_CONT)
ary_img_fp_0mm = file_fp_0mm[b'Gaussmeared_image_iter5_shrinked_0;1'].values
ary_img_fp_5mm = file_fp_5mm[b'Gaussmeared_image_iter5_shrinked_0;1'].values
ary_img_fp_cont = file_fp_cont[b'Gaussmeared_image_iter5_shrinked_0;1'].values

file_s4a6_0mm = uproot.open(ROOTFILE_S4A6_0MM)
file_s4a6_5mm = uproot.open(ROOTFILE_S4A6_5MM)
ary_img_s4a6_0mm = file_s4a6_0mm[b'Gaussmeared_image_iter5_shrinked_0;1'].values
ary_img_s4a6_5mm = file_s4a6_5mm[b'Gaussmeared_image_iter5_shrinked_0;1'].values

# define bin center
bins = np.linspace(-70, 70, 142)
width = bins[1] - bins[0]
bin_center = bins[1:] - width

# prepare projections
ary_proj_fp_0mm = np.sum(ary_img_fp_0mm, axis=1)
ary_proj_fp_5mm = np.sum(ary_img_fp_5mm, axis=1)
ary_proj_fp_cont = np.sum(ary_img_fp_cont, axis=1)
ary_x_err = np.ones(shape=(len(ary_proj_fp_0mm))) * width / 2
ary_y_fp_0mm_err = np.sqrt(ary_proj_fp_0mm)
ary_y_fp_5mm_err = np.sqrt(ary_proj_fp_5mm)
ary_y_fp_cont_err = np.sqrt(ary_proj_fp_cont)
# average projection, used for subtraction later
ary_proj_fp_avr = (ary_proj_fp_0mm + ary_proj_fp_5mm) / 2

# signal projections
ary_proj_s4a6_0mm = np.sum(ary_img_s4a6_0mm, axis=1)
ary_proj_s4a6_5mm = np.sum(ary_img_s4a6_5mm, axis=1)


# define fitting functions
def gaussian(x, mu, sigma, A, c):
    return A / (sigma * np.sqrt(2 * np.pi)) * np.exp(
        -1 / 2 * ((x - mu) / sigma) ** 2) + c


# fitting of background
popt0, pcov0 = curve_fit(gaussian, xdata=bin_center, ydata=ary_proj_fp_0mm,
                         p0=[0.0, 1.0, np.sum(ary_proj_fp_0mm) * width, 400])

# Control plot
fig, axs = plt.subplots(1, 2, figsize=(14, 8))
axs[0].set_xlabel("Position along beam axis [mm]")
axs[0].set_ylabel("Counts")
axs[0].errorbar(x=bin_center, y=ary_proj_fp_0mm, xerr=ary_x_err, yerr=ary_y_fp_0mm_err, fmt=".",
                color="red", label="BP0mm", zorder=0)
axs[0].errorbar(x=bin_center, y=ary_proj_fp_5mm, xerr=ary_x_err, yerr=ary_y_fp_5mm_err, fmt=".",
                color="blue", label="BP5mm", zorder=0)
axs[0].legend()
axs[0].grid()
axs[1].set_xlabel("Position along beam axis [mm]")
axs[1].set_ylabel("Counts")
axs[1].errorbar(x=bin_center, y=ary_proj_fp_0mm, xerr=ary_x_err, yerr=ary_y_fp_0mm_err, fmt=".",
                color="red", zorder=0, alpha=0.1)
axs[1].errorbar(x=bin_center, y=ary_proj_fp_5mm, xerr=ary_x_err, yerr=ary_y_fp_5mm_err, fmt=".",
                color="blue", zorder=0, alpha=0.1)
axs[1].plot(bin_center, gaussian(bin_center, *popt0), color="red", linestyle="--",
            label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]), zorder=5)
axs[1].plot(bin_center, ary_proj_fp_avr, color="limegreen", label="Average", zorder=10)

axs[1].legend()
axs[1].grid()
"""
plt.errorbar(x=bin_center, y=ary_proj_cont, xerr=ary_x_err, yerr=ary_y_cont_err, fmt=".",
             color="black", label="Cont.")
"""
plt.show()

# interpolation of signal, FP curve and average curve
ary_x_new = np.arange(-70, 70, 0.2)
tck_s4a6_0mm = interpolate.splrep(bin_center, ary_proj_s4a6_0mm, s=0)
tck_s4a6_5mm = interpolate.splrep(bin_center, ary_proj_s4a6_5mm, s=0)
ary_proj_s4a6_0mm_int = interpolate.splev(ary_x_new, tck_s4a6_0mm, der=0)
ary_proj_s4a6_5mm_int = interpolate.splev(ary_x_new, tck_s4a6_5mm, der=0)

tck_fp_0mm = interpolate.splrep(bin_center, ary_proj_fp_0mm, s=0)
tck_fp_5mm = interpolate.splrep(bin_center, ary_proj_fp_5mm, s=0)
ary_proj_fp_0mm_int = interpolate.splev(ary_x_new, tck_fp_0mm, der=0)
ary_proj_fp_5mm_int = interpolate.splev(ary_x_new, tck_fp_5mm, der=0)
ary_proj_fp_avg_int = (ary_proj_fp_0mm_int + ary_proj_fp_5mm_int) / 2

# control plot with interpolation
plt.figure(figsize=(12, 8))
plt.xlabel("Position along beam axis [mm]")
plt.ylabel("Counts")
plt.plot(bin_center, ary_proj_s4a6_0mm, ".",
         color="red", label="S4A6 BP0mm", zorder=0)
plt.plot(bin_center, ary_proj_s4a6_5mm, ".",
         color="blue", label="S4A6 BP5mm", zorder=0)
plt.plot(bin_center, ary_proj_fp_avr, ".",
         color="limegreen", label="Avg. FP bg.", zorder=0)
plt.plot(ary_x_new, ary_proj_s4a6_0mm_int, linestyle="--", color="darkred",
         label="S4A6 BP0mm\nCubic spline")
plt.plot(ary_x_new, ary_proj_s4a6_5mm_int, linestyle="--", color="darkblue",
         label="S4A6 BP5mm\nCubic spline")
plt.plot(ary_x_new, ary_proj_fp_avg_int, linestyle="--", color="darkgreen",
         label="FP bg. avg.\nCubic spline")
plt.legend()
plt.grid()
plt.show()

# subtraction plots (plotted for interpolation)
ary_proj_fp_avg_int *= 1.38
plt.figure(figsize=(12, 8))
plt.xlabel("Position along beam axis [mm]")
plt.ylabel("Counts")
plt.plot(ary_x_new, ary_proj_s4a6_0mm_int - ary_proj_fp_avg_int, ".",
         color="red", label="S4A6 BP0mm - avg", zorder=0)
plt.plot(ary_x_new, ary_proj_s4a6_5mm_int - ary_proj_fp_avg_int, ".",
         color="blue", label="S4A6 BP5mm - avg", zorder=0)
plt.legend()
plt.grid()
plt.show()

# chi^2 fit for range shift
"""
def chi2(ary_y1, ary_y2):
    return
"""


def RSME(ary_y1, ary_y2, steps=30):
    list_rsme = []
    shift = []

    for k in range(1, steps):
        y1 = ary_y1[:-k]
        y2 = ary_y2[k:]
        rsme_ij = np.sqrt(np.sum([(i - j) ** 2 for i, j in zip(y1, y2)]) / len(y1))
        list_rsme.append(rsme_ij)
        shift.append(k * 0.2)

    for k in range(1, steps):
        y1 = ary_y2[:-k]
        y2 = ary_y1[k:]
        rsme_ij = np.sqrt(np.sum([(i - j) ** 2 for i, j in zip(y1, y2)]) / len(y1))
        list_rsme.append(rsme_ij)
        shift.append(-k * 0.2)

    return list_rsme, shift


rsme, shift = RSME(ary_proj_s4a6_0mm_int[300:400] - ary_proj_fp_avg_int[300:400],
                   ary_proj_s4a6_5mm_int[300:400] - ary_proj_fp_avg_int[300:400])
plt.figure()
plt.plot(shift, rsme, ".")
plt.show()

# subtraction plots (plotted for interpolation)
plt.figure(figsize=(12, 8))
plt.xlabel("Position along beam axis [mm]")
plt.ylabel("Counts")
plt.plot(ary_x_new, ary_proj_s4a6_0mm_int - ary_proj_fp_avg_int, ".",
         color="red", label="S4A6 BP0mm - avg", zorder=0)
plt.plot(ary_x_new, ary_proj_s4a6_5mm_int - ary_proj_fp_avg_int, ".",
         color="blue", label="S4A6 BP5mm - avg", zorder=0)
for k in range(1, 30):
    y1 = ary_proj_s4a6_0mm_int[:-k] - ary_proj_fp_avg_int[:-k]
    # y2 = ary_proj_s4a6_0mm_int[k:] - ary_proj_fp_avg_int[k:]
    plt.plot(ary_x_new[k:], y1, ".", color="black")
plt.legend()
plt.grid()
plt.show()
