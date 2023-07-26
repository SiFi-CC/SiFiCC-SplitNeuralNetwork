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
plt.figure(figsize=(12, 8))
plt.xlabel("Beam z-axis [mm]")
plt.ylabel("Counts")
plt.errorbar(x=bin_center, y=ary_proj_fp_0mm, xerr=ary_x_err, yerr=ary_y_fp_0mm_err, fmt=".",
             color="red", label="BP0mm", zorder=0)
plt.errorbar(x=bin_center, y=ary_proj_fp_5mm, xerr=ary_x_err, yerr=ary_y_fp_5mm_err, fmt=".",
             color="blue", label="BP5mm", alpha=0.5, zorder=0)
plt.plot(bin_center, gaussian(bin_center, *popt0), color="red", linestyle="--",
         label=r"$\mu$ = {:.2f}""\n"r"$\sigma$ = {:.2f}".format(popt0[0], popt0[1]), zorder=5)
plt.plot(bin_center, ary_proj_fp_avr, color="limegreen", label="Average", zorder=10)
"""
plt.errorbar(x=bin_center, y=ary_proj_cont, xerr=ary_x_err, yerr=ary_y_cont_err, fmt=".",
             color="black", label="Cont.")
"""
plt.legend()
plt.grid()
plt.show()

# control plot
plt.figure(figsize=(12, 8))
plt.xlabel("Beam z-axis [mm]")
plt.ylabel("Counts")
plt.plot(bin_center, ary_proj_s4a6_0mm, ".",
         color="red", label="S4A6 BP0mm", zorder=0)
plt.plot(bin_center, ary_proj_s4a6_5mm, ".",
         color="blue", label="S4A6 BP5mm", zorder=0)
plt.plot(bin_center, ary_proj_fp_avr, ".",
         color="limegreen", label="Avg. FP bg.", zorder=0)
plt.legend()
plt.grid()
plt.show()

# subtraction plots
#ary_proj_s4a6_0mm /= (np.sum(ary_proj_s4a6_0mm) * width)
#ary_proj_s4a6_5mm /= (np.sum(ary_proj_s4a6_5mm) * width)
#ary_proj_fp_avr /= (np.sum(ary_proj_fp_avr) * width)
ary_proj_fp_avr = ary_proj_fp_avr # *1.4
plt.figure(figsize=(12, 8))
plt.xlabel("Beam z-axis [mm]")
plt.ylabel("Counts")
plt.plot(bin_center, ary_proj_s4a6_0mm - ary_proj_fp_avr, ".",
         color="red", label="S4A6 BP0mm - avg", zorder=0)
plt.plot(bin_center, ary_proj_s4a6_5mm - ary_proj_fp_avr, ".",
         color="blue", label="S4A6 BP5mm - avg", zorder=0)
plt.legend()
plt.grid()
plt.show()
