import numpy as np
import copy
import os
import matplotlib.pyplot as plt

from classes import RootParser
from classes import root_files

########################################################################################################################

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"

########################################################################################################################

npz_data = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons.npz")
ary_meta = npz_data["META"]
ary_mc = npz_data["MC_TRUTH"]

max_e = 17.0
_, counts = np.unique(ary_meta[:, 2], return_counts=True)
class_weights = [len(ary_meta[:, 2]) / (2 * counts[0]), len(ary_meta[:, 2]) / (2 * counts[1])]
# class_weights = [1 / counts[0], 1 / counts[1]]

# calculate energy weights
bins = np.concatenate([np.arange(0.0, max_e + 0.1, 0.1), [int(max(ary_mc[:, 0]))]])
hist, _ = np.histogram(ary_mc[:, 0], bins=bins)

# manual energy weights
bins2 = np.array([0.0, 2.5, 4.3, 4.5, 8.0, max_e + 2.0])
hist2 = np.array([0.1, 1.0, 5.0, 3.0, 5.0])

fig, axs = plt.subplots(3, 1)
axs[0].set_xlim(0.0, max_e + 1.0)
axs[0].set_xticks([])
axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
axs[0].set_ylabel("Total counts")
axs[1].set_ylabel("weight")
axs[1].set_xlim(0.0, max_e + 1.0)
axs[1].set_xticks([])
axs[2].set_xlim(0.0, max_e + 1.0)
axs[2].set_ylabel("weight")
axs[2].set_xlabel("MC Primary Energy [MeV]")
axs[0].hist(ary_mc[:, 0], bins=bins, histtype=u"step", color="black", label="MCEnergyPrimary")
axs[0].legend()
axs[1].plot(bins[:-1] + 0.05, len(ary_mc[:, 0]) / (len(bins) - 1) / hist, color="blue", label="bin weights")
axs[1].legend()
axs[2].stairs(hist2, bins2, color="red", label="pre-defined weights")
axs[2].legend()
plt.tight_layout()
plt.show()

