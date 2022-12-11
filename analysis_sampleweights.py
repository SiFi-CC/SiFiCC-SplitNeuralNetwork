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

fig, axs = plt.subplots(2, 1)
axs[0].set_xlim(0.0, max_e + 1.0)
axs[0].hist(ary_mc[:, 0], bins=bins, histtype=u"step", color="black", label="MCEnergyPrimary")
axs[0].legend()
# axs[1].hlines(xmin=0.0, xmax=21.0, y=len(ary_mc[:, 0]) / (len(bins) - 1), color="blue", linestyles="--")
axs[1].set_xlim(0.0, max_e + 1.0)
axs[1].plot(bins[:-1] + 0.05, len(ary_mc[:, 0]) / (len(bins) - 1) / hist, color="blue", label="weights")
axs[1].legend()
axs[0].set_ylabel("Total counts")
axs[1].set_ylabel("weight")
axs[1].set_xlabel("MC Primary Energy [MeV]")
plt.show()

print("\nClass weights")
print(class_weights)
print("\nEnergy weights")
print(ary_meta.shape[0] / (len(bins) - 1) / hist)

ary_w = np.zeros(shape=(ary_meta.shape[0],))
ary_targets = ary_meta[:, 2]
for i in range(len(ary_w)):
    for j in range(len(bins) - 1):
        if bins[j] < ary_mc[i, 0] < bins[j + 1]:
            energy_weight = len(ary_targets) / (len(bins) - 1) / hist[j]
            ary_w[i] = energy_weight * class_weights[int(ary_targets[i])]
            break

for i in range(20):
    print(ary_mc[i, 0], ary_w[i])
