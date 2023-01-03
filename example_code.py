import numpy as np
import os
import matplotlib.pyplot as plt

import src.utilities
from src import RootParser
from src import root_files

########################################################################################################################

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"

########################################################################################################################

# Reading root files and export it to npz files
root1 = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)
# root1.export_npz(dir_npz + "OptimisedGeometry_BP0mm_2e10protons.npz")

root2 = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_offline)
# root2.export_npz(dir_npz + "OptimisedGeometry_BP5mm_4e9protons.npz")


ary_sp_pos = []
ary_sp_pos_s1ax = []
for i, event in enumerate(root1.iterate_events(n=1000000)):
    # get indices of clusters sorted by highest energy and module
    idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

    if event.is_ideal_compton:
        ary_sp_pos.append(event.MCPosition_source.z)

        if len(idx_scatterer) == 1:
            ary_sp_pos_s1ax.append(event.MCPosition_source.z)

bins = np.arange(-80, 20, 1.0)
width = abs(bins[0] - bins[1])

hist1, _ = np.histogram(ary_sp_pos, bins=bins)
hist2, _ = np.histogram(ary_sp_pos_s1ax, bins=bins)

# generate plots
# MC Total / MC Ideal Compton
plt.figure()
plt.title("MC Source Position z")
plt.xlabel("z-position [mm]")
plt.ylabel("counts (normalized)")
# total event histogram
plt.hist(ary_sp_pos, bins=bins, histtype=u"step", color="black", label="Total Ideal Compton", density=True, alpha=0.5,
         linestyle="--")
plt.hist(ary_sp_pos_s1ax, bins=bins, histtype=u"step", color="red", label="NN positives", density=True, alpha=0.5,
         linestyle="--")
plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
             np.sqrt(hist1) / np.sum(hist1) / width, color="black", fmt=".")
plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
             np.sqrt(hist2) / np.sum(hist2) / width, color="red", fmt=".")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
