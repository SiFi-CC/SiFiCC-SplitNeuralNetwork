# header
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

matplotlib.rcParams.update({'font.size': 12})

# load MCTRUTH files (.npz format)
# define file paths
os.chdir(os.getcwd() + "/..")
dir_main = os.getcwd()

dir_data = dir_main + "/data/"
dir_plots = dir_main + "/plots/"

# filenames
filename1 = "OptimisedGeometry_BP0mm_2e10protons.npz"
filename2 = "OptimisedGeometry_BP5mm_4e9protons.npz"

# load initial npz file
npz_0mm = np.load(dir_data + filename1)
npz_5mm = np.load(dir_data + filename2)

ary_0mm_MC = npz_0mm["MC_TRUTH"]
ary_5mm_MC = npz_5mm["MC_TRUTH"]

########################################################################################################################

ary_primary_energy_0mm = ary_0mm_MC[:, 5]
ary_primary_energy_0mm = ary_primary_energy_0mm[np.argsort(ary_primary_energy_0mm)]

y_perc = np.arange(1.0, ary_primary_energy_0mm.shape[0] + 1, 1.0)
y_perc /= (ary_primary_energy_0mm.shape[0] + 1)

bins = np.arange(0.0, 22.0, 1.0)
weights = np.ones((ary_primary_energy_0mm.shape[0],)) / len(ary_primary_energy_0mm)
idx_90perc = int(len(ary_primary_energy_0mm) * 0.9)

plt.figure()
plt.xlabel("MC Primary Energy [MeV]")
plt.ylabel("% of events")
plt.xlim(0.0, 20.0)
plt.xticks(bins[::2])
plt.plot(ary_primary_energy_0mm, y_perc, color="black")
plt.hist(ary_primary_energy_0mm, bins=bins, histtype=u"step", color="blue", alpha=0.5, weights=weights)
plt.hlines(y=1.0, xmax=20.0, xmin=0.0, color="black", linestyles="--")
plt.vlines(ymax=1.0, ymin=0.0, x=ary_primary_energy_0mm[idx_90perc], color="red", linestyles="--",
           label="90% at {:.1f} MeV".format(ary_primary_energy_0mm[idx_90perc]))
plt.legend()
plt.show()

########################################################################################################################

e_cut = 3  # in MeV
print("\n### Energy cut at {} MeV".format(e_cut))
n_events = np.sum((ary_primary_energy_0mm > e_cut) * 1)
print("Events in sample: {} ({:.1f}%)".format(n_events, n_events / len(ary_primary_energy_0mm) * 100))

########################################################################################################################

plt.figure()
plt.hist(ary_0mm_MC[:, 5], bins=np.arange(0.0, 10.0, 0.1), histtype=u"step", color="black")
plt.show()
