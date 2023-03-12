import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({'font.size': 16})

from src import CBSelector

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"

# ----------------------------------------------------------------------------------------------------------------------
# Data read-in

npz_nn_0mm = np.load(
    dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz")
npz_nn_5mm = np.load(
    dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz")

ary_nn_0mm = npz_nn_0mm["NN_PRED"]
ary_nn_5mm = npz_nn_5mm["NN_PRED"]

npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")
ary_meta_0mm = npz_lookup_0mm["META"]
ary_meta_5mm = npz_lookup_5mm["META"]
ary_mc_0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_5mm = npz_lookup_5mm["MC_TRUTH"]
ary_cb_0mm = npz_lookup_0mm["CB_RECO"]
ary_cb_5mm = npz_lookup_5mm["CB_RECO"]

# ----------------------------------------------------------------------------------------------------------------------
# Analysis script

list_dac_nn_0mm_ic = []
list_dac_cb_0mm_ic = []

list_dac_nn_0mm_bg = []
list_dac_cb_0mm_bg = []

for i in range(ary_nn_0mm.shape[0]):
    if ary_nn_0mm[i, 0] < 0.5:
        continue

    dac_nn = CBSelector.beam_origin(*ary_nn_0mm[i, 1:9], beam_diff=20, inverse=False, return_dac=True)
    dac_cb = CBSelector.beam_origin(*ary_cb_0mm[i, 1:9], beam_diff=20, inverse=False, return_dac=True)

    if ary_mc_0mm[i, 0] == 1.0:
        list_dac_nn_0mm_ic.append(dac_nn)
        list_dac_cb_0mm_ic.append(dac_cb)
    else:
        list_dac_nn_0mm_bg.append(dac_nn)
        list_dac_cb_0mm_bg.append(dac_cb)

n_cb_ic = np.sum((np.array(list_dac_cb_0mm_ic) < 20)*1)
n_cb_bg = np.sum((np.array(list_dac_cb_0mm_bg) < 20)*1)

n_nn_ic = np.sum((np.array(list_dac_nn_0mm_ic) < 20)*1)
n_nn_bg = np.sum((np.array(list_dac_nn_0mm_bg) < 20)*1)

print("### DCA Statistic:   ")
print("Cut-Based below DCA 20mm: ")
print("Total: ", n_cb_ic + n_cb_bg)
print("IdealCompton: ", n_cb_ic)
print("Background: ", n_cb_bg)
print("Neural Network below DCA 20mm: ")
print("Total: ", n_nn_ic + n_nn_bg)
print("IdealCompton: ", n_nn_ic)
print("Background: ", n_nn_bg)

"""
# scatter plot
plt.figure()
plt.xlabel("DCA NN [mm]")
plt.ylabel("DCA CB [mm]")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.scatter(list_dac_nn_0mm_bg, list_dac_cb_0mm_bg, color="blue", label="Background")
plt.scatter(list_dac_nn_0mm_ic, list_dac_cb_0mm_ic, color="orange", label="Ideal Compton")
plt.plot([0.0, 100.0], [0.0, 100.0], color="red", linestyle="--")
plt.legend()
plt.tight_layout()
plt.show()
"""
bins = np.arange(0.0, 100.0, 1.0)
plt.figure()
plt.xlabel("DCA [mm]")
plt.ylabel("counts")
plt.hist(list_dac_nn_0mm_ic, bins=bins, histtype=u"step", color="blue", linestyle="--", label="NN IC")
plt.hist(list_dac_nn_0mm_bg, bins=bins, histtype=u"step", color="blue", label="NN BG")
plt.hist(list_dac_cb_0mm_ic, bins=bins, histtype=u"step", color="black", linestyle="--", label="CB IC")
plt.hist(list_dac_cb_0mm_bg, bins=bins, histtype=u"step", color="black", label="CB BG")
plt.yscale('log')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("DCA_analysis.png")
