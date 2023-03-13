import numpy as np
import math
import os
import matplotlib.pyplot as plt
from src import MLEMBackprojection

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

# ----------------------------------------------------------------------------------------------------------------------
# Base image back-projection of Monte-Carlo truth

npz_lookup_0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz")
npz_lookup_5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")

ary_mc_0mm = npz_lookup_0mm["MC_TRUTH"]
ary_mc_5mm = npz_lookup_5mm["MC_TRUTH"]

ary_mc_0mm = ary_mc_0mm[ary_mc_0mm[:, 0] == 1, :]
ary_mc_5mm = ary_mc_5mm[ary_mc_5mm[:, 0] == 1, :]

n = 100000
image_0mm = MLEMBackprojection.reconstruct_image(ary_mc_0mm[:n, 1],
                                                 ary_mc_0mm[:n, 2],
                                                 ary_mc_0mm[:n, 3],
                                                 ary_mc_0mm[:n, 4],
                                                 ary_mc_0mm[:n, 5],
                                                 ary_mc_0mm[:n, 6],
                                                 ary_mc_0mm[:n, 7],
                                                 ary_mc_0mm[:n, 8],
                                                 apply_filter=True)

MLEMBackprojection.plot_backprojection_image(image_0mm, "", dir_plots + "MLEM_backproj_S1AX_MC_truth_BP0MM")
