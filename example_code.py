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

npz_data = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_DNN_Base.npz")
features = npz_data["features"]

n_cluster = 8
n_features = 9
list_mean = []
list_std = []
list_idx = np.arange(0, n_cluster * n_features, n_features)
for i in range(9):
    # print(np.array(list_idx) + i)
    ary_con = np.reshape(features[:, np.array(list_idx) + i], (features.shape[0]*n_cluster,))
    list_mean.append(np.mean(ary_con))
    list_std.append(np.std(ary_con))
    print(list_mean[i], list_std[i])

list_mean = list_mean * n_cluster
list_std = list_std * n_cluster

for i in range(len(list_mean)):
    mean = list_mean[i]
    std = list_std[i]
    features[:, i] -= mean
    features[:, i] /= std

x_feat = features[6, :]
x_feat = np.reshape(x_feat, (8, 9))
print(x_feat)
