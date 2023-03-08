import numpy as np
import os

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"


def change_padding(filename, new_padding, tag, n_cluster=6):
    # open training file
    npz_data = np.load(filename)
    ary_features = npz_data["features"]
    ary_theta = npz_data["theta"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_w = npz_data["weights"]
    ary_meta = npz_data["META"]

    list_padding_old = np.array([0.0, -1.0, -1.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0, 0.0] * n_cluster)
    list_padding_new = np.array(new_padding * n_cluster)

    for i in range(ary_features.shape[1]):
        ary_idx = ary_features[:, i] == list_padding_old[i]
        ary_features[ary_idx, i] = list_padding_new[i]

    with open(filename[:-4] + "_" + tag + ".npz", 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            theta=ary_theta,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            weights=ary_w,
                            META=ary_meta)
    print("Padding changed in: " + filename)
    print("New file saved as: " + filename[:-4] + "_" + tag + ".npz")


# ----------------------------------------------------------------------------------------------------------------------
# Main usage

f1 = "OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz"
f2 = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz"
f3 = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz"

change_padding(dir_npz + f1, [0.0, -10.0, -10.0,  0.0, -100.0, -100.0,  0.0,  0.0,  0.0,  0.0], "anp")
change_padding(dir_npz + f2, [0.0, -10.0, -10.0,  0.0, -100.0, -100.0,  0.0,  0.0,  0.0,  0.0], "anp")
change_padding(dir_npz + f3, [0.0, -10.0, -10.0,  0.0, -100.0, -100.0,  0.0,  0.0,  0.0,  0.0], "anp")
