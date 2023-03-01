"""
Script for flipping the beam direction of saved npz files

"""

import os
import numpy as np

# define directory paths
dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_results = dir_main + "/results/"

f1 = "OptimisedGeometry_Continuous_2e10protons_DNN_S1AX.npz"
f2 = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz"
f3 = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz"


def beam_flip(filename):
    # open training file
    npz_data = np.load(dir_npz + filename)
    ary_features = npz_data["features"]
    ary_theta = npz_data["theta"]
    ary_targets_clas = npz_data["targets_clas"]
    ary_targets_reg1 = npz_data["targets_reg1"]
    ary_targets_reg2 = npz_data["targets_reg2"]
    ary_w = npz_data["weights"]
    ary_meta = npz_data["META"]

    ary_features[:, [5, 15, 25, 35, 45, 55, 65, 75]] *= -1
    ary_targets_reg2[:, [2, 5]] *= -1
    ary_meta[:, 2] *= -1

    with open(dir_npz + filename[:-4] + "_flip.npz", 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            theta=ary_theta,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            weights=ary_w,
                            META=ary_meta)
    print(filename + " flipped!")


beam_flip(f1)
beam_flip(f2)
beam_flip(f3)
