import numpy as np
import os

from classes import RootParser
from classes import root_files

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

from classes import InputGeneratorDenseBase
InputGeneratorDenseBase.gen_input(root1)
InputGeneratorDenseBase.gen_input(root2)

"""
npz_data = np.load(dir_npz + "NNinputDenseBase.npz")
ary_features = npz_data["features"]
ary_targets = npz_data["targets"]
ary_w = npz_data["weights"]
"""


