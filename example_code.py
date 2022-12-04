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
root1 = RootParser(root_files.OptimisedGeometry_BP0mm_2e10protons)
root1.export_npz(dir_npz + "OptimisedGeometry_BP0mm_2e10protons.npz")

root2 = RootParser(root_files.OptimisedGeometry_BP5mm_4e9protons)
root2.export_npz(dir_npz + "OptimisedGeometry_BP5mm_4e9protons.npz")


