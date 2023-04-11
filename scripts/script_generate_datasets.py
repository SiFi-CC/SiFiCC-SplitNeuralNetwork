import numpy as np
import os
import src.utilities
from src import RootParser
from src import root_files
from src import NPZParser

from inputgenerator import InputGenerator_RNN_Base

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

# ----------------------------------------------------------------------------------------------------------------------

root_0mm = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
root_5mm = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_offline)
root_con = RootParser(dir_main + root_files.OptimisedGeometry_Continuous_2e10protons_withTimestamps_offline)

# ----------------------------------------------------------------------------------------------------------------------

os.chdir(dir_main)
InputGenerator_RNN_Base.gen_input(root_0mm)
InputGenerator_RNN_Base.gen_input(root_5mm)
InputGenerator_RNN_Base.gen_input(root_con)
