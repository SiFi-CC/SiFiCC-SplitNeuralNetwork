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
root1 = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
# root1.export_npz(dir_npz + "OptimisedGeometry_BP0mm_2e10protons.npz")

root2 = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_offline)
# root2.export_npz(dir_npz + "OptimisedGeometry_BP5mm_4e9protons.npz")

from src import EventDisplay

counter = 0
for i, event in enumerate(root1.iterate_events(n=None)):
    if event.is_compton_pseudo_complete and not event.is_compton_pseudo_distributed:
        EventDisplay.event_display(root1, event_position=i)
        counter += 1
    if counter >= 10:
        break
