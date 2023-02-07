# -------------------------------------------------------------------
# Script to generate MLEM export files with Monte Carlo Truth data
# Focused on the ideal compton event tag

import numpy as np
import os

from src import RootParser
from src import root_files
from src import MLEMExport

########################################################################################################################

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"

########################################################################################################################

# Reading root files and export it to npz files
root_BP0mm = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
root_BP5mm = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_offline)

# generate target arrays
for root_parser in [root_BP0mm, root_BP5mm]:
    # grab ideal compton event statistic
    n_events = 0
    for event in root_parser.iterate_events(n=None):
        if not event.is_ideal_compton:
            continue
        n_events += 1

    ary_data = np.zeros(shape=(n_events, 8), dtype=np.float32)

    idx = 0
    for i, event in enumerate(root_parser.iterate_events(n=None)):
        if not event.is_ideal_compton:
            continue

        ary_data[idx, :] = [event.MCEnergy_e,
                            event.MCEnergy_p,
                            event.MCPosition_e_first.x,
                            event.MCPosition_e_first.y,
                            event.MCPosition_e_first.z,
                            event.MCPosition_p_first.x,
                            event.MCPosition_p_first.y,
                            event.MCPosition_p_first.z]
        idx += 1

    # generate MLEM export
    MLEMExport.export_mlem(ary_data[:, 0],
                           ary_data[:, 1],
                           ary_data[:, 2],
                           ary_data[:, 3],
                           ary_data[:, 4],
                           ary_data[:, 5],
                           ary_data[:, 6],
                           ary_data[:, 7],
                           filename=root_parser.file_name + "_MonteCarlo")
