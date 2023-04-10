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

def MLEM_Monte_Carlo():
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


def MLEM_Cut_Based():
    # Reading root files and export it to npz files
    root_BP0mm = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
    root_BP5mm = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_offline)

    # generate target arrays
    for root_parser in [root_BP0mm, root_BP5mm]:
        ary_data = np.zeros(shape=(root_parser.events_entries, 9))

        ary_data[:, 0] = root_parser.events["Identified"].array()
        ary_data[:, 1] = root_parser.events["RecoEnergy_e"]["value"].array()
        ary_data[:, 2] = root_parser.events["RecoEnergy_p"]["value"].array()
        ary_data[:, 3] = root_parser.events["RecoPosition_e"]["position"].array().x
        ary_data[:, 4] = root_parser.events["RecoPosition_e"]["position"].array().y
        ary_data[:, 5] = root_parser.events["RecoPosition_e"]["position"].array().z
        ary_data[:, 6] = root_parser.events["RecoPosition_p"]["position"].array().x
        ary_data[:, 7] = root_parser.events["RecoPosition_p"]["position"].array().y
        ary_data[:, 8] = root_parser.events["RecoPosition_p"]["position"].array().z

        # filter for identified events
        ary_data = ary_data[ary_data[:, 0] != 0, :]
        print(len(ary_data), "events")

        # generate MLEM export
        MLEMExport.export_mlem(ary_data[:, 1],
                               ary_data[:, 2],
                               ary_data[:, 3],
                               ary_data[:, 4],
                               ary_data[:, 5],
                               ary_data[:, 6],
                               ary_data[:, 7],
                               ary_data[:, 8],
                               filename=root_parser.file_name + "_cutbased",
                               b_arc=False,
                               b_comptonkinematics=False,
                               b_dacfilter=False,
                               b_backscattering=False,
                               b_elim=False)


MLEM_Cut_Based()
