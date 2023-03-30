import numpy as np
import os
import src.utilities
from src import RootParser
from src import root_files
from src import NPZParser

from inputgenerator import InputGenerator_DNN_S1AX

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


def tagging_statistics(root_file, n):
    n_events = 0
    n_real_coincidence = 0
    n_compton = 0
    n_compton_distributed = 0
    n_compton_pseudo_distributed = 0
    n_compton_pseudo_complete = 0

    for i, event in enumerate(root_file.iterate_events(n=n)):
        n_events += 1
        if event.is_real_coincidence:
            n_real_coincidence += 1
        if event.is_compton:
            n_compton += 1
        if event.is_compton_distributed:
            n_compton_distributed += 1
        if event.is_compton_pseudo_distributed:
            n_compton_pseudo_distributed += 1
        if event.is_compton_pseudo_complete:
            n_compton_pseudo_complete += 1

    print("### Tagging statistics:")
    print("Total events: {}".format(n_events))
    print("RealCoincidence: {} ({:.1f}%)".format(n_real_coincidence, n_real_coincidence / n_events * 100))
    print("Compton: {} ({:.1f}%)".format(n_compton, n_compton / n_events * 100))
    print("ComptonPseudoComplete: {} ({:.1f}%)".format(n_compton_pseudo_complete,
                                                       n_compton_pseudo_complete / n_events * 100))
    print("ComptonPseudoDistributed: {} ({:.1f}%)".format(n_compton_pseudo_distributed,
                                                          n_compton_pseudo_distributed / n_events * 100))
    print("ComptonDistributed: {} ({:.1f}%)".format(n_compton_distributed, n_compton_distributed / n_events * 100))


tagging_statistics(root_file=root_0mm, n=100000)

# ----------------------------------------------------------------------------------------------------------------------
