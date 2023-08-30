####################################################################################################
# Script Analysis phantom-hits advanced
#
# Script for analysing for the phenomenon of phantom compton events
# Phantom hits are described by having a valid compton event structure but the absorber
# interaction of a primary gamma is missing, instead a secondary fills the role with a valid
# position. In the advanced script only root files containing energy depositions of interactions
# can be used as these are scanned to determine pair-production processes
#
####################################################################################################

import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})

from SiFiCCNN.root import RootParser, RootFiles
from SiFiCCNN.utils.physics import vector_angle

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    path = os.path.abspath(os.path.join(path, os.pardir))
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
path_main = path
path_root = path + "/root_files/"

# load root files
root_parser = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_taggingv2)

# --------------------------------------------------------------------------------------------------
# Main loop over both root files

# number of entries considered for analysis (either integer or None for all entries available)
n = 10000

if n is None:
    n_entries = root_parser.events_entries
else:
    n_entries = n

# predefine all needed variables

# pp: Pair-production
# ph: Phantom-hits
n_pp = 0
n_pp_ph = 0
n_ph = 0

# main iteration over root file
for i, event in enumerate(root_parser.iterate_events(n=n)):
    tag = event.get_distcompton_tag()
    if event.b_phantom_hit:
        n_ph += 1

    # scan for events with zero energy depositions
    n_e_miss = np.sum((event.MCEnergyDeps_p == 0.0)*1)
    if n_e_miss > 0.0:
        # iterate over new interaction list
        for j in range(len(event.MCEnergyDeps_p)):
            tmp_vec = event.MCPosition_p[j] - event.MCComptonPosition
            r = tmp_vec.mag
            tmp_vec /= tmp_vec.mag
            tmp_angle = vector_angle(tmp_vec, event.MCDirection_scatter)

            if tmp_angle == 0 and event.MCEnergyDeps_p[j] == 0.0:
                n_pp += 1

                if event.b_phantom_hit:
                    n_pp_ph += 1


# ##################################################################################################
# Analysis output
# print general statistics of root file for control
print("LOADED " + root_parser.file_name)
print("Pair-production      : {:.1f} % total".format(n_pp / n * 100))
print("    - phantom hits   : {:.1f} % (of PP)".format(n_pp_ph / n_pp * 100))
print("    - phantom hits   : {:.1f} % (of PH total)".format(n_pp_ph / n_ph * 100))

# ##################################################################################################
# Determining the correct acceptance threshold
