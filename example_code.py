import numpy as np
import os

import classes.utilities
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


from inputgenerator.InputGeneratorEnergyCut import gen_input
gen_input(root1)

npz_data = np.load(dir_npz + "NNInputDenseBaseEnergyCut_OptimisedGeometry_BP0mm_2e10protons.npz")
ary_features = npz_data["features"]
print(ary_features)

"""
from scripts import event_display

# event_display.event_display(root1, 1534607)

counter = 0
while counter < 10:
    i = np.random.randint(0, root1.events.numentries)
    event = root1.get_event(i)

    # conditions
    if event.is_ideal_compton:
        continue
    if event.MCSimulatedEventType != 2:
        continue

    if event.MCEnergy_Primary > 4.6 or event.MCEnergy_Primary < 4.2:
        continue
            
    event_display.event_display(root1, n=i)
    counter += 1
"""