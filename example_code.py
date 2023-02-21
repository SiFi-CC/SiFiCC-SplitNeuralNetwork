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

root3 = RootParser(dir_main + root_files.OptimisedGeometry_Continuous_2e10protons_withTimestamps_offline)

from inputgenerator import InputGenerator_DNN_Base
from inputgenerator import InputGenerator_DNN_S1AX

InputGenerator_DNN_S1AX.gen_input(root3)
InputGenerator_DNN_Base.gen_input(root3)


def backrpojection():
    from src import MLEMBackprojection
    from src import Plotter
    npz_data = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_DNN_Base.npz")
    y_class = npz_data["targets_clas"]
    y_energy = npz_data["targets_reg1"]
    y_position = npz_data["targets_reg2"]
    idx_pos = y_class == 1

    image = MLEMBackprojection.reconstruct_image(y_energy[idx_pos, 0],
                                                 y_energy[idx_pos, 1],
                                                 y_position[idx_pos, 0],
                                                 y_position[idx_pos, 1],
                                                 y_position[idx_pos, 2],
                                                 y_position[idx_pos, 3],
                                                 y_position[idx_pos, 4],
                                                 y_position[idx_pos, 5], )
    Plotter.plot_backprojection(image, "")


def time_plot(root3):
    list_times = []
    list_times_bg = []
    list_times_sg = []
    for i in range(root3.events_entries):
        event = root3.get_event(i)
        for time in event.RecoClusterTimestamps_relative:
            if time != 0.0:
                list_times.append(time)

                if event.is_ideal_compton:
                    list_times_sg.append(time)
                else:
                    list_times_bg.append(time)

    bins = np.arange(-5.0, 5.0, 0.05)
    hist1, _ = np.histogram(list_times_sg, bins=bins)
    hist2, _ = np.histogram(list_times_bg, bins=bins)

    plt.figure()
    plt.xlabel(r"$\Delta t$ [ns]")
    plt.ylabel("counts")
    plt.hist(list_times, bins=bins, color="orange", alpha=0.7, label="All events")
    plt.errorbar(bins[:-1] + 0.05 / 2, hist1, np.sqrt(hist1) / 5, fmt=".", color="red", label="Positives")
    plt.errorbar(bins[:-1] + 0.05 / 2, hist2, np.sqrt(hist2) / 5, fmt=".", color="black", label="Negatives")
    plt.legend()
    plt.tight_layout()
    # plt.grid()
    plt.show()
