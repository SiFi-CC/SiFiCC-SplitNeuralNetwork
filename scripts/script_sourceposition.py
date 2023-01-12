"""
Script:
Showcase of Monte-Carlo source position distribution for different scenarios:
- CB-selected events
- S1AX events

"""

import numpy as np
import os
import matplotlib.pyplot as plt


def source_position(root1, root2):
    # root1 and root2 are two different RootParser object for 2 bragg peak positions
    dir_main = os.getcwd()

    n = 400000
    counter_1 = 0
    counter_2 = 0

    list_sp_cb_1 = []
    list_sp_cb_2 = []

    list_ic_1 = []
    list_ic_2 = []

    list_dc_1 = []
    list_dc_2 = []

    for i, event in enumerate(root1.iterate_events(n=n)):
        if event.Identified != 0 and event.MCPosition_source.z != 0.0:
            list_sp_cb_1.append(event.MCPosition_source.z)

        if event.is_ideal_compton and np.sum(event.RecoClusterEnergies_values) > 1.0:
            list_ic_1.append(event.MCPosition_source.z)

        if event.is_complete_distributed_compton:
            list_dc_1.append(event.MCPosition_source.z)

        if np.sum(event.RecoClusterEnergies_values) > 1.0:
            counter_1 += 1

    for i, event in enumerate(root2.iterate_events(n=n)):
        if event.Identified != 0 and event.MCPosition_source.z != 0.0:
            list_sp_cb_2.append(event.MCPosition_source.z)

        if event.is_ideal_compton and np.sum(event.RecoClusterEnergies_values) > 1.0:
            list_ic_2.append(event.MCPosition_source.z)

        if event.is_complete_distributed_compton:
            list_dc_2.append(event.MCPosition_source.z)

        if np.sum(event.RecoClusterEnergies_values) > 1.0:
            counter_2 += 1

    print("{:.1f}".format(counter_1/n*100))
    print("{:.1f}".format(counter_2/n*100))

    # plot MC Source Position z-direction
    bins = np.arange(-80, 20, 1.0)
    width = abs(bins[0] - bins[1])
    hist1, _ = np.histogram(list_sp_cb_1, bins=bins)
    hist2, _ = np.histogram(list_sp_cb_2, bins=bins)

    hist3, _ = np.histogram(list_ic_1, bins=bins)
    hist4, _ = np.histogram(list_ic_2, bins=bins)

    hist5, _ = np.histogram(list_dc_1, bins=bins)
    hist6, _ = np.histogram(list_dc_2, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z (Cut-Based)")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts (normalized)")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(list_sp_cb_1, bins=bins, histtype=u"step", color="black", label="0mm", density=True,
             alpha=0.5, linestyle="--")
    plt.hist(list_sp_cb_2, bins=bins, histtype=u"step", color="red", label="5mm", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
                 np.sqrt(hist1) / np.sum(hist1) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
                 np.sqrt(hist2) / np.sum(hist2) / width, color="red", fmt=".")
    plt.vlines(ymax=0.02, ymin=0, x=-11.0, color="blue", linestyles="--")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("MC Source Position z (Ideal Compton)")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts (normalized)")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(list_ic_1, bins=bins, histtype=u"step", color="black", label="0mm", density=True,
             alpha=0.5, linestyle="--")
    plt.hist(list_ic_2, bins=bins, histtype=u"step", color="red", label="5mm", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist3 / np.sum(hist3) / width,
                 np.sqrt(hist3) / np.sum(hist3) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist4 / np.sum(hist4) / width,
                 np.sqrt(hist4) / np.sum(hist4) / width, color="red", fmt=".")
    plt.vlines(ymax=0.02, ymin=0, x=-11.0, color="blue", linestyles="--")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.title("MC Source Position z (Distributed Compton)")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts (normalized)")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(list_dc_1, bins=bins, histtype=u"step", color="black", label="0mm", density=True,
             alpha=0.5, linestyle="--")
    plt.hist(list_dc_2, bins=bins, histtype=u"step", color="red", label="5mm", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist5 / np.sum(hist5) / width,
                 np.sqrt(hist5) / np.sum(hist5) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist6 / np.sum(hist6) / width,
                 np.sqrt(hist6) / np.sum(hist6) / width, color="red", fmt=".")
    plt.vlines(ymax=0.02, ymin=0, x=-11.0, color="blue", linestyles="--")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()