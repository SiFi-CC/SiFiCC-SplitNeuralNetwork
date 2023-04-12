import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})

from src import RootParser
from src import root_files

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

root_0mm = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_offline)
root_5mm = RootParser(dir_main + root_files.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_offline)


# root_con = RootParser(dir_main + root_files.OptimisedGeometry_Continuous_2e10protons_withTimestamps_offline)

# ----------------------------------------------------------------------------------------------------------------------
# Analysis Compton angle comparison

def theta_compare(root_parser, n):
    list_theta_energy = []
    list_theta_vecdot = []
    list_theta_diff = []

    for i, event in enumerate(root_parser.iterate_events(n=n)):
        if event.theta_energy == 0.0 or event.theta_dotvec == 0.0:
            continue
        list_theta_energy.append(event.theta_energy)
        list_theta_vecdot.append(event.theta_dotvec)
        list_theta_diff.append(event.theta_dotvec - event.theta_energy)

    bins = np.linspace(0, np.pi, 100)
    bins_diff = np.linspace(-np.pi / 2, np.pi / 2, 100)
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].set_xlabel(r"$\theta$ [rad]")
    axs[0].set_ylabel("Counts (a.u.)")
    axs[0].hist(list_theta_energy, bins=bins, histtype=u"step", color="blue", label="Energy")
    axs[0].hist(list_theta_vecdot, bins=bins, histtype=u"step", color="red", label="Vector dot")
    axs[0].grid()
    axs[0].legend()
    axs[1].set_xlabel(r"$\theta_{DotVec} - \theta_{Energy}$ [rad]")
    axs[1].set_ylabel("Counts (a.u.)")
    axs[1].hist(list_theta_diff, bins=bins_diff, histtype=u"step", color="black", label="Energy")
    axs[1].set_yscale("log")
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Tagging statistics


def tagging_statistics(root_file, n):
    n_events = 0
    n_real_coincidence = 0
    n_compton = 0
    n_compton_distributed = 0
    n_compton_pseudo_distributed = 0
    n_compton_pseudo_complete = 0
    n_ideal_compton = 0

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
        if event.is_ideal_compton:
            n_ideal_compton += 1

    print("### Tagging statistics:")
    print("Total events: {}".format(n_events))
    print("RealCoincidence: {} ({:.1f}%)".format(n_real_coincidence, n_real_coincidence / n_events * 100))
    print("Compton: {} ({:.1f}%)".format(n_compton, n_compton / n_events * 100))
    print("ComptonPseudoComplete: {} ({:.1f}%)".format(n_compton_pseudo_complete,
                                                       n_compton_pseudo_complete / n_events * 100))
    print("ComptonPseudoDistributed: {} ({:.1f}%)".format(n_compton_pseudo_distributed,
                                                          n_compton_pseudo_distributed / n_events * 100))
    print("ComptonDistributed: {} ({:.1f}%)".format(n_compton_distributed, n_compton_distributed / n_events * 100))
    print("ComptonIdeal: {} ({:.1f}%)".format(n_ideal_compton, n_ideal_compton / n_events * 100))

tagging_statistics(root_file=root_0mm, n=10000)

# ----------------------------------------------------------------------------------------------------------------------
