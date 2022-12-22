import numpy as np
import os

from classes import RootParser
from classes import root_files


def check_compton_kinematics(event):
    # TODO: check correctness
    ELECTRON_MASS = 0.511
    event_energy = np.sum(event.RecoClusterEnergies_values)
    event_energy_uncertainty = np.sqrt(np.sum(event.RecoClusterEnergies_values ** 2))
    compton_edge = ((event_energy / (1 + ELECTRON_MASS / (2 * event_energy))),
                    event_energy * (ELECTRON_MASS + event_energy) / (ELECTRON_MASS / 2 + event_energy) / (
                            ELECTRON_MASS / 2 + event_energy) * event_energy_uncertainty)
    electron_energy_value, electron_energy_uncertainty = event.get_electron_energy()
    # print(electron_energy_value - electron_energy_uncertainty, compton_edge[0] + compton_edge[1])
    if electron_energy_value - electron_energy_uncertainty > compton_edge[0] + compton_edge[1]:
        return False
    return True


def main():
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"

    root_data = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)

    n = 1000
    positives = 0.0

    for i, event in enumerate(root_data.iterate_events(n=n)):
        cb_reco_tag = 1
        cb_true_tag = event.Identified

        # sort cluster by index
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

        # check scatterer cluster multiplicity
        if not len(idx_scatterer) == 1:
            cb_reco_tag = 0

        if not check_compton_kinematics(event):
            cb_reco_tag = 0

        if cb_reco_tag == 1.0 and cb_true_tag != 0.0:
            positives += 1.0
        if cb_reco_tag == 0.0 and cb_true_tag == 0.0:
            positives += 1.0

    print("Positive Rate: {:.1f}%".format(positives / n * 100))


if __name__ == "__main__":
    main()
