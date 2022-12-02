########################################################################################################################
# Neural Network input denotation (very much work in progress):
# Current goal is to define a script for generating different versions of neural network input files
# An argument parser has been defined to help executing the script on different systems (local or on linux-cluster)
#
########################################################################################################################

import numpy as np
import os
import copy

from classes import Rootdata
from classes import root_files


def parser():
    """
    Defines argument parser. Main goal is to make handling the data for cluster or local use easier.

    return: parser
    """
    import argparse

    parser = argparse.ArgumentParser(description="Some helper text", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--local", action="store_true", help="Use local stored root files")
    parser.add_argument("--cluster", action="store_true", help="Use cluster stored root files")
    return parser.parse_args()


########################################################################################################################
# Generation of Neural Network input data
########################################################################################################################

def gen_output_dense(input_file, file_name):
    ######################################################################################
    # general settings for output file generation
    n_events = 1000  # Number of final events to be exported to output file

    n_cluster_scatterer = 4  # number of clusters in scatterer used
    n_cluster_absorber = 4  # number of clusters in absorber used

    use_sort_energy = True  # all clusters are sorted by energy, else sorted by position
    use_cluster_count = True  # adds cluster counts per module as features
    use_standardize = True  # normalize all features to mean 0 std 1

    # test, validation and training set percentages of total events used for output
    p_test = 0.2
    p_valid = 0.1
    p_train = 0.7
    ######################################################################################

    # set total number of events used
    if n_events is None:
        n_events = input_file.events_entries

    # calculate final amount of features
    n_cluster = n_cluster_scatterer + n_cluster_absorber
    num_features = 0
    num_features += 5 * n_cluster_scatterer + 5 * n_cluster_absorber
    if use_cluster_count:
        num_features += 2

    # create empty arrays for storage
    ary_features = np.zeros(shape=(n_events, num_features))
    ary_targets = np.zeros(shape=(n_events,))
    ary_w_primary_energy = np.zeros(shape=(n_events,))

    # main iteration over root file
    for i, event in enumerate(input_file.iterate_events(n=n_events)):

        # get indices of clusters sorted by highest energy and module
        if use_sort_energy:
            idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)
        else:
            # TODO: fix inverse ordering of clusters
            idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=False)

        # fill up scatterer and absorber entries with zero padding
        ary_feat_scatterer = np.array([0, -1, 0, -55, -55] * n_cluster_scatterer)
        ary_feat_absorber = np.array([0, -1, 0, -55, -55] * n_cluster_absorber)
        for j, idx in enumerate(idx_scatterer):
            if j >= 4:
                break
            ary_feat_scatterer[j * 5: (j * 5) + 5] = [event.RecoClusterEntries[idx],
                                                      event.RecoClusterEnergies_values[idx],
                                                      event.RecoClusterPosition[idx].x,
                                                      event.RecoClusterPosition[idx].y,
                                                      event.RecoClusterPosition[idx].z]
        for j, idx in enumerate(idx_absorber):
            if j >= 4:
                break
            ary_feat_absorber[j * 5: (j * 5) + 5] = [event.RecoClusterEntries[idx],
                                                     event.RecoClusterEnergies_values[idx],
                                                     event.RecoClusterPosition[idx].x,
                                                     event.RecoClusterPosition[idx].y,
                                                     event.RecoClusterPosition[idx].z]

        # fill scatterer absorber features first
        ary_features[i, :5 * n_cluster_scatterer] = ary_feat_scatterer
        ary_features[i, 5 * n_cluster_scatterer:5 * n_cluster_scatterer + 5 * n_cluster_absorber] = ary_feat_absorber
        # fill cluster counts, if not used, filled with 0 and removed later
        ary_features[i, 5 * n_cluster:5 * n_cluster + 2] = np.array([len(idx_scatterer), len(idx_absorber)])

        # target: ideal compton events tag
        ary_targets[i] = event.is_ideal_compton * 1

        # energy weighting: first only primary energy is stored
        ary_w_primary_energy[i] = event.MCEnergy_Primary

    # determination of final weighting
    # define class weights
    _, counts = np.unique(ary_targets, return_counts=True)
    class_weights = [1 / counts[0], 1 / counts[1]]

    # energy weights
    energy_bins = np.arange(0.0, 20.0, 0.1)
    hist, _ = np.histogram(ary_w_primary_energy, bins=energy_bins)
    for i in range(len(ary_w_primary_energy)):
        for j in range(len(energy_bins) - 1):
            if energy_bins[j] < ary_w_primary_energy[i] < energy_bins[j + 1]:
                # primary energy weight
                ary_w_primary_energy[i] = 1 / hist[j]
                # class weight
                ary_w_primary_energy[i] *= class_weights[int(ary_targets[i])]
                break

    # normalization
    for i in range(ary_features.shape[1]):
        ary_features[:, i] = (ary_features[:, i] - np.mean(ary_features[:, i])) / np.std(ary_features[:, i])

    # train valid test split
    idx = np.arange(0, ary_features.shape[0], 1.0, dtype=int)
    np.random.shuffle(idx)
    idx_step1 = int(len(idx) * p_train)
    idx_step2 = int(len(idx) * (p_train + p_valid))

    idx_train = idx[0:idx_step1]
    idx_valid = idx[idx_step1:idx_step2]
    idx_test = idx[idx_step2:]

    # save file as .npz
    with open(file_name + "_training.npz", 'wb') as f_train:
        np.savez_compressed(f_train,
                            features=ary_features[idx_train],
                            targets=ary_targets[idx_train],
                            weights=ary_w_primary_energy[idx_train])
    with open(file_name + "_valid.npz", 'wb') as f_valid:
        np.savez_compressed(f_valid,
                            features=ary_features[idx_valid],
                            targets=ary_targets[idx_valid],
                            weights=ary_w_primary_energy[idx_valid])
    with open(dir_data + file_name + "_test.npz", 'wb') as f_test:
        np.savez_compressed(f_test,
                            features=ary_features[idx_test],
                            targets=ary_targets[idx_test],
                            weights=ary_w_primary_energy[idx_test])


########################################################################################################################

def __main__():
    # define directory paths
    global dir_data

    dir_main = os.getcwd()
    dir_data = dir_main + "/data/"
    file_name = ""

    # input file
    args = parser()
    if args.local:
        input_file = Rootdata(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)
    if args.cluster:
        input_file = Rootdata(root_files.OptimisedGeometry_BP0mm_2e10protons)
