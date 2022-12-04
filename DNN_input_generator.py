########################################################################################################################
# Deep Neural Network input generator (very much work in progress):
# For fast and easy use, the input generator is created individually for every neural network model
# An argument parser has been defined to help to executing the script on different systems (local or on linux-cluster)
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

    argparser = argparse.ArgumentParser(description="Some helper text", formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument("--local", action="store_true", help="Use local stored root files")
    argparser.add_argument("--cluster", action="store_true", help="Use cluster stored root files")
    return argparser.parse_args()


########################################################################################################################
# Define settings for input generation
########################################################################################################################

file_name = "DNN_input_OptimizedGeometry_BP0mm_2e10protons"

n_events = None # Number of final events to be exported to output file
n_cluster_scatterer = 3  # number of clusters in scatterer used
n_cluster_absorber = 5  # number of clusters in absorber used
n_cluster = n_cluster_scatterer + n_cluster_absorber

# test, validation and training set percentages of total events used for output
p_test = 0.2
p_valid = 0.1
p_train = 0.7

########################################################################################################################

# define directory paths
dir_main = os.getcwd()
dir_data = dir_main + "/data/"

# input file
args = parser()
# if the input file is undefined, the local file will be used
input_file = Rootdata(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)
if args.local:
    input_file = Rootdata(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)
if args.cluster:
    input_file = Rootdata(root_files.OptimisedGeometry_BP0mm_2e10protons)

# set total number of events used
if n_events is None:
    n_events = input_file.events_entries

# calculate final amount of features
num_features = 2  # starting at 2 for cluster counts in each module
num_features += 5 * n_cluster_scatterer + 5 * n_cluster_absorber

# create empty arrays for storage
ary_features = np.zeros(shape=(n_events, num_features))
ary_targets = np.zeros(shape=(n_events,))
ary_w = np.zeros(shape=(n_events,))
ary_event_number = np.zeros(shape=(n_events,))

# main iteration over root file
for i, event in enumerate(input_file.iterate_events(n=n_events)):

    # get indices of clusters sorted by highest energy and module
    idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

    # fill up scatterer and absorber entries
    # Non-existing clusters are filled with points outside the detector range
    ary_feat_scatterer = np.array([0, -1, 0, -55, -55] * n_cluster_scatterer)
    ary_feat_absorber = np.array([0, -1, 0, -55, -55] * n_cluster_absorber)

    for j, idx in enumerate(idx_scatterer):
        if j >= n_cluster_scatterer:
            break
        ary_feat_scatterer[j * 5: (j * 5) + 5] = [event.RecoClusterEntries[idx],
                                                  event.RecoClusterEnergies_values[idx],
                                                  event.RecoClusterPosition[idx].x,
                                                  event.RecoClusterPosition[idx].y,
                                                  event.RecoClusterPosition[idx].z]
    for j, idx in enumerate(idx_absorber):
        if j >= n_cluster_absorber:
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
    ary_w[i] = event.MCEnergy_Primary

    # write global event number
    ary_event_number[i] = event.EventNumber

# train valid test split
idx = np.arange(0, ary_features.shape[0], 1.0, dtype=int)
np.random.shuffle(idx)
idx_step1 = int(len(idx) * p_train)
idx_step2 = int(len(idx) * (p_train + p_valid))

ary_idx_train = idx[0:idx_step1]
ary_idx_valid = idx[idx_step1:idx_step2]
ary_idx_test = idx[idx_step2:]

# save final output file
with open(dir_data + file_name + ".npz", 'wb') as f_output:
    np.savez_compressed(f_output,
                        features=ary_features,
                        targets=ary_targets,
                        weights=ary_w,
                        idx_train=ary_idx_train,
                        idx_valid=ary_idx_valid,
                        idx_test=ary_idx_test,
                        eventnumber=ary_event_number)
