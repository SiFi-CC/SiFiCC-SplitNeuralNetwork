import numpy as np
import os

from classes import Rootdata
from classes import root_files

dir_main = os.getcwd()
dir_data = dir_main + "/data/"


########################################################################################################################
# Neural Network input denotation (very much work in progress):
# NNinput: denoting the generated .npz file is for neural network training, conaining features and targets
# OptimizedGeometry: denoting the root file
# BPXmm: source position
# XeXprotons: number of protons simulated for the file
#
# CLstat: usage of cluster index and cluster number
# MAXCLs4a4: number of clusters used for each detector module
#
# Example: NNinput_OptimizedGeometry_BP0mm_2e10protons_MAXCLs4a4
########################################################################################################################

def CLstat_MAXCLs4a4(n=None):
    """
    create NN input data with features: # scatterer clusters, # absorber clusters,
                                        4 highest energy cluster for each module

    :param n: number of entries in the training data. If none: all in the root file
    :return: npz file containing features, targets
    """

    # load root file and set number of events to iterate
    root = Rootdata(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)
    if n is None:
        n = root.events_entries

    # create empty dataframe
    num_features = 42
    ary_features = np.zeros(shape=(n, num_features))
    ary_targets = np.zeros(shape=(n,))
    ary_weighting = np.zeros(shape=(n,))

    for i, event in enumerate(root.iterate_events(n=n)):
        # get indices of clusters sorted by highest energy and module
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

        # fill up scatterer and absorber entries with zero padding
        cl_scatterer = np.zeros(shape=(20,))
        cl_absorber = np.zeros(shape=(20,))

        for j, idx in enumerate(idx_scatterer):
            if j >= 4:
                break
            cl_scatterer[j * 5: (j * 5) + 5] = [event.RecoClusterEntries[idx],
                                                event.RecoClusterEnergies_values[idx],
                                                event.RecoClusterPosition[idx].x,
                                                event.RecoClusterPosition[idx].y,
                                                event.RecoClusterPosition[idx].z]
        for j, idx in enumerate(idx_absorber):
            if j >= 4:
                break
            cl_absorber[j * 5: (j * 5) + 5] = [event.RecoClusterEntries[idx],
                                               event.RecoClusterEnergies_values[idx],
                                               event.RecoClusterPosition[idx].x,
                                               event.RecoClusterPosition[idx].y,
                                               event.RecoClusterPosition[idx].z]

        # fill dataset with extracted features
        ary_features[i, :2] = [len(idx_scatterer), len(idx_absorber)]
        ary_features[i, 2:22] = cl_scatterer
        ary_features[i, 22:42] = cl_absorber
        # target: ideal compton events tag
        ary_targets[i] = event.is_ideal_compton * 1
        # weighting: primary energy for now
        ary_weighting[i] = event.MCEnergy_Primary

    # save file as .npz
    with open(dir_data + "NNinput_OptimizedGeometry_BP0mm_2e10protons_MAXCLs4a4.npz", 'wb') as f_train:
        np.savez_compressed(f_train,
                            features=ary_features,
                            targets=ary_targets,
                            weighting=ary_weighting)


########################################################################################################################

CLstat_MAXCLs4a4(n=None)
