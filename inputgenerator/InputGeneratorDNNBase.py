# some header

def gen_input(RootParser):
    """
    Description of generation method and motivation for choosing this method

    Args:
        RootParser (obj: RootParser): RootParser object

    Return:
         none
    """

    import numpy as np
    import os

    ####################################################################################################################
    # global settings for easier on the fly changes
    NAME_TAG = "DNNBase"

    n_cluster_scatterer = 2
    n_cluster_absorber = 6

    # grab correct filepath
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    # calculate final amount of features
    num_features = 2  # starting at 2 for cluster counts in each module
    num_features += 9 * n_cluster_scatterer + 9 * n_cluster_absorber
    n_cluster = n_cluster_scatterer + n_cluster_absorber
    n_events = RootParser.events_entries

    # create empty arrays for storage
    ary_features = np.zeros(shape=(n_events, num_features), dtype=np.float32)
    ary_targets_clas = np.zeros(shape=(n_events,), dtype=np.float32)
    ary_targets_reg1 = np.zeros(shape=(n_events, 2), dtype=np.float32)
    ary_targets_reg2 = np.zeros(shape=(n_events, 6), dtype=np.float32)
    ary_w = np.zeros(shape=(n_events,), dtype=np.float32)
    ary_meta = np.zeros(shape=(n_events, 3), dtype=np.float32)

    # legacy targets
    ary_targets = np.zeros(shape=(n_events,), dtype=np.float32)

    # main iteration over root file
    for i, event in enumerate(RootParser.iterate_events(n=None)):

        # get indices of clusters sorted by highest energy and module
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

        # cluster feature format:
        # [entries, energy, pos x, pos y, pos z, energy unc,pos x unc, pos y unc, pos z unc]

        # fill up scatterer and absorber entries
        # Non-existing clusters are filled with points outside the detector range
        # filled values need same type as target array
        ary_feat_scatterer = np.array([0.0, -1.0, 0.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0] * n_cluster_scatterer)
        ary_feat_absorber = np.array([0.0, -1.0, 0.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0] * n_cluster_absorber)

        for j, idx in enumerate(idx_scatterer):
            if j >= n_cluster_scatterer:
                break
            ary_feat_scatterer[j * 9: (j * 9) + 9] = [event.RecoClusterEntries[idx],
                                                      event.RecoClusterEnergies_values[idx],
                                                      event.RecoClusterPosition[idx].x,
                                                      event.RecoClusterPosition[idx].y,
                                                      event.RecoClusterPosition[idx].z,
                                                      event.RecoClusterEnergies_uncertainty[idx],
                                                      event.RecoClusterPosition_uncertainty[idx].x,
                                                      event.RecoClusterPosition_uncertainty[idx].y,
                                                      event.RecoClusterPosition_uncertainty[idx].z]

        for j, idx in enumerate(idx_absorber):
            if j >= n_cluster_absorber:
                break
            ary_feat_absorber[j * 9: (j * 9) + 9] = [event.RecoClusterEntries[idx],
                                                     event.RecoClusterEnergies_values[idx],
                                                     event.RecoClusterPosition[idx].x,
                                                     event.RecoClusterPosition[idx].y,
                                                     event.RecoClusterPosition[idx].z,
                                                     event.RecoClusterEnergies_uncertainty[idx],
                                                     event.RecoClusterPosition_uncertainty[idx].x,
                                                     event.RecoClusterPosition_uncertainty[idx].y,
                                                     event.RecoClusterPosition_uncertainty[idx].z]

        # fill final feature array with scatterer and absorber features
        ary_features[i, :9 * n_cluster_scatterer] = ary_feat_scatterer
        ary_features[i, 9 * n_cluster_scatterer:9 * n_cluster] = ary_feat_absorber

        # fill cluster counts at the end of the feature array
        ary_features[i, 9 * n_cluster:9 * n_cluster + 2] = np.array([len(idx_scatterer), len(idx_absorber)])

        # target: ideal compton events tag
        ary_targets[i] = event.is_ideal_compton * 1

        ary_targets_clas[i] = event.is_ideal_compton * 1
        ary_targets_reg1[i, :] = np.array([event.MCEnergy_e, event.MCEnergy_p])
        ary_targets_reg2[i, :] = np.array([event.MCPosition_e_first.x,
                                           event.MCPosition_e_first.y,
                                           event.MCPosition_e_first.z,
                                           event.MCPosition_p_first.x,
                                           event.MCPosition_p_first.y,
                                           event.MCPosition_p_first.z])

        # sample weighting
        ary_w[i] = 1.0

        # write global event number
        ary_meta[i, :] = [event.EventNumber, event.MCEnergy_Primary, np.sum(event.RecoClusterEnergies_values)]

    # save final output file
    with open(dir_npz + RootParser.file_name + "_" + NAME_TAG + ".npz", 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            weights=ary_w,
                            META=ary_meta)
