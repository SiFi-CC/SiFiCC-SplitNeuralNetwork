import numpy as np
import os


def gen_input(RootParser):
    """
    Description of generation method and motivation for choosing this method

    Args:
        RootParser (obj: RootParser): RootParser object

    Return:
         None
    """
    # input parameter handler
    list_rootparser = []
    if type(RootParser) is not list:
        list_rootparser.append(RootParser)
    else:
        list_rootparser = RootParser

    # grab correct filepath
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    # global settings for easier on the fly changes
    NAME_TAG = "DNN_S1AX"
    n_cluster_scatterer = 1
    n_cluster_absorber = 5

    n_features = 10 * (n_cluster_scatterer + n_cluster_absorber)
    n_cluster = n_cluster_scatterer + n_cluster_absorber

    # get the number of valid events per root file
    list_entries = []

    for i in range(len(list_rootparser)):
        n_events = 0
        root_parser = list_rootparser[i]
        print("Scanning ", root_parser.file_name)
        ary_cluster_positions = root_parser.events["RecoClusterPositions.position"].array()
        ary_identified = root_parser.events["Identified"].array()

        for j in range(root_parser.events_entries):
            counter_scatterer = 0
            counter_absorber = 0
            for tvec in ary_cluster_positions[j]:
                if 150.0 - 20.8 / 2 < tvec.x < 150.0 + 20.8 / 2:
                    counter_scatterer += 1
                if 270.0 - 46.8 / 2 < tvec.x < 270.0 + 46.8 / 2:
                    counter_absorber += 1

            # filter condition
            if counter_scatterer == 1 and counter_absorber > 0:
                n_events += 1
        print(n_events, "valid events found")
        print(root_parser.events_entries, "total events in the given root file")
        list_entries.append(n_events)

    n_events = np.sum(list_entries)
    print("total entries: ", n_events)

    # create empty arrays for storage
    ary_features = np.zeros(shape=(n_events, n_features), dtype=np.float32)
    ary_targets_clas = np.zeros(shape=(n_events,), dtype=np.float32)
    ary_targets_reg1 = np.zeros(shape=(n_events, 2), dtype=np.float32)
    ary_targets_reg2 = np.zeros(shape=(n_events, 6), dtype=np.float32)
    ary_w = np.zeros(shape=(n_events,), dtype=np.float32)
    ary_meta = np.zeros(shape=(n_events, 6), dtype=np.float32)
    ary_theta = np.zeros(shape=(n_events,), dtype=np.float32)
    ary_tags = np.zeros(shape=(n_events, 5), dtype=np.float32)
    # legacy targets
    ary_targets = np.zeros(shape=(n_events,), dtype=np.float32)

    idx_pos = 0
    for k in range(len(list_rootparser)):
        # main iteration over root file
        for i, event in enumerate(list_rootparser[k].iterate_events(n=None)):

            # get indices of clusters sorted by highest energy and module
            idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

            if not len(idx_scatterer) == 1:
                continue

            if not len(idx_absorber) > 0:
                continue

            # fill up scatterer and absorber entries
            # Non-existing clusters are filled with points outside the detector range
            # filled values need same type as target array
            ary_feat_scatterer = np.array(
                [0.0, -1.0, -1.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0, 0.0] * n_cluster_scatterer)
            ary_feat_absorber = np.array([0.0, -1.0, -1.0, 0.0, -55.0, -55.0, 0.0, 0.0, 0.0, 0.0] * n_cluster_absorber)

            for j, idx in enumerate(idx_scatterer):
                if j >= n_cluster_scatterer:
                    break
                ary_feat_scatterer[j * 10: (j * 10) + 10] = [event.RecoClusterEntries[idx],
                                                             event.RecoClusterTimestamps_relative[idx],
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
                ary_feat_absorber[j * 10: (j * 10) + 10] = [event.RecoClusterEntries[idx],
                                                            event.RecoClusterTimestamps_relative[idx],
                                                            event.RecoClusterEnergies_values[idx],
                                                            event.RecoClusterPosition[idx].x,
                                                            event.RecoClusterPosition[idx].y,
                                                            event.RecoClusterPosition[idx].z,
                                                            event.RecoClusterEnergies_uncertainty[idx],
                                                            event.RecoClusterPosition_uncertainty[idx].x,
                                                            event.RecoClusterPosition_uncertainty[idx].y,
                                                            event.RecoClusterPosition_uncertainty[idx].z]

            # fill final feature array with scatterer and absorber features
            ary_features[idx_pos, :10 * n_cluster_scatterer] = ary_feat_scatterer
            ary_features[idx_pos, 10 * n_cluster_scatterer:10 * n_cluster] = ary_feat_absorber

            # target: ideal compton events tag
            ary_targets[idx_pos] = event.is_ideal_compton * 1

            ary_targets_clas[idx_pos] = event.is_ideal_compton * 1
            ary_targets_reg1[idx_pos, :] = np.array([event.MCEnergy_e, event.MCEnergy_p])
            ary_targets_reg2[idx_pos, :] = np.array([event.MCPosition_e_first.x,
                                                     event.MCPosition_e_first.y,
                                                     event.MCPosition_e_first.z,
                                                     event.MCPosition_p_first.x,
                                                     event.MCPosition_p_first.y,
                                                     event.MCPosition_p_first.z])

            # write weighting
            ary_w[idx_pos] = np.sum(list_entries) / len(list_entries) / list_entries[k]

            ary_theta[idx_pos] = event.theta

            # write meta data
            ary_meta[idx_pos, :] = [event.EventNumber,
                                    event.MCEnergy_Primary,
                                    event.MCPosition_source.z,
                                    event.MCPosition_source.y,
                                    event.RecoClusterEnergies_values[idx_scatterer[0]],
                                    np.sum(event.RecoClusterEnergies_values[idx_absorber])]

            # write tags
            ary_tags[idx_pos, :] = [event.is_real_coincidence*1,
                                    event.is_compton*1,
                                    event.is_compton_pseudo_complete*1,
                                    event.is_compton_pseudo_distributed*1,
                                    event.is_compton_distributed*1]

            idx_pos += 1

    # save final output file
    str_savefile = ""
    if type(RootParser) is list:
        str_savefile = dir_npz + "OptimizedGeometry_Mixed_" + NAME_TAG + ".npz"
    else:
        str_savefile = dir_npz + RootParser.file_name + "_" + NAME_TAG + ".npz"

    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            theta=ary_theta,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            weights=ary_w,
                            META=ary_meta,
                            tags=ary_tags)
