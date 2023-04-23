import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------------------

def export_sxax(RootCluster,
                dir_target="",
                n_cs=4,
                n_ca=6):
    # grab correct filepath
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    # Global generator settings
    NAME_TAG = "RNN_" + "S" + str(n_cs) + "A" + str(n_ca)
    n_timesteps = (n_cs + n_ca)
    n_features = 10
    n_cluster = n_cs + n_ca

    # Input feature shape
    n_samples = RootCluster.events_entries

    print("Input feature shape: ")
    print("Samples: ", n_samples)
    print("Timesteps: ", n_timesteps)
    print("Features: ", n_features)

    # create empty arrays for storage
    ary_features = np.zeros(shape=(n_samples, n_timesteps, n_features), dtype=np.float32)
    ary_targets_clas = np.zeros(shape=(n_samples,), dtype=np.float32)
    ary_targets_reg1 = np.zeros(shape=(n_samples, 2), dtype=np.float32)
    ary_targets_reg2 = np.zeros(shape=(n_samples, 6), dtype=np.float32)
    ary_targets_reg3 = np.zeros(shape=(n_samples,), dtype=np.float32)
    ary_w = np.zeros(shape=(n_samples,), dtype=np.float32)
    ary_meta = np.zeros(shape=(n_samples, 6), dtype=np.float32)

    # main iteration over root file
    for i, event in enumerate(RootCluster.iterate_events(n=None)):

        # get indices of clusters sorted by position in reverse order
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=False)

        for j, idx in enumerate(np.flip(idx_scatterer)):
            if j >= n_cs:
                break
            ary_features[i, j, :] = [event.RecoClusterEntries[idx],
                                     event.RecoClusterTimestamps_relative[idx],
                                     event.RecoClusterEnergies_values[idx],
                                     event.RecoClusterPosition[idx].x,
                                     event.RecoClusterPosition[idx].y,
                                     event.RecoClusterPosition[idx].z,
                                     event.RecoClusterEnergies_uncertainty[idx],
                                     event.RecoClusterPosition_uncertainty[idx].x,
                                     event.RecoClusterPosition_uncertainty[idx].y,
                                     event.RecoClusterPosition_uncertainty[idx].z]

        for j, idx in enumerate(np.flip(idx_absorber)):
            if j >= n_ca:
                break
            ary_features[i, (j + n_cs), :] = [event.RecoClusterEntries[idx],
                                              event.RecoClusterTimestamps_relative[idx],
                                              event.RecoClusterEnergies_values[idx],
                                              event.RecoClusterPosition[idx].x,
                                              event.RecoClusterPosition[idx].y,
                                              event.RecoClusterPosition[idx].z,
                                              event.RecoClusterEnergies_uncertainty[idx],
                                              event.RecoClusterPosition_uncertainty[idx].x,
                                              event.RecoClusterPosition_uncertainty[idx].y,
                                              event.RecoClusterPosition_uncertainty[idx].z]

        # target: ideal compton events tag
        ary_targets_clas[i] = event.is_ideal_compton * 1
        ary_targets_reg1[i, :] = np.array([event.MCEnergy_e, event.MCEnergy_p])
        ary_targets_reg2[i, :] = np.array([event.MCPosition_e_first.x,
                                           event.MCPosition_e_first.y,
                                           event.MCPosition_e_first.z,
                                           event.MCPosition_p_first.x,
                                           event.MCPosition_p_first.y,
                                           event.MCPosition_p_first.z])
        ary_targets_reg3[i] = event.theta_dotvec

        # write weighting
        ary_w[i] = 1.0

        # write meta data
        ary_meta[i, :] = [event.EventNumber,
                          event.MCEnergy_Primary,
                          event.MCPosition_source.z,
                          event.MCPosition_source.y,
                          event.RecoClusterEnergies_values[idx_scatterer[0]],
                          np.sum(event.RecoClusterEnergies_values[idx_absorber])]

    # save final output file
    str_savefile = dir_target + RootCluster.file_name + "_" + NAME_TAG + ".npz"

    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            targets_reg3=ary_targets_reg3,
                            weights=ary_w,
                            meta=ary_meta)
