import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------------------

def export_sxax(RootParser,
                dir_target="",
                n_ca=7):
    # grab correct filepath
    dir_main = os.getcwd()
    dir_npz = dir_main + "/npz_files/"

    # Global generator settings
    NAME_TAG = "RNN_" + "S1" + "A" + str(n_ca)
    n_timesteps = (1 + n_ca)
    n_features = 10
    n_cluster = 1 + n_ca

    # Input feature shape
    n_samples = 0
    ary_cluster_positions = RootParser.events["RecoClusterPositions.position"].array()
    for j in range(RootParser.events_entries):
        counter_scatterer = 0
        counter_absorber = 0
        for tvec in ary_cluster_positions[j]:
            if 150.0 - 20.8 / 2 < tvec.x < 150.0 + 20.8 / 2:
                counter_scatterer += 1
            if 270.0 - 46.8 / 2 < tvec.x < 270.0 + 46.8 / 2:
                counter_absorber += 1

        # filter condition
        if counter_scatterer == 1 and counter_absorber > 0:
            n_samples += 1
    print(n_samples, "valid events found")

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
    idx_sample = 0
    for i, event in enumerate(RootParser.iterate_events(n=None)):

        # get indices of clusters sorted by position in reverse order
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=False)
        
        if len(idx_scatterer) > 1:
            continue

        idx = idx_scatterer[0]
        ary_features[idx_sample, 0, :] = [event.RecoClusterEntries[idx],
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
            ary_features[idx_sample, (j + 1), :] = [event.RecoClusterEntries[idx],
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
        ary_targets_clas[idx_sample] = event.compton_tag * 1
        ary_targets_reg1[idx_sample, :] = np.array([event.target_energy_e, event.target_energy_p])
        ary_targets_reg2[idx_sample, :] = np.array([event.target_position_e.x,
                                                    event.target_position_e.y,
                                                    event.target_position_e.z,
                                                    event.target_position_p.x,
                                                    event.target_position_p.y,
                                                    event.target_position_p.z])
        ary_targets_reg3[idx_sample] = event.target_angle_theta

        # write weighting
        ary_w[idx_sample] = 1.0

        # write meta data
        ary_meta[idx_sample, :] = [event.EventNumber,
                                   event.MCEnergy_Primary,
                                   event.MCPosition_source.z,
                                   event.MCPosition_source.y,
                                   event.RecoClusterEnergies_values[idx_scatterer[0]],
                                   np.sum(event.RecoClusterEnergies_values[idx_absorber])]

        idx_sample += 1

    # save final output file
    str_savefile = dir_target + RootParser.file_name + "_" + NAME_TAG + ".npz"

    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=ary_features,
                            targets_clas=ary_targets_clas,
                            targets_reg1=ary_targets_reg1,
                            targets_reg2=ary_targets_reg2,
                            targets_reg3=ary_targets_reg3,
                            weights=ary_w,
                            meta=ary_meta)
