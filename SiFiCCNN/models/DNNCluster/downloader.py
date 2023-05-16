import numpy as np
import os


def load(Root,
         sx=4,
         ax=6,
         n=None):
    # define dataset name
    dataset_name = "DenseCluster"
    dataset_name += "_" + "S" + str(sx) + "A" + str(ax)
    dataset_name += "_" + Root.file_name

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/dataset/$dataset_name$/"

    # get current path, go two subdirectories higher
    path = os.path.dirname(os.path.abspath(__file__))
    for i in range(3):
        path = os.path.dirname(path)
    path = os.path.join(path, "datasets", "SiFiCCNN", dataset_name)

    # feature dimension
    n_timesteps = (sx + ax)
    n_features = 10

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Input feature shape
    if n is None:
        n_samples = Root.events_entries
    else:
        n_samples = n

    # create empty arrays for storage
    features = np.zeros(shape=(n_samples, n_timesteps, n_features),
                        dtype=np.float32)
    targets_clas = np.zeros(shape=(n_samples,), dtype=np.float32)
    targets_energy = np.zeros(shape=(n_samples, 2), dtype=np.float32)
    targets_position = np.zeros(shape=(n_samples, 6), dtype=np.float32)
    targets_theta = np.zeros(shape=(n_samples,), dtype=np.float32)
    pe = np.zeros(shape=(n_samples,), dtype=np.float32)
    sp = np.zeros(shape=(n_samples,), dtype=np.float32)

    # main iteration over root file
    for i, event in enumerate(Root.iterate_events(n=n)):

        # get indices of clusters sorted by position in reverse order
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(
            use_energy=False)

        for j, idx in enumerate(np.flip(idx_scatterer)):
            if j >= sx:
                break
            features[i, j, :] = [event.RecoClusterEntries[idx],
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
            if j >= ax:
                break
            features[i, (j + sx), :] = [event.RecoClusterEntries[idx],
                                        event.RecoClusterTimestamps_relative[
                                            idx],
                                        event.RecoClusterEnergies_values[idx],
                                        event.RecoClusterPosition[idx].x,
                                        event.RecoClusterPosition[idx].y,
                                        event.RecoClusterPosition[idx].z,
                                        event.RecoClusterEnergies_uncertainty[
                                            idx],
                                        event.RecoClusterPosition_uncertainty[
                                            idx].x,
                                        event.RecoClusterPosition_uncertainty[
                                            idx].y,
                                        event.RecoClusterPosition_uncertainty[
                                            idx].z]

        # target: ideal compton events tag
        targets_clas[i] = event.compton_tag * 1
        targets_energy[i, :] = np.array([event.target_energy_e,
                                         event.target_energy_p])
        targets_position[i, :] = np.array([event.target_position_e.x,
                                           event.target_position_e.y,
                                           event.target_position_e.z,
                                           event.target_position_p.x,
                                           event.target_position_p.y,
                                           event.target_position_p.z])
        targets_theta[i] = event.target_angle_theta

        # write meta data
        sp[i] = event.MCPosition_source.z
        pe[i] = event.MCEnergy_Primary

    # save data to compressed numpy archives
    np.savez_compressed(path + "/features", features)
    np.savez_compressed(path + "/targets_clas", targets_clas)
    np.savez_compressed(path + "/targets_energy", targets_energy)
    np.savez_compressed(path + "/targets_position", targets_position)
    np.savez_compressed(path + "/targets_theta", targets_theta)
    np.savez_compressed(path + "/primary_energy", pe)
    np.savez_compressed(path + "/source_position_z", sp)

    # verbose
    print("Dataset generated at: " + path)
    print("Name: ", dataset_name)
    print("Events: ", n_samples)
    print("Feature dimension: (None, {},{})".format(int(n_timesteps),
                                                    int(n_features)))
