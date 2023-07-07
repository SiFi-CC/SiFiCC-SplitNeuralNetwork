import numpy as np
import os


def load(RootParser,
         path="",
         sx=4,
         ax=6,
         n=None):
    """
    Script to generate a dataset in graph basis.

    Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array size, one to read the
    data. Final data is stored as npy files, separated by their usage.

    Args:
        RootParser  (root Object): root object containing root file
        path        (str): destination path, if not given it will default to
                           scratch_g4rt1
        sx          (int): Number of maximum scatterer cluster used
        ax          (int): Number of maximum absorber cluster used
        n           (int): Number of events sampled from root file

    Return:
         None
    """

    # define dataset name
    dataset_name = "DenseCluster"
    dataset_name += "S" + str(sx) + "A" + str(ax)
    dataset_name += "_" + RootParser.file_name

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/dataset/$dataset_name$/"

    # grab correct filepath, generate dataset in target directory.
    if path == "":
        path = "/net/scratch_g4rt1/fenger/datasets/"
    path = os.path.join(path, "SiFiCCNN_DenseCluster", dataset_name)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # determine the number of events in the final dataset
    if n is None:
        n_samples = RootParser.events_entries
    else:
        n_samples = n

    # feature dimension
    n_timesteps = (sx + ax)
    n_features = 10
    print("Number of events to be created: ", n_samples)
    print("Event features: ", n_features)
    print("Event time-steps (Number of clusters): ", n_timesteps)

    # create empty arrays for storage
    sample_features = np.zeros(shape=(n_samples, n_timesteps, n_features), dtype=np.float32)
    sample_labels = np.zeros(shape=(n_samples,), dtype=np.int)
    sample_attributes = np.zeros(shape=(n_samples, 8), dtype=np.float32)
    sample_pe = np.zeros(shape=(n_samples,), dtype=np.float32)
    sample_sp = np.zeros(shape=(n_samples,), dtype=np.float32)

    # main iteration over root file
    index = 0
    for i, event in enumerate(RootParser.iterate_events(n=n_samples)):

        # get indices of clusters sorted by position in reverse order
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(
            use_energy=False)

        for j, idx in enumerate(np.flip(idx_scatterer)):
            if j >= sx:
                break
            sample_features[index, j, :] = [event.RecoClusterEntries[idx],
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
            sample_features[index, (j + sx), :] = [event.RecoClusterEntries[idx],
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
        sample_labels[index] = event.compton_tag * 1
        sample_attributes[index, :] = np.array([event.target_energy_e,
                                                event.target_energy_p,
                                                event.target_position_e.x,
                                                event.target_position_e.y,
                                                event.target_position_e.z,
                                                event.target_position_p.x,
                                                event.target_position_p.y,
                                                event.target_position_p.z])

        # write meta data
        sample_sp[index] = event.MCPosition_source.z
        sample_pe[index] = event.MCEnergy_Primary

        index += 1

    # save data to compressed numpy archives
    np.save(path + "/" + dataset_name + "_sample_features.npy", sample_features)
    np.save(path + "/" + dataset_name + "_sample_labels.npy", sample_labels)
    np.save(path + "/" + dataset_name + "_sample_attributes.npy", sample_attributes)
    np.save(path + "/" + dataset_name + "_sample_pe.npy", sample_pe)
    np.save(path + "/" + dataset_name + "_sample_sp.npy", sample_sp)
