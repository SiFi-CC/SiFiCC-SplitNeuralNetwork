import numpy as np
import os


def sificcnn_dense_sxax(Root,
                        sx=4,
                        ax=6,
                        n=None):
    # definitions
    dataset_name = Root.file_name + "_SiFiCCNNDense"
    dataset_name += "DNN_" + "S" + str(sx) + "A" + str(ax)
    n_features = 10 * (sx + ax)

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/dataset/$dataset_name$/"
    dir_main = os.getcwd()
    dir_datasets = dir_main + "/dataset/"
    dir_final = dir_datasets + "/SiFiCCNN/" + dataset_name + "/"

    if not os.path.isdir(dir_final):
        os.makedirs(dir_final, exist_ok=True)

    # Input feature shape
    if n is None:
        n_samples = Root.events_entries
    else:
        n_samples = n

    print("Input feature shape: ")
    print("Samples: ", n_samples)
    print("Features: ", n_features)

    # create empty arrays for storage
    features = np.zeros(shape=(n_samples, n_features), dtype=np.float32)
    targets_clas = np.zeros(shape=(n_samples,), dtype=np.float32)
    targets_energy = np.zeros(shape=(n_samples, 2), dtype=np.float32)
    targets_position = np.zeros(shape=(n_samples, 6), dtype=np.float32)
    targets_theta = np.zeros(shape=(n_samples,), dtype=np.float32)
    meta = np.zeros(shape=(n_samples, 6), dtype=np.float32)

    # main iteration over root file
    for i, event in enumerate(Root.iterate_events(n=n)):

        # get indices of clusters sorted by position in reverse order
        idx_scatterer, idx_absorber = event.sort_clusters_by_module(
            use_energy=False)

        for j, idx in enumerate(np.flip(idx_scatterer)):
            if j >= sx:
                break
            features[i, j * 10: (j * 10) + 10] = [
                event.RecoClusterEntries[idx],
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
            features[i, (j + sx) * 10: ((j + sx) * 10) + 10] = [
                event.RecoClusterEntries[idx],
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
        targets_clas[i] = event.compton_tag * 1
        targets_energy[i, :] = np.array([event.MCEnergy_e, event.MCEnergy_p])
        targets_position[i, :] = np.array([event.MCPosition_e_first.x,
                                           event.MCPosition_e_first.y,
                                           event.MCPosition_e_first.z,
                                           event.MCPosition_p_first.x,
                                           event.MCPosition_p_first.y,
                                           event.MCPosition_p_first.z])
        targets_theta[i] = event.theta_dotvec

        # write meta data
        meta[i, :] = [event.EventNumber,
                      event.MCEnergy_Primary,
                      event.MCPosition_source.x,
                      event.MCPosition_source.y,
                      event.MCPosition_source.z]

    # save final output file
    str_savefile = dir_final + dataset_name + ".npz"

    with open(str_savefile, 'wb') as f_output:
        np.savez_compressed(f_output,
                            features=features,
                            targets_clas=targets_clas,
                            targets_energy=targets_energy,
                            targets_position=targets_position,
                            targets_thet=targets_theta,
                            meta=meta)


def sificcnn_graph(Root, n=None):
    """
    Script to generate a dataset for spektral package usage.

    Args:
        RootParser  (root Object): root object containing root file
        n                   (int): Number of events sampled from root file

    Return:
         None
    """

    dataset_name = Root.file_name + "_SiFiCCCluster"

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/dataset/$dataset_name$/"
    dir_main = os.getcwd()
    dir_datasets = dir_main + "/dataset/"
    dir_final = dir_datasets + "/SiFiCCdatasets/" + dataset_name + "/"

    if not os.path.isdir(dir_final):
        os.makedirs(dir_final, exist_ok=True)

    # grab dimensions from root file for debugging
    n_graph_features = 10
    n_edge_features = 0
    n_samples = Root.events_entries
    print("\n# Input feature shape: ")
    print("Samples: ", n_samples)
    print("Graph features: ", n_graph_features)
    print("Graph features: ", n_edge_features)

    file_A = open(dir_final + dataset_name + "_A.txt", "w")
    file_graph_indicator = open(
        dir_final + dataset_name + "_graph_indicator.txt", "w")
    file_graph_labels = open(dir_final + dataset_name + "_graph_labels.txt",
                             "w")
    file_node_attributes = open(
        dir_final + dataset_name + "_node_attributes.txt", "w")
    file_graph_attributes = open(
        dir_final + dataset_name + "_graph_attributes.txt", "w")
    file_edge_attributes = open(
        dir_final + dataset_name + "_edge_attributes.txt", "w")

    node_id = 0
    edge_id = 0
    for i, event in enumerate(Root.iterate_events(n=n)):
        # get number of cluster
        n_cluster = int(len(event.RecoClusterEntries))
        idx_scat, idx_abs = event.sort_clusters_by_module()
        for j in range(n_cluster):
            for k in range(n_cluster):
                if j in idx_abs and k in idx_scat:
                    continue

                file_A.write(
                    str(node_id) + ", " + str(node_id - j + k) + "\n")

                # exception for self loops
                if j != k:
                    # grab edge features
                    r, phi, theta = event.get_edge_features(j, k)
                else:
                    r, phi, theta = 0, 0, 0
                file_edge_attributes.write(
                    str(r) + "," + str(phi) + "," + str(theta) + "\n"
                )

            file_graph_indicator.write(str(i) + "\n")
            file_node_attributes.write(
                str(float(event.RecoClusterEntries[j])) + "," +
                str(float(event.RecoClusterTimestamps_relative[j])) + "," +
                str(float(event.RecoClusterEnergies_values[j])) + "," +
                str(float(event.RecoClusterEnergies_uncertainty[j])) + "," +
                str(float(event.RecoClusterPosition.x[j])) + "," +
                str(float(event.RecoClusterPosition.y[j])) + "," +
                str(float(event.RecoClusterPosition.z[j])) + "," +
                str(float(
                    event.RecoClusterPosition_uncertainty.x[j])) + "," +
                str(float(
                    event.RecoClusterPosition_uncertainty.y[j])) + "," +
                str(float(
                    event.RecoClusterPosition_uncertainty.z[j])) + "\n")
            node_id += 1

        file_graph_labels.write(str(event.compton_tag * 1) + "\n")
        file_graph_attributes.write(str(event.target_energy_e) + "," +
                                    str(event.target_energy_p) + "," +
                                    str(event.target_position_e.x) + "," +
                                    str(event.target_position_e.y) + "," +
                                    str(event.target_position_e.z) + "," +
                                    str(event.target_position_p.x) + "," +
                                    str(event.target_position_p.y) + "," +
                                    str(event.target_position_p.z) + "\n")

    file_A.close()
    file_graph_labels.close()
    file_graph_indicator.close()
    file_node_attributes.close()
    file_graph_attributes.close()
