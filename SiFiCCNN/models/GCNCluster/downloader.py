import numpy as np
import os


def load(RootParser,
         path="",
         n=None):
    """
    Script to generate a dataset in graph basis.

    Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array
    size, one to read the data. Final data is stored as npy files, separated by
    their usage.

    Args:
        RootParser  (root Object): root object containing root file
        path        (str): destination path, if not given it will default to
                           scratch_g4rt1
        n           (int): Number of events sampled from root file

    Return:
         None
    """

    # define dataset name
    dataset_name = "GraphCluster"
    dataset_name += "_" + RootParser.file_name

    # grab correct filepath, generate dataset in target directory.
    if path == "":
        path = "/net/scratch_g4rt1/fenger/datasets/"
    path = os.path.join(path, "SiFiCCNN_GraphCluster", dataset_name)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs is needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    if n is None:
        n_graphs = RootParser.events_entries
    else:
        n_graphs = n
    n_nodes = 0

    # grab dimensions from root file for debugging
    n_graph_features = 10
    n_edge_features = 3
    n_samples = RootParser.events_entries
    print("\n# Input feature shape: ")
    print("Samples: ", n_samples)
    print("Graph features: ", n_graph_features)
    print("Graph features: ", n_edge_features)

    file_A = open(path + "/" + dataset_name + "_A.txt", "w")
    file_graph_indicator = open(
        path + "/" + dataset_name + "_graph_indicator.txt", "w")
    file_graph_labels = open(path + "/" + dataset_name + "_graph_labels.txt",
                             "w")
    file_node_attributes = open(
        path + "/" + dataset_name + "_node_attributes.txt", "w")
    file_graph_attributes = open(
        path + "/" + dataset_name + "_graph_attributes.txt", "w")
    file_edge_attributes = open(
        path + "/" + dataset_name + "_edge_attributes.txt", "w")

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
