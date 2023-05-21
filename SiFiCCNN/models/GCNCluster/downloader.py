import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------------------

def load(Root, n=None):
    """
    Script to generate a dataset for spektral package usage.

    Args:
        RootParser  (root Object): root object containing root file
        n                   (int): Number of events sampled from root file

    Return:
         None
    """

    # define dataset name
    dataset_name = "GraphCluster"
    dataset_name += "_" + Root.file_name

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/dataset/$dataset_name$/"

    # get current path, go two subdirectories higher
    path = os.path.dirname(os.path.abspath(__file__))
    for i in range(3):
        path = os.path.dirname(path)
    path = os.path.join(path, "datasets", "SiFiCCNN", dataset_name)

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Input feature shape
    if n is None:
        n_samples = Root.events_entries
    else:
        n_samples = n

    # grab dimensions from root file for debugging
    n_graph_features = 10
    n_edge_features = 3
    n_samples = Root.events_entries
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
