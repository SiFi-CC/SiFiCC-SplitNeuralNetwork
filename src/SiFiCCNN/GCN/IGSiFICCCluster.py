import numpy as np
import os


# ----------------------------------------------------------------------------------------------------------------------

def gen_SiFiCCCluster(RootParser, n=None):
    """
    Script to generate a dataset for spektral package usage.

    Args:
        RootParser  (Root Object): Root object containing root file
        n                   (int): Number of events sampled from root file

    Return:
         None
    """

    dataset_name = RootParser.file_name + "_SiFiCCCluster"

    # grab correct filepath, generate dataset in target directory.
    # If directory doesn't exist, it will be created.
    # "MAIN/datasets/$dataset_name$/"
    dir_main = os.getcwd()
    dir_datasets = dir_main + "/datasets/"
    dir_final = dir_datasets + "/SiFiCCdatasets/" + dataset_name + "/"

    if not os.path.isdir(dir_final):
        os.makedirs(dir_final, exist_ok=True)

    # grab dimensions from root file for debugging
    n_graph_features = 10
    n_edge_features = 0
    n_samples = RootParser.events_entries
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
        dir_final + dataset_name + "_edfe_attributes.txt", "w")

    node_id = 0
    edge_id = 0
    for i, event in enumerate(RootParser.iterate_events(n=n)):
        # get number of cluster
        n_cluster = int(len(event.RecoClusterEntries))
        for j in range(n_cluster):
            for k in range(n_cluster):
                if j != k:
                    file_A.write(
                        str(node_id) + ", " + str(node_id - j + k) + "\n")

                    # grab edge features
                    r, phi, theta = event.get_edge_features(j, k)
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
