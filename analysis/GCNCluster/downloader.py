import numpy as np
import os


def load(RootParser,
         path="",
         energy_cut=None,
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
        energy_cut  (float): energy cut applied to deposited energy
        n           (int): Number of events sampled from root file

    Return:
         None
    """

    # define dataset name
    dataset_name = "GraphCluster"
    if energy_cut is not None:
        dataset_name += "_ECUT" + str(energy_cut).replace(".", "DOT") + "MEV"
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
    # Energy cut (if needed) applied
    print("Counting number of graphs to be created")
    if n is None:
        n_samples = RootParser.events_entries
    else:
        n_samples = n

    if energy_cut is not None:
        n_graphs = 0
        n_nodes = 0
        m_edges = 0
        for i, event in enumerate(RootParser.iterate_events(n=n_samples)):
            if np.sum(event.RecoClusterEnergies_values) > energy_cut:
                n_graphs += 1
                n_nodes += len(event.RecoClusterEntries)
                m_edges += len(event.RecoClusterEntries) * len(event.RecoClusterEntries)
    else:
        if n is None:
            n_graphs = RootParser.events_entries
        else:
            n_graphs = n
        n_nodes = 0
        m_edges = 0
        for i, event in enumerate(RootParser.iterate_events(n=n_graphs)):
            n_nodes += len(event.RecoClusterEntries)
            m_edges += len(event.RecoClusterEntries) * len(event.RecoClusterEntries)
    print("Number of Graphs to be created: ", n_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Graph features: ", 10)
    print("Graph features: ", 3)

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int)
    ary_graph_labels = np.zeros(shape=(n_graphs,), dtype=np.bool)
    ary_node_attributes = np.zeros(shape=(n_nodes, 10), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(n_graphs, 8), dtype=np.float32)
    ary_edge_attributes = np.zeros(shape=(m_edges, 3), dtype=np.float32)
    # meta data
    ary_ep = np.zeros(shape=(n_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(n_graphs,), dtype=np.float32)

    node_id = 0
    edge_id = 0
    for i, event in enumerate(RootParser.iterate_events(n=n_graphs)):
        # get number of cluster
        n_cluster = int(len(event.RecoClusterEntries))
        # idx_scat, idx_abs = event.sort_clusters_by_module()
        for j in range(n_cluster):
            for k in range(n_cluster):
                """
                if j in idx_abs and k in idx_scat:
                    continue
                """
                ary_A[edge_id, :] = [node_id, node_id - j + k]

                # exception for self loops
                if j != k:
                    # grab edge features
                    r, phi, theta = event.get_edge_features(j, k)
                else:
                    r, phi, theta = 0, 0, 0

                ary_edge_attributes[edge_id, :] = [r, phi, theta]
                edge_id += 1

            ary_graph_indicator[node_id] = i
            ary_node_attributes[node_id, :] = [event.RecoClusterEntries[j],
                                               event.RecoClusterTimestamps_relative[j],
                                               event.RecoClusterEnergies_values[j],
                                               event.RecoClusterEnergies_uncertainty[j],
                                               event.RecoClusterPosition.x[j],
                                               event.RecoClusterPosition.y[j],
                                               event.RecoClusterPosition.z[j],
                                               event.RecoClusterPosition_uncertainty.x[j],
                                               event.RecoClusterPosition_uncertainty.y[j],
                                               event.RecoClusterPosition_uncertainty.z[j]]
            node_id += 1

        ary_graph_labels[i] = event.compton_tag
        ary_graph_attributes[i, :] = [event.target_energy_e,
                                      event.target_energy_p,
                                      event.target_position_e.x,
                                      event.target_position_e.y,
                                      event.target_position_e.z,
                                      event.target_position_p.x,
                                      event.target_position_p.y,
                                      event.target_position_p.z]

        ary_ep[i] = event.MCEnergy_Primary
        ary_sp[i] = event.MCPosition_source.z

    np.save(path + "/" + dataset_name + "_A.npy", ary_A)
    np.save(path + "/" + dataset_name + "_graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + dataset_name + "_graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + dataset_name + "_node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + dataset_name + "_graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + dataset_name + "_edge_attributes.npy", ary_edge_attributes)
    np.save(path + "/" + dataset_name + "_graph_ep.npy", ary_ep)
    np.save(path + "/" + dataset_name + "_graph_sp.npy", ary_sp)
