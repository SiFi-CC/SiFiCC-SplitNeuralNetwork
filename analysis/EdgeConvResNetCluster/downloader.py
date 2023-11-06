import numpy as np
import os


def load(RootParser,
         path="",
         n=None,
         coordinate_system="CRACOW"):
    """
    Script to generate a dataset in graph basis. Inspired by the TUdataset "PROTEIN"

    Two iterations over the root file are needed: one to determine the array size, one to read the
    data. Final data is stored as npy files, separated by their usage.

    Args:
        RootParser  (root Object):  root object containing root file
        path        (str):          destination path, if not given it will default to
                                    scratch_g4rt1
        n           (int or None):  Number of events sampled from root file
        coordinate_system (str):    Coordinate system of the given root file, everything will be
                                    converted to Aachen coordinate system

    Return:
         None
    """

    # define dataset name, constructed from the given input and the final data structure
    # grab correct filepath, generate dataset in target directory
    base = "GraphSiPM"
    tag = "TEST"
    name_set = "{}_{}".format(base, RootParser.file_name)
    if tag != "":
        name_set += "_{}".format(tag)

    if path == "":
        path = "/net/scratch_g4rt1/fenger/datasets/"
    path = os.path.join(path, "SiFiCCNN_{}".format(base), name_set)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    print("Counting number of graphs to be created")
    if n is None:
        n = RootParser.events_entries

    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(RootParser.iterate_events(n=n)):
        k_graphs += 1
        n_nodes += len(event.RecoClusterEntries)
        m_edges += len(event.RecoClusterEntries) * len(event.RecoClusterEntries)
    print("Number of Graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)
    print("Graph features: ", 10)
    print("Graph features: ", 3)

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool)
    ary_node_attributes = np.zeros(shape=(n_nodes, 10), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    # ary_edge_attributes = np.zeros(shape=(m_edges, 6), dtype=np.float32)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)

    # main iteration over root file, containing beta coincidence check
    node_id = 0
    edge_id = 0
    for i, event in enumerate(RootParser.iterate_events(n=n)):
        # get number of cluster
        n_cluster = int(len(event.RecoClusterEntries))

        # coincidence check
        idx_scat, idx_abs = event.sort_clusters_by_module()
        if not (len(idx_scat) >= 1 and len(idx_abs) >= 1):
            continue

        for j in range(n_cluster):
            for k in range(n_cluster):
                """
                # self-loop exception
                if j in idx_abs and k in idx_scat:
                    continue
                """

                """
                # determine edge attributes
                if j != k:
                    # grab edge features in polar and cartesian representation
                    r, phi, theta = event.get_edge_features(j, k, cartesian=False)
                    dx, dy, dz = event.get_edge_features(j, k, cartesian=True)
                else:
                    r, phi, theta = 0, 0, 0
                    dx, dy, dz = 0, 0, 0
                ary_edge_attributes[edge_id, :] = [dx, dy, dz, r, phi, theta]
                """

                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            ary_graph_indicator[node_id] = i

            # exception for different coordinate systems
            if coordinate_system == "CRACOW":
                ary_node_attributes[node_id, :] = [event.RecoClusterEntries[j],
                                                   event.RecoClusterTimestamps_relative[j],
                                                   event.RecoClusterEnergies_values[j],
                                                   event.RecoClusterEnergies_uncertainty[j],
                                                   event.RecoClusterPosition.z[j],
                                                   -event.RecoClusterPosition.y[j],
                                                   event.RecoClusterPosition.x[j],
                                                   event.RecoClusterPosition_uncertainty.z[j],
                                                   event.RecoClusterPosition_uncertainty.y[j],
                                                   event.RecoClusterPosition_uncertainty.x[j]]
            if coordinate_system == "AACHEN":
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

        # grab target labels and attributes
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[i] = distcompton_tag * 1
        if coordinate_system == "CRACOW":
            ary_graph_attributes[i, :] = [target_energy_e,
                                          target_energy_p,
                                          target_position_e.z,
                                          -target_position_e.y,
                                          target_position_e.x,
                                          target_position_p.z,
                                          -target_position_p.y,
                                          target_position_p.x]
            ary_sp[i] = event.MCPosition_source.x
        if coordinate_system == "AACHEN":
            ary_graph_attributes[i, :] = [target_energy_e,
                                          target_energy_p,
                                          target_position_e.x,
                                          target_position_e.y,
                                          target_position_e.z,
                                          target_position_p.x,
                                          target_position_p.y,
                                          target_position_p.z]
            ary_sp[i] = event.MCPosition_source.z

        ary_pe[i] = event.MCEnergy_Primary

    np.save(path + "/" + name_set + "_A.npy", ary_A)
    np.save(path + "/" + name_set + "_graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + name_set + "_graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + name_set + "_node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + name_set + "_graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + name_set + "_graph_pe.npy", ary_pe)
    np.save(path + "/" + name_set + "_graph_sp.npy", ary_sp)
