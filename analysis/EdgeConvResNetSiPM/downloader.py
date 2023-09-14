import numpy as np
import os


def load(RootParser,
         path="",
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
        n           (int or None): Number of events sampled from root file

    Return:
         None
    """

    # define dataset name
    dataset_name = "GraphSiPM"
    dataset_name += "_" + RootParser.file_name

    # grab correct filepath, generate dataset in target directory.
    if path == "":
        path = "/net/scratch_g4rt1/fenger/datasets/"
    path = os.path.join(path, "SiFiCCNN_GraphSiPM", dataset_name)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs is needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    print("Counting number of graphs to be created")
    if n is None:
        n = RootParser.events_entries

    k_graphs = 0
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(RootParser.iterate_events(n=n)):
        idx_scatterer, idx_absorber = event.sort_sipm_by_module()
        if len(idx_scatterer) >= 1 and len(idx_absorber) >= 1:
            k_graphs += 1
            n_nodes += len(event.SiPM_id)
            m_edges += len(event.SiPM_id) * len(event.SiPM_id)
    print("Number of Graphs to be created: ", k_graphs)
    print("Total number of nodes to be created: ", n_nodes)

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(m_edges, 2), dtype=np.int)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int)
    ary_graph_labels = np.zeros(shape=(k_graphs,), dtype=np.bool)
    ary_node_attributes = np.zeros(shape=(n_nodes, 5), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(k_graphs, 8), dtype=np.float32)
    # meta data
    ary_pe = np.zeros(shape=(k_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(k_graphs,), dtype=np.float32)

    graph_id = 0
    node_id = 0
    edge_id = 0
    for i, event in enumerate(RootParser.iterate_events(n=n)):
        # get number of sipm's per module
        idx_scatterer, idx_absorber = event.sort_sipm_by_module()
        # exception
        if len(idx_scatterer) < 1 and len(idx_absorber) < 1:
            continue

        n_sipm = len(event.SiPM_id)
        for j in range(n_sipm):
            for k in range(n_sipm):
                ary_A[edge_id, :] = [node_id, node_id - j + k]
                edge_id += 1

            ary_graph_indicator[node_id] = graph_id
            ary_node_attributes[node_id, :] = [event.SiPM_position.x[j],
                                               event.SiPM_position.y[j],
                                               event.SiPM_position.z[j],
                                               event.SiPM_triggertime[j],
                                               event.SiPM_qdc[j]]

            node_id += 1

        # grab target labels and attributes
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[graph_id] = distcompton_tag
        ary_graph_attributes[graph_id, :] = [target_energy_e,
                                             target_energy_p,
                                             target_position_e.x,
                                             target_position_e.y,
                                             target_position_e.z,
                                             target_position_p.x,
                                             target_position_p.y,
                                             target_position_p.z]

        ary_pe[graph_id] = event.MCEnergy_Primary
        ary_sp[graph_id] = event.MCPosition_source.z
        graph_id += 1

    np.save(path + "/" + dataset_name + "_A.npy", ary_A)
    np.save(path + "/" + dataset_name + "_graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + dataset_name + "_graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + dataset_name + "_node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + dataset_name + "_graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + dataset_name + "_graph_pe.npy", ary_pe)
    np.save(path + "/" + dataset_name + "_graph_sp.npy", ary_sp)
