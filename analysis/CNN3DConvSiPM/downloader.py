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
    dataset_name = "ConvSiPM"
    dataset_name += "_" + RootParser.file_name

    # grab correct filepath, generate dataset in target directory.
    if path == "":
        path = "/net/scratch_g4rt1/fenger/datasets/"
    path = os.path.join(path, "SiFiCCNN_ConvSiPM", dataset_name)
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

    # Pre-determine the final array size.
    # Total number of graphs is needed (n samples)
    # Total number of nodes (Iteration over root file needed)
    print("Counting number of graphs to be created")
    if n is None:
        n_graphs = RootParser.events_entries
    else:
        n_graphs = n
    n_nodes = 0
    m_edges = 0
    for i, event in enumerate(RootParser.iterate_events(n=n_graphs)):
        n_nodes += len(event.SiPM_id)
        m_edges += len(event.SiPM_id) * len(event.SiPM_id)
    print("Number of Graphs to be created: ", n_graphs)
    print("Total number of nodes to be created: ", n_nodes)

    # creating final arrays
    # datatypes are chosen for minimal size possible (duh)
    ary_A = np.zeros(shape=(n_nodes, 3), dtype=np.int)
    ary_graph_indicator = np.zeros(shape=(n_nodes,), dtype=np.int)
    ary_graph_labels = np.zeros(shape=(n_graphs,), dtype=np.bool)
    ary_node_attributes = np.zeros(shape=(n_nodes, 2), dtype=np.float32)
    ary_graph_attributes = np.zeros(shape=(n_graphs, 8), dtype=np.float32)
    # meta data
    ary_ep = np.zeros(shape=(n_graphs,), dtype=np.float32)
    ary_sp = np.zeros(shape=(n_graphs,), dtype=np.float32)

    node_id = 0
    for i, event in enumerate(RootParser.iterate_events(n=n_graphs)):
        # get number of triggered SiPMs
        n_sipm = int(len(event.SiPM_id))
        for j in range(n_sipm):
            x, y, z = event.sipm_id_to_position(event.SiPM_id[j])
            ary_A[node_id, :] = [x, y, z]

            ary_graph_indicator[node_id] = i
            ary_node_attributes[node_id, :] = [event.SiPM_qdc[j],
                                               event.SiPM_triggertime[j]]
            node_id += 1

        # grab target labels and attributes
        distcompton_tag = event.get_distcompton_tag()
        target_energy_e, target_energy_p = event.get_target_energy()
        target_position_e, target_position_p = event.get_target_position()
        ary_graph_labels[i] = distcompton_tag * 1
        ary_graph_attributes[i, :] = [target_energy_e,
                                      target_energy_p,
                                      target_position_e.x,
                                      target_position_e.y,
                                      target_position_e.z,
                                      target_position_p.x,
                                      target_position_p.y,
                                      target_position_p.z]

        ary_ep[i] = event.MCEnergy_Primary
        ary_sp[i] = event.MCPosition_source.z

    np.save(path + "/" + dataset_name + "_A.npy", ary_A)
    np.save(path + "/" + dataset_name + "_graph_indicator.npy", ary_graph_indicator)
    np.save(path + "/" + dataset_name + "_graph_labels.npy", ary_graph_labels)
    np.save(path + "/" + dataset_name + "_node_attributes.npy", ary_node_attributes)
    np.save(path + "/" + dataset_name + "_graph_attributes.npy", ary_graph_attributes)
    np.save(path + "/" + dataset_name + "_graph_pe.npy", ary_ep)
    np.save(path + "/" + dataset_name + "_graph_sp.npy", ary_sp)
