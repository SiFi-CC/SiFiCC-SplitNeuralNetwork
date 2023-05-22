import numpy as np
import os


def load(Root,
         n=None):
    # define dataset name
    dataset_name = "DenseSiPM"
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

    # write files for storage
    file_A = open(path + "/" + dataset_name + "_A.txt", "w")
    file_graph_indicator = open(
        path + "/" + dataset_name + "_graph_indicator.txt", "w")
    file_graph_labels = open(path + "/" + dataset_name + "_graph_labels.txt",
                             "w")
    file_node_attributes = open(
        path + "/" + dataset_name + "_node_attributes.txt", "w")
    file_graph_attributes = open(
        path + "/" + dataset_name + "_graph_attributes.txt", "w")

    # main iteration over root file
    for i, event in enumerate(Root.iterate_events(n=n)):
        # coincidence check
        n_scatterer = 0
        n_absorber = 0
        for j in range(len(event.SiPM_id)):
            if 150.0 - 14.0 / 2.0 < event.SiPM_position[j].x < 150.0 + 14.0 / 2.0:
                n_scatterer += 1
            if 270.0 - 30.0 / 2.0 < event.SiPM_position[j].x < 270.0 + 30.0 / 2.0:
                n_absorber += 1

        if n_scatterer == 0 or n_absorber == 0:
            continue

        # get the number of triggered sipms
        n_sipm = len(event.SiPM_id)
        for j in range(n_sipm):
            x, y, z = event.sipm_id_to_position(event.SiPM_id[j])
            file_A.write(str(x) + "," + str(y) + "," + str(z) + "\n")

            file_graph_indicator.write(str(i) + "\n")
            file_node_attributes.write(str(int(event.SiPM_qdc[j])) + "," +
                                       str(event.SiPM_triggertime[j]) + "\n")

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

    # verbose
    print("Dataset generated at: " + path)
    print("Name: ", dataset_name)
    print("Events: ", n_samples)
