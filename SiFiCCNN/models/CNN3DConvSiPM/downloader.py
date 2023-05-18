import numpy as np
import os


def load(Root,
         padding=2,
         gap_padding=4,
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

    # create empty arrays for storage
    dimx = 12
    dimy = 2
    dimz = 32
    features = np.zeros(shape=(n_samples,
                               dimx + 2 * padding + gap_padding,
                               dimy + 2 * padding,
                               dimz + 2 * padding,
                               2))

    targets_clas = np.zeros(shape=(n_samples,), dtype=np.float32)
    targets_energy = np.zeros(shape=(n_samples, 2), dtype=np.float32)
    targets_position = np.zeros(shape=(n_samples, 6), dtype=np.float32)
    targets_theta = np.zeros(shape=(n_samples,), dtype=np.float32)
    pe = np.zeros(shape=(n_samples,), dtype=np.float32)
    sp = np.zeros(shape=(n_samples,), dtype=np.float32)

    # main iteration over root file
    for i, event in enumerate(Root.iterate_events(n=n)):
        features[i, :, :, :, :] = event.get_sipm_feature_map(padding,
                                                          gap_padding)

        # target: ideal compton events tag
        targets_clas[i] = event.compton_tag * 1
        targets_energy[i, :] = np.array([event.target_energy_e,
                                         event.target_energy_p])
        targets_position[i, :] = np.array([event.target_position_e.x,
                                           event.target_position_e.y,
                                           event.target_position_e.z,
                                           event.target_position_p.x,
                                           event.target_position_p.y,
                                           event.target_position_p.z])
        targets_theta[i] = event.target_angle_theta

        # write meta data
        sp[i] = event.MCPosition_source.z
        pe[i] = event.MCEnergy_Primary

    # save data to compressed numpy archives
    np.savez_compressed(path + "/features", features)
    np.savez_compressed(path + "/targets_clas", targets_clas)
    np.savez_compressed(path + "/targets_energy", targets_energy)
    np.savez_compressed(path + "/targets_position", targets_position)
    np.savez_compressed(path + "/targets_theta", targets_theta)
    np.savez_compressed(path + "/primary_energy", pe)
    np.savez_compressed(path + "/source_position_z", sp)

    # verbose
    print("Dataset generated at: " + path)
    print("Name: ", dataset_name)
    print("Events: ", n_samples)
    print("Feature dimension: (None, {},{}, {}, 2)".format(features.shape[1],
                                                           features.shape[2],
                                                           features.shape[3]))


