import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import radius_neighbors_graph


def download_adj():
    # TODO: add asserts for complete adj matrix
    # TODO: improve loading of correct root file

    from SiFiCCNN.root import Root, RootFiles

    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    RootParser = Root.Root(
        path_main + RootFiles.FinalDetectorVersion_RasterCoupling_OPM_38e8protons_local)

    nIDs = 7 * 2 * 16 + 16 * 2 * 16 - 1 + 1
    ary_id_pos = np.zeros(shape=(nIDs, 3), dtype=np.float32)

    for i, event in enumerate(RootParser.iterate_events(n=10000)):
        for j in range(len(event.SiPM_id)):
            if ary_id_pos[event.SiPM_id[j], 0] == 0:
                ary_id_pos[event.SiPM_id[j], :] = [event.SiPM_position[j].x,
                                                   event.SiPM_position[j].y,
                                                   event.SiPM_position[j].z]

    # check if sipm position array is filled properly
    ary_id_pos_sum = np.sum(ary_id_pos, axis=1)
    print("Array % filled: {:.1f}%".format((1 - (np.sum((ary_id_pos_sum == 0) * 1) / nIDs)) * 100))
    np.save("sipm_positions.npy", ary_id_pos)


def load_adj():
    ary_positions = np.load("sipm_positions.npy")

    # modify y position to allow fibre connections
    y_mod = 15.0
    ary_positions[:, 1] /= 51.0
    ary_positions[:, 1] *= y_mod / 2

    A = radius_neighbors_graph(ary_positions, radius=6.0)
    A = np.maximum(A, A.T)

    return A
