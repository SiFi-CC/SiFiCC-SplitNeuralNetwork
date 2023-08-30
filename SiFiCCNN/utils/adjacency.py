import numpy as np
import os
from sklearn.neighbors import radius_neighbors_graph


def gen_adj_pos_4to1():
    # TODO: add asserts for complete adj matrix
    # TODO: improve loading of correct root file

    from SiFiCCNN.root import RootParser, RootFiles

    # get current path, go two subdirectories higher
    path = os.getcwd()
    while True:
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
        path = os.path.abspath(os.path.join(path, os.pardir))
    path_root = path + "/root_files/"
    # define RootParser
    root_parser = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_taggingv2)

    nIDs = 7 * 2 * 16 + 16 * 2 * 16 - 1 + 1
    ary_id_pos = np.zeros(shape=(nIDs, 3), dtype=np.float32)

    for i, event in enumerate(root_parser.iterate_events(n=10000)):
        for j in range(len(event.SiPM_id)):
            if ary_id_pos[event.SiPM_id[j], 0] == 0:
                ary_id_pos[event.SiPM_id[j], :] = [event.SiPM_position[j].x,
                                                   event.SiPM_position[j].y,
                                                   event.SiPM_position[j].z]

    # check if sipm position array is filled properly
    ary_id_pos_sum = np.sum(ary_id_pos, axis=1)
    print("Array % filled: {:.1f}%".format((1 - (np.sum((ary_id_pos_sum == 0) * 1) / nIDs)) * 100))

    # write array to txt file
    np.savetxt(fname="adj_positions_4to1.txt", X=ary_id_pos)


def get_sparse_adj_4to1(path=""):
    ary_id_pos = np.loadtxt(path + "adj_positions_4to1.txt")

    # modify y position to allow fibre connections
    y_mod = 5.0
    ary_id_pos[:, 1] /= 51.0
    ary_id_pos[:, 1] *= y_mod / 2

    adj = radius_neighbors_graph(ary_id_pos, radius=6.0)

    return adj
