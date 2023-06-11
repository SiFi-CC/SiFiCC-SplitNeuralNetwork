import numpy as np
import os

from SiFiCCNN.root import Root
from SiFiCCNN.ImageReconstruction import IRExport


def export_cc6_cutbasedreco(RootParser,
                            energy_cut=None):
    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path

    path_root = path_main + "/root_files/"

    # generate target arrays
    cb_reco = np.zeros(shape=(RootParser.events_entries, 10), dtype=np.float32)
    cb_reco[:, 0] = RootParser.events["Identified"].array()
    cb_reco[:, 1] = RootParser.events["RecoEnergy_e"]["value"].array()
    cb_reco[:, 2] = RootParser.events["RecoEnergy_p"]["value"].array()
    cb_reco[:, 3] = RootParser.events["RecoPosition_e"]["position"].array().x
    cb_reco[:, 4] = RootParser.events["RecoPosition_e"]["position"].array().y
    cb_reco[:, 5] = RootParser.events["RecoPosition_e"]["position"].array().z
    cb_reco[:, 6] = RootParser.events["RecoPosition_p"]["position"].array().x
    cb_reco[:, 7] = RootParser.events["RecoPosition_p"]["position"].array().y
    cb_reco[:, 8] = RootParser.events["RecoPosition_p"]["position"].array().z

    # filter for identified events
    print("Apply Cut-Based Reco identification")
    print("Before: {}".format(len(cb_reco)))
    cb_reco = cb_reco[cb_reco[:, 0] != 0, :]
    print("After : {}".format(len(cb_reco)))

    # if energy cut is needed, create mask to apply it
    if energy_cut is not None:
        mask = np.zeros(shape=(RootParser.events_entries,), dtype=np.int)
        for i, event in enumerate(RootParser.iterate_events(n=None)):
            if np.sum(event.RecoClusterEnergies_values) > energy_cut:
                mask[i] = 1
        print("Apply energy cut ({:1f} MEV)".format(energy_cut))
        print("Before: {}".format(len(cb_reco)))
        cb_reco = cb_reco[mask == 1, :]
        print("After : {}".format(len(cb_reco)))

    # generate filename
    file_name = "CC6IR_CBRECO"
    if energy_cut is not None:
        file_name += "_ECUT" + str(energy_cut).replace(".", "DOT") + "_"
    file_name += RootParser.file_name

    # generate CC6 export
    IRExport.export_CC6(ary_e=cb_reco[:, 1],
                        ary_p=cb_reco[:, 2],
                        ary_ex=cb_reco[:, 3],
                        ary_ey=cb_reco[:, 4],
                        ary_ez=cb_reco[:, 5],
                        ary_px=cb_reco[:, 6],
                        ary_py=cb_reco[:, 7],
                        ary_pz=cb_reco[:, 8],
                        ary_theta=cb_reco[:, 9],
                        filename=file_name,
                        veto=False)
