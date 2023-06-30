import numpy as np
import os

from SiFiCCNN.root import RootParser
from SiFiCCNN.ImageReconstruction import IRExport


def main():
    """
    # parameter
    energy_cut = None
    """
    # Used root files
    ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_taggingv3.root"
    ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_taggingv3.root"

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path

    path_root = path_main + "/root_files/"

    # generate target arrays
    for file in [ROOT_FILE_BP0mm, ROOT_FILE_BP5mm]:
        root_parser = RootParser.RootParser(path_root + file)

        cb_reco = np.zeros(shape=(root_parser.events_entries, 10), dtype=np.float32)
        cb_reco[:, 0] = root_parser.events["Identified"].array()
        cb_reco[:, 1] = root_parser.events["RecoEnergy_e"]["value"].array()
        cb_reco[:, 2] = root_parser.events["RecoEnergy_p"]["value"].array()
        cb_reco[:, 3] = root_parser.events["RecoPosition_e"]["position"].array().x
        cb_reco[:, 4] = root_parser.events["RecoPosition_e"]["position"].array().y
        cb_reco[:, 5] = root_parser.events["RecoPosition_e"]["position"].array().z
        cb_reco[:, 6] = root_parser.events["RecoPosition_p"]["position"].array().x
        cb_reco[:, 7] = root_parser.events["RecoPosition_p"]["position"].array().y
        cb_reco[:, 8] = root_parser.events["RecoPosition_p"]["position"].array().z

        # filter for identified events
        print("Apply Cut-Based Reco identification")
        print("Before: {}".format(len(cb_reco)))
        cb_reco = cb_reco[cb_reco[:, 0] != 0, :]
        print("After : {}".format(len(cb_reco)))
        """"
        # if energy cut is needed, create mask to apply it
        if energy_cut is not None:
            mask = np.zeros(shape=(root_parser.events_entries,), dtype=np.int)
            for i, event in enumerate(root_parser.iterate_events(n=None)):
                if np.sum(event.RecoClusterEnergies_values) > energy_cut:
                    mask[i] = 1
            print("Apply energy cut ({:1f} MEV)".format(energy_cut))
            print("Before: {}".format(len(cb_reco)))
            cb_reco = cb_reco[mask == 1, :]
            print("After : {}".format(len(cb_reco)))
        """
        # generate filename
        file_name = "CC6IR_CBRECO"
        """
        if energy_cut is not None:
            file_name += "_ECUT" + str(energy_cut).replace(".", "DOT") + "_"
        """
        file_name += root_parser.file_name

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

        # Generate CBRECO files with True Compton tagged events only
        print("GENERATE DATASET WITH TRUE COMPTON TAGGED EVENTS ONLY\n")

        cb_reco = np.zeros(shape=(root_parser.events_entries, 10), dtype=np.float32)
        cb_reco[:, 0] = root_parser.events["Identified"].array()
        cb_reco[:, 1] = root_parser.events["RecoEnergy_e"]["value"].array()
        cb_reco[:, 2] = root_parser.events["RecoEnergy_p"]["value"].array()
        cb_reco[:, 3] = root_parser.events["RecoPosition_e"]["position"].array().x
        cb_reco[:, 4] = root_parser.events["RecoPosition_e"]["position"].array().y
        cb_reco[:, 5] = root_parser.events["RecoPosition_e"]["position"].array().z
        cb_reco[:, 6] = root_parser.events["RecoPosition_p"]["position"].array().x
        cb_reco[:, 7] = root_parser.events["RecoPosition_p"]["position"].array().y
        cb_reco[:, 8] = root_parser.events["RecoPosition_p"]["position"].array().z

        ary_mask = np.zeros(shape=(root_parser.events_entries,))
        for i, event in enumerate(root_parser.iterate_events(n=None)):
            # event.set_tags_awal()
            if event.compton_tag:
                ary_mask[i] = 1
            else:
                continue

        # filter for identified events
        print("Apply true compton tag identification")
        print("Before: {}".format(len(cb_reco)))
        cb_reco = cb_reco[ary_mask == 1, :]
        print("After : {}".format(len(cb_reco)))

        # generate filename
        file_name = "CC6IR_CBRECO_TRUECOMPTON_"
        file_name += root_parser.file_name

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


if __name__ == "__main__":
    main()
