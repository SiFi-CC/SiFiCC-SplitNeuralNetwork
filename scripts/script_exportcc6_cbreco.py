####################################################################################################
# CC6 Export CB-Reco script
#
# This script main task is to take the cut-based reconstruction of a root file and export it to
# the needed format for the CC6 image reconstruction. Two sets of reconstruction will be exported.
#   - Cut-based reconstruction identified events
#   - True Compton tagged events (Control for tagging)
#
####################################################################################################

import numpy as np
import os
import sys
import argparse

from SiFiCCNN.root import RootParser
from SiFiCCNN.ImageReconstruction import IRExport


def main(root_file, tagging):
    print("LOADING: {}".format(root_file))

    # Find correct parent path of repository
    # iterate backwards in directories until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    path_root = path_main + "/root_files/"

    # create RootParser object from stated root file
    root_parser = RootParser.RootParser(path_root + root_file)

    # generate mask based on tagging used (mask first column in ary)
    # iteration over root file
    ary_reco = np.zeros(shape=(root_parser.events_entries, 9), dtype=np.float32)
    for i, event in enumerate(root_parser.iterate_events(n=None)):
        target_tag = 0
        if tagging == "identified":
            target_tag = 1 if event.Identified != 0 else 0
        if tagging == "distcompton":
            target_tag = event.get_distcompton_tag() * 1
        if tagging == "distcompton_awal":
            target_tag = event.get_distcompton_tag_awal() * 1

        target_energy_e, target_energy_p = event.get_reco_energy()
        target_position_e, target_position_p = event.get_reco_position()

        ary_reco[i, 0] = target_tag
        ary_reco[i, 1] = target_energy_e
        ary_reco[i, 2] = target_energy_p
        ary_reco[i, 3] = target_position_e.x
        ary_reco[i, 4] = target_position_e.y
        ary_reco[i, 5] = target_position_e.z
        ary_reco[i, 6] = target_position_p.x
        ary_reco[i, 7] = target_position_p.y
        ary_reco[i, 8] = target_position_p.z

    # apply mask
    print("Apply mask: ({})".format(tagging))
    print("Before: {}".format(len(ary_reco)))
    ary_reco = ary_reco[ary_reco[:, 0] == 1, :]
    print("After : {}".format(len(ary_reco)))

    # generate filename
    file_name = "CC6IR_CBRECO_"
    if tagging == "identified":
        file_name += "IDENTIFIED_"
    if tagging == "distcompton":
        file_name += "DISTCOMPTON_"
    if tagging == "distcompton_awal":
        file_name += "DISTCOMPTONAWAL_"
    file_name += root_parser.file_name

    # generate CC6 export
    IRExport.export_CC6(ary_e=ary_reco[:, 1],
                        ary_p=ary_reco[:, 2],
                        ary_ex=ary_reco[:, 3],
                        ary_ey=ary_reco[:, 4],
                        ary_ez=ary_reco[:, 5],
                        ary_px=ary_reco[:, 6],
                        ary_py=ary_reco[:, 7],
                        ary_pz=ary_reco[:, 8],
                        ary_theta=None,
                        filename=file_name,
                        veto=True,
                        verbose=1)


if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--rf", type=str, help="Target root file")
    parser.add_argument("--tag", type=str, help="Type of event selection")
    args = parser.parse_args()

    if args.rf is None:
        print("Invalid --rf argument given!")
        sys.exit()

    if args.tag not in ["identified", "distcompton", "distcompton_awal"]:
        print("Invalid --tag argument given!")
        print("Possible choices: 'identified', 'distcompton', 'distcompton_awal'")
        sys.exit()

    ROOTFILE = "OptimisedGeometry_BP0mm_2e10protons_taggingv3.root"
    main(ROOTFILE, tagging=args.tag)

    # main(args.rf, use_tagging=args.tag)
