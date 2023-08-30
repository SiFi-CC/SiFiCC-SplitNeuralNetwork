####################################################################################################
# Script Neural Network Prediction with CB Reconstruction
#
# Script for using neural network classification prediction and overloading them with cut-based
# reconstruction.
#
####################################################################################################

import numpy as np
import os

from SiFiCCNN.ImageReconstruction import IRExport
from SiFiCCNN.root import RootParser, RootFiles


def main():
    # defining hyper parameters
    RUN_NAME = "DNNCluster_S4A6"
    threshold = 0.5

    # Datasets used
    # Training file used for classification and regression training
    # Generated via an input generator, contain one Bragg-peak position
    DATASET_CONT = "DenseClusterS4A6_OptimisedGeometry_Continuous_2e10protons_taggingv3"
    DATASET_0MM = "DenseClusterS4A6_OptimisedGeometry_BP0mm_2e10protons_taggingv3"
    DATASET_5MM = "DenseClusterS4A6_OptimisedGeometry_BP5mm_4e9protons_taggingv3"

    ROOTFILE_CONT = RootFiles.onetoone_cont_taggingv2
    ROOTFILE_0MM = RootFiles.onetoone_BP0mm_taggingv2
    ROOTFILE_5MM = RootFiles.onetoone_BP5mm_taggingv2

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    path_results = path_main + "/results/" + RUN_NAME + "/"
    path_datasets = path_main + "/datasets/"
    path_root = path_main + "/root_files/"

    list_data_files = [DATASET_0MM, DATASET_5MM]
    list_root_files = [ROOTFILE_0MM, ROOTFILE_5MM]

    for i, file in enumerate(list_data_files):

        # load neural network prediction
        # os.chdir(path_results + file + "/")
        path_export = path_results + file + "/"

        npz_data = np.load(path_export + file + "_prediction.npz")
        nn_pred = npz_data["nn_pred"]

        # grab cut-based reconstruction from root file
        root_parser = RootParser.RootParser(path_root + list_root_files[i])
        for j, event in enumerate(root_parser.iterate_events(n=None)):
            reco_energy_e, reco_energy_p = event.get_reco_energy()
            reco_position_e, reco_position_p = event.get_reco_position()

            nn_pred[j, 1:] = [reco_energy_e,
                              reco_energy_p,
                              reco_position_e.x,
                              reco_position_e.y,
                              reco_position_e.z,
                              reco_position_p.x,
                              reco_position_p.y,
                              reco_position_p.z]

        # This is done this way cause y_scores gets a really dumb shape from tensorflow
        idx_clas_pos = [nn_pred[i, 0] > threshold for i in range(len(nn_pred))]

        # export to root file compatible with CC6 image reconstruction
        IRExport.export_CC6(ary_e=nn_pred[idx_clas_pos, 1],
                            ary_p=nn_pred[idx_clas_pos, 2],
                            ary_ex=nn_pred[idx_clas_pos, 3],
                            ary_ey=nn_pred[idx_clas_pos, 4],
                            ary_ez=nn_pred[idx_clas_pos, 5],
                            ary_px=nn_pred[idx_clas_pos, 6],
                            ary_py=nn_pred[idx_clas_pos, 7],
                            ary_pz=nn_pred[idx_clas_pos, 8],
                            filename=path_export + "CC6IR_NNPRED_CBRECO_" + file + "_theta" + str(
                                threshold).replace(".", ""),
                            verbose=1,
                            veto=True)


if __name__ == "__main__":
    main()
