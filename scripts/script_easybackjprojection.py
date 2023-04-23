import numpy as np
import os
from src.SiFiCCNN.ImageReconstruction import IRBackprojection
from src.SiFiCCNN.Plotter import PTImageReconstruction

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"


# ---------------------------------------------------------------------------------------------------------------------

def get_ir_from_neuralnetwork(run_name,
                              generator_name,
                              f_sample_0mm,
                              f_sample_5mm,
                              figure_name):
    NAME_FILE1 = "OptimisedGeometry_BP0mm_2e10protons_withTimestamps"
    NAME_FILE2 = "OptimisedGeometry_BP5mm_4e9protons_withTimestamps"
    NAME_FILE1 += "_" + generator_name
    NAME_FILE2 += "_" + generator_name
    npz_data_BP0mm = np.load(dir_results + run_name + "/" + NAME_FILE1 + "/" + NAME_FILE1 + ".npz")
    npz_data_BP5mm = np.load(dir_results + run_name + "/" + NAME_FILE2 + "/" + NAME_FILE2 + ".npz")
    ary_nn_pred_bp0mm = npz_data_BP0mm["NN_PRED"]
    ary_nn_pred_bp5mm = npz_data_BP5mm["NN_PRED"]

    proj_0mm, proj0mm_err = IRBackprojection.get_backprojection(ary_score=ary_nn_pred_bp0mm[:, 0],
                                                                ary_e1=ary_nn_pred_bp0mm[:, 1],
                                                                ary_e2=ary_nn_pred_bp0mm[:, 2],
                                                                ary_x1=ary_nn_pred_bp0mm[:, 3],
                                                                ary_y1=ary_nn_pred_bp0mm[:, 4],
                                                                ary_z1=ary_nn_pred_bp0mm[:, 5],
                                                                ary_x2=ary_nn_pred_bp0mm[:, 6],
                                                                ary_y2=ary_nn_pred_bp0mm[:, 7],
                                                                ary_z2=ary_nn_pred_bp0mm[:, 8],
                                                                ary_theta=ary_nn_pred_bp0mm[:, 9],
                                                                use_theta="DOTVEC",
                                                                optimized=True,
                                                                veto=True,
                                                                f_sample=f_sample_0mm,
                                                                n_subsample=20,
                                                                scatz=80.0,
                                                                scaty=40.0,
                                                                threshold=0.5,
                                                                verbose=1)

    proj_5mm, proj5mm_err = IRBackprojection.get_backprojection(ary_score=ary_nn_pred_bp5mm[:, 0],
                                                                ary_e1=ary_nn_pred_bp5mm[:, 1],
                                                                ary_e2=ary_nn_pred_bp5mm[:, 2],
                                                                ary_x1=ary_nn_pred_bp5mm[:, 3],
                                                                ary_y1=ary_nn_pred_bp5mm[:, 4],
                                                                ary_z1=ary_nn_pred_bp5mm[:, 5],
                                                                ary_x2=ary_nn_pred_bp5mm[:, 6],
                                                                ary_y2=ary_nn_pred_bp5mm[:, 7],
                                                                ary_z2=ary_nn_pred_bp5mm[:, 8],
                                                                ary_theta=ary_nn_pred_bp5mm[:, 9],
                                                                use_theta="DOTVEC",
                                                                optimized=True,
                                                                veto=True,
                                                                f_sample=f_sample_5mm,
                                                                n_subsample=20,
                                                                scatz=80.0,
                                                                scaty=40.0,
                                                                threshold=0.5,
                                                                verbose=1)

    PTImageReconstruction.plot_beamprojection_dual(proj_0mm,
                                                   proj0mm_err,
                                                   proj_5mm,
                                                   proj5mm_err,
                                                   labels=["BP0mm", "BP5mm"],
                                                   figure_name=figure_name)


get_ir_from_neuralnetwork("DNN_S4X6",
                          "DNN_S4A6",
                          0.01,
                          0.05,
                          dir_plots + "IRPB_DNN_S4X6_2e8protons_dotvec_optimized")
