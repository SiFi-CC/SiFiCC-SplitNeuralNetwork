import numpy as np
import os
import matplotlib.pyplot as plt
from src import Plotter
from src import ToyGenerator

plt.rcParams.update({'font.size': 14})

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"

# ----------------------------------------------------------------------------------------------------------------------
# Toy set generation
"""
# Base sample
ToyGenerator.create_toy_set(FILE_NAME="S1AX_continuous_master",
                            TAG="base",
                            PATH_NN_PRED_0MM=dir_results + "DNN_S1AX_continuous_master/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz",
                            PATH_NN_PRED_5MM=dir_results + "DNN_S1AX_continuous_master/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz",
                            PATH_MC_TRUTH_0MM=dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz",
                            PATH_MC_TRUTH_5MM=dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz",
                            f_reg_ee=1.0,
                            f_reg_ep=1.0,
                            f_reg_xe=1.0,
                            f_reg_ye=1.0,
                            f_reg_ze=1.0,
                            f_reg_xp=1.0,
                            f_reg_yp=1.0,
                            f_reg_zp=1.0,
                            f_eff=1.0,
                            f_pur=1.0,
                            mod_bg=False)
ToyGenerator.generate_control_plots(FILE_NAME="S1AX_continuous_master",
                                    TAG="base",
                                    PATH_NN_PRED=dir_toy + "S1AX_continuous_master" + "/" + "S1AX_continuous_master" + "_" + "base" + "_toy.npz",
                                    PATH_MC_TRUTH_0MM=dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz",
                                    PATH_MC_TRUTH_5MM=dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz")

"""

# ----------------------------------------------------------------------------------------------------------------------
# Automation scripts
"""
ToyGenerator.create_toy_set(FILE_NAME="S1AX_continuous_an",
                            TAG="tp00",
                            PATH_NN_PRED_0MM=dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S1AX.npz",
                            PATH_NN_PRED_5MM=dir_results + "DNN_S1AX_continuous_an/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX/OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S1AX.npz",
                            PATH_MC_TRUTH_0MM=dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz",
                            PATH_MC_TRUTH_5MM=dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz",
                            f_reg_ee=1.0,
                            f_reg_ep=1.0,
                            f_reg_xe=1.0,
                            f_reg_ye=1.0,
                            f_reg_ze=1.0,
                            f_reg_xp=1.0,
                            f_reg_yp=1.0,
                            f_reg_zp=1.0,
                            f_fp=1.0,
                            f_tp=0.0,
                            mod_bg=False)
"""
"""
ToyGenerator.create_toy_set_cutbased(FILE_NAME="S1AX_CB_reco",
                                     TAG="fp00",
                                     PATH_LOOKUP_0MM=dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_S1AX_lookup.npz",
                                     PATH_LOOKUP_5MM=dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_S1AX_lookup.npz",
                                     f_reg_ee=1.0,
                                     f_reg_ep=1.0,
                                     f_reg_xe=1.0,
                                     f_reg_ye=1.0,
                                     f_reg_ze=1.0,
                                     f_reg_xp=1.0,
                                     f_reg_yp=1.0,
                                     f_reg_zp=1.0,
                                     f_fp=0.0,
                                     f_tp=1.0,
                                     mod_bg=False)
"""