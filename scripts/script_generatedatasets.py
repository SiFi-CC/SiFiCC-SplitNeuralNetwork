import os
import numpy as np

from src.SiFiCCNN.Root.RootParser import Root
from src.SiFiCCNN.Root import RootFiles
from src.SiFiCCNN.Root import RootExport

from src.SiFiCCNN.InputGenerator import IGClusterSXAX
from src.SiFiCCNN.InputGenerator import IGRNNClusterSXAX

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"


# ----------------------------------------------------------------------------------------------------------------------

def generate_sxax(n_cs, n_ca):
    rootcluster_cont = Root(dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)
    rootcluster_bp0mm = Root(dir_main + RootFiles.OptimisedGeometry_BP0mm_2e10protons_withTimestamps_local)
    rootcluster_bp5mm = Root(dir_main + RootFiles.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_local)

    # IGRNNClusterSXAX.export_sxax(rootcluster_cont, dir_npz, n_cs, n_ca)
    # IGRNNClusterSXAX.export_sxax(rootcluster_bp0mm, dir_npz, n_cs, n_ca)
    # IGRNNClusterSXAX.export_sxax(rootcluster_bp5mm, dir_npz, n_cs, n_ca)

    # generate classical cut-based reco output
    RootExport.export_classicreco(rootcluster_bp0mm, dir_npz)
    RootExport.export_classicreco(rootcluster_bp5mm, dir_npz)
    RootExport.export_classicreco(rootcluster_cont, dir_npz)


def check_sxax():
    # npz_data_cont = np.load(dir_npz + "OptimisedGeometry_Continuous_2e10protons_DNN_S4A6.npz")
    # npz_data_bp0mm = np.load(dir_npz + "OptimisedGeometry_BP0mm_2e10protons_withTimestamps_DNN_S4A6.npz")
    npz_data_bp5mm = np.load(dir_npz + "OptimisedGeometry_BP5mm_4e9protons_withTimestamps_DNN_S4A6.npz")

    npz_feature_bp5mm = npz_data_bp5mm["features"]
    print(npz_feature_bp5mm.shape)
    print(npz_feature_bp5mm[1, :])


generate_sxax(4, 6)
# check_sxax()
