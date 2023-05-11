import os
import numpy as np
import spektral

from src.SiFiCCNN.Root import RootFiles
from src.SiFiCCNN.Root import RootParser

from src.SiFiCCNN.GCN import IGSiFICCCluster

################################################################################
# Set paths
################################################################################

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"
dir_datasets = dir_main + "/datasets/"
dir_results = dir_main + "/results/"

################################################################################
# generate dataset
################################################################################

n = 100000

rootparser_cont = RootParser.Root(
    dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)

IGSiFICCCluster.gen_SiFiCCCluster(rootparser_cont, n=n)
