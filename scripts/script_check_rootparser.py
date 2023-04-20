import os
import numpy as np

from src.SiFiCCNN.Root import RootParser
from src.SiFiCCNN.Root import RootFiles

dir_main = os.getcwd() + "/../"
dir_root = dir_main + "/root_files/"

# ----------------------------------------------------------------------------------------------------------------------

rootfile = RootParser.Root(dir_main + RootFiles.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_local)
print(rootfile.events_keys)
print("GLOBAL: ", rootfile.ifglobal)
print("RECO: ", rootfile.ifreco)
print("CLUSTER: ", rootfile.ifcluster)
print("SIPM: ", rootfile.ifsipm)
event = rootfile.get_event(0)
