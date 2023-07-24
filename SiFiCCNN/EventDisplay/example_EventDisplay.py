import numpy as np
import os

from SiFiCCNN.EventDisplay import Display

from SiFiCCNN.root import RootFiles

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
    path = os.path.abspath(os.path.join(path, os.pardir))

path_main = path
path_root = path_main + "/root_files/"

ROOTFILE = path_root + RootFiles.onetoone_cont_taggingv2


display = Display.Display(ROOTFILE)
display.selector_index(3)
display.summary()
display.show()

