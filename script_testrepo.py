"""
Basic script to test if all libraries are loaded correctly.
Test that the pytohn environment has loaded all needed packages
for GPU training.
"""

# import important packages
import tensorflow
import uproot
import uproot_methods

# test classes
from classes import Detector
from classes import Event
from classes import root_files
from classes import Rootdata
from classes import utilities

# check their versions
print("\ntensorflow: version ", tensorflow.__version__)
print("uproot: version ", uproot.__version__)
print("uproot_methods: version ", uproot_methods.__version__)

# test tensorflow GPU availability
print("\n### Tensorflow GPU training:")
print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))
print(tensorflow.config.list_physical_devices('GPU'))
