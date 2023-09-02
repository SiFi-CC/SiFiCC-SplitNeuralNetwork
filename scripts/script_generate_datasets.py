####################################################################################################
# Script generate datasets
#
# Script for mass generating datasets. This script does not contain fancy conditions. Just
# comment out the datasets that need generating.
#
####################################################################################################

import os
from SiFiCCNN.root import RootParser, RootFiles

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    path = os.path.abspath(os.path.join(path, os.pardir))
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
path_main = path
path_root = path + "/root_files/"
path_datasets = path + "/datasets/"

####################################################################################################
"""
from analysis.DNNCluster import downloader
root_parser_0mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_taggingv2)
root_parser_5mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP5mm_taggingv2)
root_parser_cont = RootParser.RootParser(path_root + RootFiles.onetoone_cont_taggingv2)
downloader.load(root_parser_0mm, path=path_datasets, n=None)
downloader.load(root_parser_5mm, path=path_datasets, n=None)
downloader.load(root_parser_cont, path=path_datasets, n=None)
"""

from analysis.EdgeConvResNetCluster import downloader

root_parser_0mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_simV2)
root_parser_5mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP5mm_simV2)
root_parser_cont = RootParser.RootParser(path_root + RootFiles.onetoone_CONT_simV2)
root_parser_m5mm = RootParser.RootParser(path_root + RootFiles.onetoone_BPminus5mm_simV2)
downloader.load(root_parser_0mm, path=path_datasets, n=None)
downloader.load(root_parser_5mm, path=path_datasets, n=None)
downloader.load(root_parser_cont, path=path_datasets, n=None)
downloader.load(root_parser_m5mm, path=path_datasets, n=None)
