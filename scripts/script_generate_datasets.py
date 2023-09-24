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
from analysis.EdgeConvResNetCluster import downloader
n = 10000
root_parser_0mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_simV2)
root_parser_5mm = RootParser.RootParser(path_root + RootFiles.onetoone_BP5mm_simV2)
root_parser_cont = RootParser.RootParser(path_root + RootFiles.onetoone_CONT_simV2)
#root_parser_m5mm = RootParser.RootParser(path_root + RootFiles.onetoone_BPminus5mm_simV2)
downloader.load(root_parser_0mm, path=path_datasets, n=n)
downloader.load(root_parser_5mm, path=path_datasets, n=n)
downloader.load(root_parser_cont, path=path_datasets, n=n)
#downloader.load(root_parser_m5mm, path=path_datasets, n=n)
"""

from analysis.EdgeConvResNetSiPM import downloader
n = 2000
#root_parser_0mm = RootParser.RootParser(path_root + RootFiles.fourtoone_BP0mm_simv4)
#root_parser_5mm = RootParser.RootParser(path_root + RootFiles.fourtoone_BP5mm_simv4)
root_parser_cont = RootParser.RootParser(path_root + RootFiles.fourtoone_CONT_simv4)
#root_parser_m5mm = RootParser.RootParser(path_root + RootFiles.fourtoone_BPm5mm_simv4)
#root_parser_10mm = RootParser.RootParser(path_root + RootFiles.fourtoone_BP10mm_simv4)
#downloader.load(root_parser_0mm, path=path_datasets, n=n)
#downloader.load(root_parser_5mm, path=path_datasets, n=n)
downloader.load(root_parser_cont, path=path_datasets, n=2000000)
#downloader.load(root_parser_m5mm, path=path_datasets, n=n)
#downloader.load(root_parser_10mm, path=path_datasets, n=n)

