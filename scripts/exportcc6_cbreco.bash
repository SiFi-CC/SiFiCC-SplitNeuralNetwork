#!/bin/bash

# This script will automatically produce root files for the CC6 image reconstruction
# The script_exportcc6_cbreco.py file will be used

PATH="/net/data_g4rt/projects/SiFiCC/InputforNN/ClusterNNOptimisedGeometry/"
FILE1="OptimisedGeometry_BP0mm_2e10protons_taggingv3.root"
FILE2="OptimisedGeometry_BP5mm_4e9protons_taggingv3.root"

python3 script_exportcc6_cbreco.py --rf $PATH$FILE1 --tag "identified"
python3 script_exportcc6_cbreco.py --rf $PATH$FILE1 --tag "distcompton"
python3 script_exportcc6_cbreco.py --rf $PATH$FILE1 --tag "distcompton_awal"

python3 script_exportcc6_cbreco.py --rf $PATH$FILE2 --tag "identified"
python3 script_exportcc6_cbreco.py --rf $PATH$FILE2 --tag "distcompton"
python3 script_exportcc6_cbreco.py --rf $PATH$FILE2 --tag "distcompton_awal"