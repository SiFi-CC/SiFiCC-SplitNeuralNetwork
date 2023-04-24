import os

from src.SiFiCCNN.EventDisplay import EDDisplay
from src.SiFiCCNN.Root import RootParser, RootFiles

dir_main = os.getcwd()

# ----------------------------------------------------------------------------------------------------------------------

root_parser = RootParser.Root(dir_main + RootFiles.FinalDetectorVersion_RasterCoupling_OPM_38e8protons_local)
print(root_parser.events_keys)
print(root_parser.ifglobal)
print(root_parser.ifreco)
print(root_parser.ifcluster)
print(root_parser.ifsipm)

for i in range(100):
    event = root_parser.get_event(i)
    EDDisplay.display(event)
