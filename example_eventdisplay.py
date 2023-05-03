import numpy as np
import os
import matplotlib.pyplot as plt

from src.SiFiCCNN.EventDisplay import EDDisplay
from src.SiFiCCNN.Root import RootParser, RootFiles, RootLogger

dir_main = os.getcwd()

# ----------------------------------------------------------------------------------------------------------------------

root_parser = RootParser.Root(dir_main + RootFiles.OptimisedGeometry_BP5mm_4e9protons_withTimestamps_local)
print(root_parser.events_keys)
print(root_parser.ifglobal)
print(root_parser.ifreco)
print(root_parser.ifcluster)
print(root_parser.ifsipm)

for i in range(1000):
    event = root_parser.get_event(i)

    if event.compton_tag:
        continue

    # feature_map = event.get_sipm_feature_map()
    RootLogger.print_event_summary(root_parser, i)
    EDDisplay.display(event)

    """
    fig1, axs = plt.subplots(1, 2)
    axs[0].imshow(feature_map[:, 0 + 2, :, 0])
    axs[1].imshow(feature_map[:, 1 + 2, :, 0])
    plt.show()

    fig2, axs = plt.subplots(1, 2)
    axs[0].imshow(feature_map[:, 0 + 2, :, 1])
    axs[1].imshow(feature_map[:, 1 + 2, :, 1])
    plt.show()
    """
