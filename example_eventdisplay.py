import numpy as np
import os
import matplotlib.pyplot as plt

from SiFiCCNN.EventDisplay import EventDisplay
from SiFiCCNN.root import RootParser, RootFiles

dir_main = os.getcwd()
dir_root = dir_main + "/root_files/"

# --------------------------------------------------------------------------------------------------
"""
root_parser = RootParser.RootParser(dir_root + RootFiles.onetoone_BP0mm_taggingv2)
for i, event in enumerate(root_parser.iterate_events(n=None)):

    if not event.compton_tag:
        continue

    event.summary()
    EDDisplay.display(event)
"""


# --------------------------------------------------------------------------------------------------

root_parser = RootParser.RootParser(dir_root + RootFiles.fourtoone_BP0mm_test)
test_event = root_parser.get_event(2)

display = EventDisplay.EventDisplay()
display.load_event(test_event)
display.draw_detector()
display.draw_reference_axis()
display.draw_promptgamma()
display.draw_interactions()
display.draw_cone_true()
display.draw_fibre_hits()
display.draw_sipm_hits()
display.show()

