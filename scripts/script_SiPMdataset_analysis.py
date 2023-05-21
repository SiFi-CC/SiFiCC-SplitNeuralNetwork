import numpy as np
import os
import matplotlib.pyplot as plt

from SiFiCCNN.root import Root, RootFiles
from SiFiCCNN.EventDisplay import EDDisplay

dir_main = os.getcwd() + "/../"
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_datasets = dir_main + "/datasets/SiFiCCNN/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

################################################################################
"""
# example: invalid event
root = Root.Root(
    dir_main + RootFiles.FinalDetectorVersion_RasterCoupling_OPM_38e8protons_local)

# event = root.get_event(2770)
# EDDisplay.display(event)

n = 100000
counter = 0
for i, event in enumerate(root.iterate_events(n=n)):
    if len(event.SiPM_id) == 0:
        counter += 1

print("# Percentage of invalid events: {:.1f}%".format(counter / n * 100))
print("# ({} total)".format(counter))
"""

################################################################################

root = Root.Root(
    dir_main + RootFiles.FinalDetectorVersion_RasterCoupling_OPM_38e8protons_local)

list_qdc = []
list_triggertime = []

n = 1000000
for i, event in enumerate(root.iterate_events(n=n)):
    for j in range(len(event.SiPM_triggertime)):
        list_qdc.append(event.SiPM_qdc[j])
        list_triggertime.append(event.SiPM_triggertime[j])

tt_90p = np.percentile(list_triggertime, 90)
bins_tt = np.arange(0, 5, 0.1)
plt.figure()
plt.xlabel("Trigger-times [ns]")
plt.ylabel("Counts")
plt.hist(list_triggertime, bins=bins_tt, histtype=u"step", color="black")
plt.vlines(x=tt_90p, ymin=0, ymax=n / len(bins_tt) * 50, color="red",
           linestyles="--")
plt.show()

qdc_90p = np.percentile(list_qdc, 90)
bins_qdc = np.arange(0, 5, 0.1)
plt.figure()
plt.xlabel("Photon count")
plt.ylabel("Counts")
plt.hist(list_qdc, bins=100, histtype=u"step", color="black")
plt.vlines(x=qdc_90p, ymin=0, ymax=n / 100 * 50, color="red", linestyles="--")
plt.show()

print("QDC 90percentile: {}".format(qdc_90p))
print("Trigger-time 90percentile: {}".format(tt_90p))
