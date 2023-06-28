####################################################################################################
# Script tagging analysis
#
# Script for analysing the compton tagging statistics (distributed compton events) for multiple
# datasets. Main usage is to check if the improvements to the simulation for writing monte-carlo
# information into the root file and therefore get the correct Compton tagging statistics.
#
####################################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
from SiFiCCNN.root import RootParser, RootFiles

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    path = os.path.abspath(os.path.join(path, os.pardir))
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
path_main = path
path_root = path + "/root_files/"

# load root files
# As a comparison the old BP0mm with taggingv1 will be loaded as well
root_parser_old = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_taggingv1)
root_parser_new = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_taggingv2)

# --------------------------------------------------------------------------------------------------
# Main loop over both root files

# number of entries considered for analysis (either integer or None for all entries available)
# n1 <=> old tagging file, n2 <=> new tagging file
n = None

if n is None:
    n1 = root_parser_old.events_entries
    n2 = root_parser_new.events_entries
else:
    n1 = n
    n2 = n

# predefine all needed variables
n1_compton = 0
n2_compton = 0
n1_nonzero = 0
n2_nonzero = 0
list1_eventtype = []
list2_eventtype = []
list1_sp = []
list2_sp = []

list1_eventtype_awal = []
n1_compton_awal = 0
list1_sp_awal = []

n1_compton_cs = 0

for i, event in enumerate(root_parser_old.iterate_events(n=n1)):
    # event.set_tags_awal()
    if event.compton_tag:
        n1_compton += 1
        list1_eventtype.append(event.MCSimulatedEventType)
        list1_sp.append(event.MCPosition_source.z)

        if event.temp_correctsecondary:
            n1_compton_cs += 1

    if event.MCEnergy_e != 0.0:
        n1_nonzero += 1

    event.set_tags_awal()
    if event.compton_tag:
        n1_compton_awal += 1
        list1_sp_awal.append(event.MCPosition_source.z)
        list1_eventtype_awal.append(event.MCSimulatedEventType)

for i, event in enumerate(root_parser_new.iterate_events(n=n2)):
    # event.set_tags_awal()
    if event.compton_tag:
        n2_compton += 1
        list2_eventtype.append(event.MCSimulatedEventType)
        list2_sp.append(event.MCPosition_source.z)
    if event.MCEnergy_e != 0.0:
        n2_nonzero += 1

# --------------------------------------------------------------------------------------------------
# Analysis output
# print general statistics of root file for control
print("LOADED " + root_parser_old.file_name)
print("Entries: {:.0f} (4e9 equivalent: {:.0f}))".format(root_parser_old.events_entries,
                                                         root_parser_old.events_entries / 5))
print("Non-zero entries:: {:.1f} %".format(n1_nonzero / n1 * 100))
print("Distributed Compton: {:.1f} %".format(n1_compton / n1 * 100))

print("CorrectSecondary: {:.1f} % total | {:.1f} % Compton".format(n1_compton_cs / n1 * 100,
                                                                   n1_compton_cs / n1_compton * 100))

print("")
print("LOADED " + root_parser_new.file_name)
print("Entries: {:.0f} (4e9 equivalent: {:.0f}))".format(root_parser_new.events_entries,
                                                         root_parser_new.events_entries))
print("Non-zero entries:: {:.1f} %".format(n2_nonzero / n2 * 100))
print("Distributed Compton: {:.1f} %".format(n2_compton / n2 * 100))

# Simulated event type plot
plt.figure(figsize=(8, 6))
bins = np.arange(0.0, 7.0, 1.0, dtype=int) + 0.5
plt.ylabel("Counts")
plt.xticks([1, 2, 3, 4, 5, 6],
           ["",
            "real\ncoincidence",
            "random\ncoincidence",
            "",
            "real\ncoincidence\n+ pile-up",
            "random\ncoincidence\n+ pile-up"], rotation=45)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
plt.hist(list1_eventtype_awal,
         bins=bins, histtype=u"step", color="red", linestyle="-.", linewidth=2.0,
         label="Old Dataset, tagging v1 (Awal)")
plt.hist(list1_eventtype,
         bins=bins, histtype=u"step", color="black", linestyle="-", linewidth=2.0,
         label="Old Dataset, tagging v2")
plt.hist(list2_eventtype, bins=bins, histtype=u"step", color="blue", linestyle="--", linewidth=2.0,
         label="New Dataset, tagging v2")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.show()

# Source position z plot
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
width = 1.0
bins = np.arange(int(min(list1_sp)), 20, width, dtype=int)
hist1, _ = np.histogram(list1_sp, bins=bins)
hist2, _ = np.histogram(list2_sp, bins=bins)
res = np.zeros(shape=(len(hist1),))
for i in range(len(hist1)):
    if hist1[i] != 0 and hist2[i] != 0:
        res[i] = (hist1[i] / np.sum(hist1) - hist2[i] / np.sum(hist2)) / (
                np.sqrt(hist2[i]) / np.sum(hist2))
    else:
        res[i] = 0

axs[0].set_ylabel("Counts (Normalized)")
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].hist(list1_sp, bins=bins, histtype=u"step", color="black",
            label="Old Dataset, tagging v2", density=True,
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1), np.sqrt(hist1) / np.sum(hist1),
                color="black", fmt=".")
axs[0].hist(list2_sp, bins=bins, histtype=u"step", color="blue",
            label="New Dataset, tagging v2", density=True,
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2), np.sqrt(hist2) / np.sum(hist2),
                color="blue", fmt=".")
axs[0].legend(loc="upper left")
axs[0].grid()
axs[1].set_xlabel("MCPosition_source.z [mm]")
axs[1].set_ylabel("Residual")
axs[1].errorbar(bins[1:] - width / 2, res, np.ones(shape=(len(hist2),)), fmt=".", color="red")
axs[1].hlines(xmin=bins[0], xmax=bins[-1], y=0, linestyle="--", color="black")
plt.tight_layout()
plt.show()

# Source position z plot Awal
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
width = 1.0
bins = np.arange(int(min(list2_sp)), 20, width, dtype=int)
hist1, _ = np.histogram(list2_sp, bins=bins)
hist2, _ = np.histogram(list1_sp_awal, bins=bins)
res = np.zeros(shape=(len(hist1),))
for i in range(len(hist1)):
    if hist1[i] != 0 and hist2[i] != 0:
        res[i] = (hist1[i] / np.sum(hist1) - hist2[i] / np.sum(hist2)) / (
                np.sqrt(hist2[i]) / np.sum(hist2))
    else:
        res[i] = 0

axs[0].set_ylabel("Counts")
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].hist(list2_sp, bins=bins, histtype=u"step", color="black",
            label="New Dataset, tagging v2", density=True,
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1), np.sqrt(hist1) / np.sum(hist1),
                color="black", fmt=".")
axs[0].hist(list1_sp_awal, bins=bins, histtype=u"step", color="blue",
            label="Old Dataset, tagging v1",
            density=True, linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2), np.sqrt(hist2) / np.sum(hist2),
                color="blue", fmt=".")
axs[0].legend(loc="upper left")
axs[0].grid()
axs[1].set_xlabel("MCPosition_source.z [mm]")
axs[1].set_ylabel("Residual")
axs[1].errorbar(bins[1:] - width / 2, res, np.ones(shape=(len(hist2),)), fmt=".", color="red")
axs[1].hlines(xmin=bins[0], xmax=bins[-1], y=0, linestyle="--", color="black")
plt.tight_layout()
plt.show()
