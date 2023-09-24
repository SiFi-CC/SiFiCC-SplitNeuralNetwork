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

# load root files with old root-file (simulation version 1) and a new one (simulation version 2)
root_parser_simv1 = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_taggingv1)
root_parser_simv2 = RootParser.RootParser(path_root + RootFiles.onetoone_BP0mm_simV2)

# --------------------------------------------------------------------------------------------------
# Main loop over both root files

# number of entries considered for analysis (either integer or None for all entries available)
# n1 <=> old tagging file, n2 <=> new tagging file
n = 3000000

if n is None:
    n1 = root_parser_simv1.events_entries
    n2 = root_parser_simv2.events_entries
else:
    n1 = n
    n2 = n

# predefine all needed variables
n_compton_simv1_oldtagging = 0
n_compton_simv1_newtagging = 0
n_compton_simv2_oldtagging = 0
n_compton_simv2_newtagging = 0

list_eventtype_simv1_oldtagging = []
list_eventtype_simv1_newtagging = []
list_eventtype_simv2_oldtagging = []
list_eventtype_simv2_newtagging = []

list_sp_simv1_oldtagging = []
list_sp_simv1_newtagging = []
list_sp_simv2_oldtagging = []
list_sp_simv2_newtagging = []

for i, event in enumerate(root_parser_simv1.iterate_events(n=n1)):
    if event.get_distcompton_tag_legacy():
        n_compton_simv1_oldtagging += 1
        list_eventtype_simv1_oldtagging.append(event.MCSimulatedEventType)
        list_sp_simv1_oldtagging.append(event.MCPosition_source.z)

    if event.get_distcompton_tag():
        n_compton_simv1_newtagging += 1
        list_eventtype_simv1_newtagging.append(event.MCSimulatedEventType)
        list_sp_simv1_newtagging.append(event.MCPosition_source.z)

for i, event in enumerate(root_parser_simv2.iterate_events(n=n1)):
    if event.get_distcompton_tag_legacy():
        n_compton_simv2_oldtagging += 1
        list_eventtype_simv2_oldtagging.append(event.MCSimulatedEventType)
        list_sp_simv2_oldtagging.append(event.MCPosition_source.z)

    if event.get_distcompton_tag():
        n_compton_simv2_newtagging += 1
        list_eventtype_simv2_newtagging.append(event.MCSimulatedEventType)
        list_sp_simv2_newtagging.append(event.MCPosition_source.z)

# --------------------------------------------------------------------------------------------------
# Analysis output
# print general statistics of root file for control
print("SimV1 legacy tagging:    Compton events: {:.1f} %".format(
    n_compton_simv1_oldtagging / n1 * 100))
print("SimV1 new tagging:       Compton events: {:.1f} %".format(
    n_compton_simv1_newtagging / n1 * 100))
print("SimV2 legacy tagging:    Compton events: {:.1f} %".format(
    n_compton_simv2_oldtagging / n2 * 100))
print("SimV2 new tagging:       Compton events: {:.1f} %".format(
    n_compton_simv2_newtagging / n2 * 100))
print("################################################")
print("Improvement by new simulation:                   {:.1f} %".format(
    (n_compton_simv2_oldtagging / n_compton_simv1_oldtagging * 100) - 100.))
print("Improvement by new tagging:                      {:.1f} %".format(
    (n_compton_simv1_newtagging / n_compton_simv1_oldtagging * 100) - 100.))
print("Improvement by new tagging on simV2:             {:.1f} %".format(
    (n_compton_simv2_newtagging / n_compton_simv2_oldtagging * 100) - 100.))
print("Improvement by new simulation on new tagging:    {:.1f} %".format(
    (n_compton_simv2_newtagging / n_compton_simv1_newtagging * 100) - 100.))
print("Total improvement:                               {:.1f} %".format(
    (n_compton_simv2_newtagging / n_compton_simv1_oldtagging * 100) - 100.))

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
plt.hist(list_eventtype_simv1_oldtagging,
         bins=bins, histtype=u"step", color="black", linestyle="-.", linewidth=2.0,
         label="SimualtionV1, legacy tagging")
plt.hist(list_eventtype_simv2_newtagging,
         bins=bins, histtype=u"step", color="red", linestyle="-", linewidth=2.0,
         label="SimulationV2, new tagging")
plt.hist(list_eventtype_simv1_newtagging, bins=bins, histtype=u"step", color="blue", linestyle="--",
         linewidth=2.0,
         label="SimulationV1, new tagging")
plt.legend(loc="upper right")
plt.grid()
plt.tight_layout()
plt.show()

# Comparison of source position distribution for the simulation version 1 and the different
# tagging algorithms
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
width = 1.0
bins = np.arange(int(min(list_sp_simv1_oldtagging)), 20, width, dtype=int)
hist1, _ = np.histogram(list_sp_simv1_oldtagging, bins=bins)
hist2, _ = np.histogram(list_sp_simv1_newtagging, bins=bins)
res = np.zeros(shape=(len(hist1),))
for i in range(len(hist1)):
    if hist1[i] != 0 and hist2[i] != 0:
        res[i] = (hist1[i] - hist2[i]) / (np.sqrt(hist2[i]))
    else:
        res[i] = 0

axs[0].set_ylabel("Counts (Normalized)")
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].hist(list_sp_simv1_oldtagging, bins=bins, histtype=u"step", color="black",
            label="SimualtionV1, legacy tagging",
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1),
                color="black", fmt=".")
axs[0].hist(list_sp_simv1_newtagging, bins=bins, histtype=u"step", color="blue",
            label="SimulationV1, new tagging",
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2),
                color="blue", fmt=".")
axs[0].legend(loc="upper left")
axs[0].grid()
axs[1].set_xlabel("MCPosition_source.z [mm]")
axs[1].set_ylabel("Residual")
axs[1].errorbar(bins[1:] - width / 2, res, np.ones(shape=(len(hist2),)), fmt=".", color="red")
axs[1].hlines(xmin=bins[0], xmax=bins[-1], y=0, linestyle="--", color="black")
plt.tight_layout()
plt.show()

# Comparison of source position distribution for the simulation version 2 and the different
# tagging algorithms
fig, axs = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
width = 1.0
bins = np.arange(int(min(list_sp_simv2_oldtagging)), 20, width, dtype=int)
hist1, _ = np.histogram(list_sp_simv2_oldtagging, bins=bins)
hist2, _ = np.histogram(list_sp_simv2_newtagging, bins=bins)
res = np.zeros(shape=(len(hist1),))
for i in range(len(hist1)):
    if hist1[i] != 0 and hist2[i] != 0:
        res[i] = (hist1[i] - hist2[i]) / (np.sqrt(hist2[i]))
    else:
        res[i] = 0

axs[0].set_ylabel("Counts")
axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
axs[0].hist(list_sp_simv2_oldtagging, bins=bins, histtype=u"step", color="black",
            label="SimulationV2, legacy tagging",
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1),
                color="black", fmt=".")
axs[0].hist(list_sp_simv2_newtagging, bins=bins, histtype=u"step", color="blue",
            label="SimulationV2, new tagging",
            linestyle="-", linewidth=1.0)
axs[0].errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2),
                color="blue", fmt=".")
axs[0].legend(loc="upper left")
axs[0].grid()
axs[1].set_xlabel("MCPosition_source.z [mm]")
axs[1].set_ylabel("Residual")
axs[1].errorbar(bins[1:] - width / 2, res, np.ones(shape=(len(hist2),)), fmt=".", color="red")
axs[1].hlines(xmin=bins[0], xmax=bins[-1], y=0, linestyle="--", color="black")
plt.tight_layout()
plt.show()
