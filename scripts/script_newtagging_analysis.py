import numpy as np
import os
import matplotlib.pyplot as plt

from SiFiCCNN.root import Root, RootFiles, RootLogger

# get current path, go two subdirectories higher
path = os.getcwd()
while True:
    path = os.path.abspath(os.path.join(path, os.pardir))
    if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
        break
path_main = path
path_root = path + "/root_files/"

root_parser_old = Root.Root(path_root + RootFiles.onetoone_BP0mm)
root_parser_new = Root.Root(path_root + RootFiles.onetoone_BP0mm_newTag)

print(root_parser_old.events_entries - 3154769)
print(root_parser_old.events_entries / 5 - root_parser_new.events_entries)

# --------------------------------------------------------------------------------------------------
# Analysis: Compton event statistic

n = 100000  # root_parser_new.events_entries
n_compton_old = 0
n_compton_new = 0

for i, event in enumerate(root_parser_old.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        n_compton_old += 1

for i, event in enumerate(root_parser_new.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        n_compton_new += 1

print("### Compton Statistics:")
print("Old Tagging: {:.1f} %".format(n_compton_old / n * 100))
print("New Tagging: {:.1f} %".format(n_compton_new / n * 100))

"""
n = 100000 # root_parser_new.events_entries
n_compton_old = 0
n_compton_new = 0

for i, event in enumerate(root_parser_old.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.MCPosition_source.z != 0.0:
        n_compton_old += 1

for i, event in enumerate(root_parser_new.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.MCPosition_source.z != 0.0:
        n_compton_new += 1

print("### Compton Statistics:")
print("Old Tagging: {:.1f} %".format(n_compton_old / n * 100))
print("New Tagging: {:.1f} %".format(n_compton_new / n * 100))
"""

# --------------------------------------------------------------------------------------------------

n = 100000
list_eventtype_old = []
list_eventtype_new = []

for i, event in enumerate(root_parser_old.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        list_eventtype_old.append(event.MCSimulatedEventType)
for i, event in enumerate(root_parser_new.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        list_eventtype_new.append(event.MCSimulatedEventType)

plt.figure()
bins = np.arange(0.0, 7.0, 1.0, dtype=int) + 0.5
plt.hist(list_eventtype_old, bins=bins, histtype=u"step", color="blue", label="old")
plt.hist(list_eventtype_new, bins=bins, histtype=u"step", color="red", label="new")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------------------
"""
n = 500000
list_sp_old = []
list_sp_new = []
list_sp_all = []
list_sp_cb = []
list_sp_awal = []

for i, event in enumerate(root_parser_old.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        list_sp_old.append(event.MCPosition_source.z)
for i, event in enumerate(root_parser_new.iterate_events(n=n)):
    # event.set_tags_awal()
    if event.compton_tag:
        list_sp_new.append(event.MCPosition_source.z)

    if event.Identified != 0 and event.MCPosition_source.z != 0.0:
        list_sp_cb.append(event.MCPosition_source.z)

    event.set_tags_awal()
    if event.compton_tag:
        list_sp_awal.append(event.MCPosition_source.z)

plt.figure()
bins = np.arange(int(min(list_sp_old)), int(max(list_sp_old)), 1.0, dtype=int)
plt.xlabel("MCPosition_source.z [mm]")
plt.hist(list_sp_old, bins=bins, histtype=u"step", color="blue", label="old", density=True)
plt.hist(list_sp_new, bins=bins, histtype=u"step", color="red", label="new", density=True)
plt.hist(list_sp_cb, bins=bins, histtype=u"step", color="green", label="CB", density=True)
plt.hist(list_sp_awal, bins=bins, histtype=u"step", color="black", label="Awal", density=True)
plt.legend()
plt.tight_layout()
plt.show()
"""