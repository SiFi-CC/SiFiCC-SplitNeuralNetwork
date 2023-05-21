import numpy as np
import os
import matplotlib.pyplot as plt

from src.SiFiCCNN.root.RootCluster import RootCluster
from src.SiFiCCNN.root import RootFiles

dir_main = os.getcwd() + "/.."
dir_root = dir_main + "/root_files/"
dir_npz = dir_main + "/npz_files/"
dir_toy = dir_main + "/toy/"
dir_results = dir_main + "/results/"
dir_plots = dir_main + "/plots/"

# Analysis settings
n = 10000  # total statistics

# ----------------------------------------------------------------------------------------------------------------------

# Load continuous root file as example (reason: it is the training file)
rootcluster_cont = RootCluster(dir_main + RootFiles.OptimisedGeometry_Continuous_2e10protons_withTimestamps_local)

# Cluster counting
n_cluster_total = []
n_cluster_scatterer = []
n_cluster_absorber = []
for i, event in enumerate(rootcluster_cont.iterate_events(n=n)):
    n_cluster_total.append(len(event.RecoClusterEntries))

    idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=False)
    n_cluster_scatterer.append(len(idx_scatterer))
    n_cluster_absorber.append(len(idx_absorber))

# Percentile analysis
print("### PERCENTILE ANALYSIS:")
print("Total number cluster     (90-th percentile): {}".format(np.percentile(n_cluster_total, 90)))
print("Scatterer number cluster (90-th percentile): {}".format(np.percentile(n_cluster_scatterer, 90)))
print("Absorber number cluster  (90-th percentile): {}".format(np.percentile(n_cluster_absorber, 90)))

# plot cluster statistics
ary_bin = np.arange(0.0, max(n_cluster_total) + 1, 1.0, ) - 0.5
hist_tot, _ = np.histogram(n_cluster_total, bins=ary_bin)
hist_scat, _ = np.histogram(n_cluster_scatterer, bins=ary_bin)
hist_abs, _ = np.histogram(n_cluster_absorber, bins=ary_bin)

# TODO: plot these results to visually represent the choice
print(np.sum(hist_tot[:8]) / np.sum(hist_tot) * 100)
print(np.sum(hist_scat[:5]) / np.sum(hist_scat) * 100)
print(np.sum(hist_abs[:5]) / np.sum(hist_abs) * 100)
