import os
import numpy as np
import spektral
import pandas as pd

from spektral.data import Dataset, Graph
from spektral.utils import io, sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ----------------------------------------------------------------------------------------------------------------------

class SiFiCCdatasets(Dataset):
    def __init__(self, name, dataset_path, **kwargs):
        self.name = name
        self.dataset_path = dataset_path
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(self.dataset_path, self.__class__.__name__,
                            self.name)

    def download(self):
        print("Dunno some download function")

    def read(self):
        # Batch index
        node_batch_index = (
            io.load_txt(
                self.path + "/" + self.name + "_graph_indicator" + ".txt").astype(
                int)
        )

        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # Read edge lists
        edges = io.load_txt(self.path + "/" + self.name + "_A" + ".txt",
                            delimiter=",").astype(int)

        # Remove duplicates and self-loops from edges
        _, mask = np.unique(edges, axis=0, return_index=True)
        # mask = mask[edges[mask, 0] != edges[mask, 1]]
        edges = edges[mask]

        # Split edges into separate edge lists
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(edges - n_nodes_cum[edge_batch_idx, None],
                           n_edges_cum)

        # Node features
        x_list = []
        x_attr = io.load_txt(
            self.path + "/" + self.name + "_node_attributes" + ".txt",
            delimiter=",")
        if x_attr.ndim == 1:
            x_attr = x_attr[:, None]
        x_list.append(x_attr)

        if len(x_list) > 0:
            x_list = np.concatenate(x_list, -1)
            ary_mean, ary_std = _get_standardization(x_list)
            x_list = _standardize(x_list, ary_mean, ary_std)
            x_list = np.split(x_list, n_nodes_cum[1:])
        else:
            print(
                "WARNING: this dataset doesn't have node attributes."
                "Consider creating manual features before using it with a "
                "Loader."
            )
            x_list = [None] * len(n_nodes)

        # Edge features
        e_list = []
        e_attr = io.load_txt(
            self.path + "/" + self.name + "_edge_attributes" + ".txt",
            delimiter=",")
        if e_attr.ndim == 1:
            e_attr = e_attr[:, None]
        e_attr = e_attr[mask]
        e_list.append(e_attr)
        if len(e_list) > 0:
            e_available = True
            e_list = np.concatenate(e_list, -1)
            e_list = np.split(e_list, n_edges_cum)
        else:
            e_available = False
            e_list = [None] * len(n_nodes)

        # Create sparse adjacency matrices and re-sort edge attributes in
        # lexicographic order
        a_e_list = [
            sparse.edge_index_to_matrix(
                edge_index=el,
                edge_weight=np.ones(el.shape[0]),
                edge_features=e,
                shape=(n, n),
            )
            for el, e, n in zip(el_list, e_list, n_nodes)
        ]
        if e_available:
            a_list, e_list = list(zip(*a_e_list))
        else:
            a_list = a_e_list

        # Labels
        labels = io.load_txt(
            self.path + "/" + self.name + "_graph_labels" + ".txt").astype(np.float32)
        # labels = _normalize(labels[:, None], "ohe")

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=spektral.utils.gcn_filter(a), e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

    def get_classweight_dict(self):
        labels = io.load_txt(
            self.path + "/" + self.name + "_graph_labels" + ".txt")

        _, counts = np.unique(labels, return_counts=True)
        class_weights = {0: len(labels) / (2 * counts[0]),
                         1: len(labels) / (2 * counts[1])}

        return class_weights


def _normalize(x, norm=None):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == "ohe":
        fnorm = OneHotEncoder(sparse=False, categories="auto")
    elif norm == "zscore":
        fnorm = StandardScaler()
    else:
        return x
    return fnorm.fit_transform(x)


def _get_standardization(x):
    """
    Returns array of mean and std of every feature

    Args:
        x (numpy array): node feature matrix

    Returns:
        ary_mean, ary_std
    """
    ary_mean = np.zeros(shape=(x.shape[1],))
    ary_std = np.zeros(shape=(x.shape[1],))

    for i in range(x.shape[1]):
        ary_mean[i] = np.mean(x[:, i])
        ary_std[i] = np.std(x[:, i])
    return ary_mean, ary_std


def _standardize(x, ary_mean, ary_std):
    for i in range(len(ary_mean)):
        x[:, i] = (x[:, i] - ary_mean[i]) / ary_std[i]
    return x
