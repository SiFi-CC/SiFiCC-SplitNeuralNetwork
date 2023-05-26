import os
import numpy as np
import spektral
import pandas as pd

from spektral.data import Dataset, Graph
from spektral.utils import io, sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


################################################################################
#
################################################################################

class GraphCluster(Dataset):
    def __init__(self,
                 name,
                 edge_atr=False,
                 adj_arg="Binary",
                 norm_x=None,
                 norm_e=None,
                 **kwargs):

        self.name = name
        self.edge_atr = edge_atr
        self.adj_arg = adj_arg

        self.norm_x = norm_x
        self.norm_e = norm_e

        super().__init__(**kwargs)

    @property
    def path(self):
        # get current path, go two subdirectories higher
        path = os.path.dirname(os.path.abspath(__file__))
        for i in range(3):
            path = os.path.dirname(path)
        path = os.path.join(path, "datasets", "SiFiCCNN_GraphCluster", self.name)

        return path

    def download(self):
        print("Dunno some download function")

    def read(self):
        # Batch index
        node_batch_index = np.load(
            self.path + "/" + self.name + "_graph_indicator.npy")  # ["arr_0"]
        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # Read edge lists
        edges = np.load(self.path + "/" + self.name + "_A.npy")  # ["arr_0"]
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
        x_attr = np.load(self.path + "/" + self.name + "_node_attributes.npy")  # ["arr_0"]
        if x_attr.ndim == 1:
            x_attr = x_attr[:, None]
        x_list.append(x_attr)

        if len(x_list) > 0:
            x_list = np.concatenate(x_list, -1)
            if self.norm_x is None:
                self.norm_x = self._get_standardization(x_list)
            x_list = self._standardize(x_list, self.norm_x)
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

        if self.edge_atr:
            e_attr = np.load(self.path + "/" + self.name + "_edge_attributes.npy")  # ["arr_0"]
            if e_attr.ndim == 1:
                e_attr = e_attr[:, None]
            e_attr = e_attr[mask]
            e_list.append(e_attr)

        if len(e_list) > 0:
            e_available = True
            e_list = np.concatenate(e_list, -1)
            if self.norm_e is None:
                self.norm_e = self._get_standardization(e_list)
            e_list = self._standardize(e_list, self.norm_e)
            e_list = np.split(e_list, n_edges_cum)
        else:
            e_available = False
            e_list = [None] * len(n_nodes)

        """
        # Create sparse adjacency matrices and re-sort edge attributes in
        # lexicographic order
        if self.adj_arg == "binary":
            # create adjacency matrices
            a_e_list = []
            for i in range(len(el_list)):
                adj = np.zeros(shape=(n_nodes[i], n_nodes[i]),
                               dtype=np.float32)
                for j in range(len(el_list[i])):
                    adj[el_list[i][j][0], el_list[i][j][1]] = 1.0
                a_e_list.append(adj)

        if self.adj_arg == "gcn_binary":
            # create adjacency matrices
            a_e_list = []
            for i in range(len(el_list)):
                adj = np.zeros(shape=(n_nodes[i], n_nodes[i]),
                               dtype=np.float32)
                for j in range(len(el_list[i])):
                    adj[el_list[i][j][0], el_list[i][j][1]] = 1.0
                a_e_list.append(spektral.utils.gcn_filter(adj))

        if self.adj_arg == "gcn_distance":
            # prepare edge attribute for distance weight
            e_attr = io.load_txt(
                self.path + "/" + self.name + "_edge_attributes" + ".txt",
                delimiter=",")

            e_list = []
            e_list.append(e_attr)
            e_list = np.concatenate(e_list, -1)
            e_list = np.split(e_list, n_edges_cum)

            # create adjacency matrices
            a_e_list = []
            for i in range(len(el_list)):
                adj = np.zeros(shape=(n_nodes[i], n_nodes[i]),
                               dtype=np.float32)
                for j in range(len(el_list[i])):
                    adj[el_list[i][j][0], el_list[i][j][1]] = e_list[i][j][0]
                a_e_list.append(spektral.utils.gcn_filter(adj))

            e_available = False
            e_list = [None] * len(n_nodes)
        """

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
        labels = np.load(self.path + "/" + self.name + "_graph_labels.npy")  # ["arr_0"]
        # labels = _normalize(labels[:, None], "ohe")

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, e=e, y=y)
            for x, a, e, y in zip(x_list, a_list, e_list, labels)
        ]

    def get_classweight_dict(self):
        labels = np.load(self.path + "/" + self.name + "_graph_labels.npy")  # ["arr_0"]

        _, counts = np.unique(labels, return_counts=True)
        class_weights = {0: len(labels) / (2 * counts[0]),
                         1: len(labels) / (2 * counts[1])}

        return class_weights

    @staticmethod
    def _get_standardization(x):
        """
        Returns array of mean and std of features along the -1 axis

        Args:
            x (numpy array): feature matrix

        Returns:
            ary_mean, ary_std
        """

        ary_norm = np.zeros(shape=(x.shape[0], 2))
        ary_norm[:, 0] = np.mean(x, axis=0)
        ary_norm[:, 0] = np.std(x, axis=0)

        return ary_norm

    @staticmethod
    def _standardize(x, ary_norm):
        for i in range(x.shape[0]):
            x[:, i] = (x[:, i] - ary_norm[i, 0]) / ary_norm[i, 0]
        return x

    def save_norm(self,
                  file_name):
        np.save(self.norm_x, file_name + "_norm_x")
        np.save(self.norm_e, file_name + "_norm_e")
