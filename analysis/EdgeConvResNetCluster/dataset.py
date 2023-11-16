import os
import numpy as np
import spektral
import pandas as pd

from spektral.data import Dataset, Graph
from spektral.utils import io, sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GraphCluster(Dataset):
    def __init__(self,
                 name,
                 adj_arg="Binary",
                 norm_x=None,
                 norm_e=None,
                 p_only=False,
                 reg_type=None,
                 **kwargs):

        self.name = name
        self.adj_arg = adj_arg
        self.p_only = p_only
        self.reg_type = reg_type

        self.norm_x = norm_x
        self.norm_e = norm_e

        super().__init__(**kwargs)

    @property
    def path(self):
        # get current path, go two subdirectories higher
        path = os.getcwd()
        while True:
            if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
                break
            path = os.path.abspath(os.path.join(path, os.pardir))
        path = os.path.join(path, "datasets", "SiFiCCNN_GraphCluster", self.name)

        return path

    def download(self):
        """
        Download method is needed if Dataset class from Spektral library is inherited. It is
        practically not needed.

        Returns:
            None
        """
        print("Missing download method!")

    def read(self):
        """
        Loading dataset from files and generates graph objects.

        Returns:

        """

        # Batch index
        node_batch_index = np.load(self.path + "/" + "graph_indicator.npy")
        n_nodes = np.bincount(node_batch_index)
        n_nodes_cum = np.concatenate(([0], np.cumsum(n_nodes)[:-1]))

        # Read edge lists
        edges = np.load(self.path + "/" + "A.npy")

        # Split edges into separate edge lists
        edge_batch_idx = node_batch_index[edges[:, 0]]
        n_edges = np.bincount(edge_batch_idx)
        n_edges_cum = np.cumsum(n_edges[:-1])
        el_list = np.split(edges - n_nodes_cum[edge_batch_idx, None],
                           n_edges_cum)

        # get node attributes (x_list)
        x_list = self._get_x_list(n_nodes_cum=n_nodes_cum)
        # get edge attributes (e_list), in this case edge features are disabled
        e_list = [None] * len(n_nodes)

        # Create sparse adjacency matrices and re-sort edge attributes in lexicographic order
        a_e_list = [sparse.edge_index_to_matrix(edge_index=el,
                                                edge_weight=np.ones(el.shape[0]),
                                                edge_features=e,
                                                shape=(n, n), )
                    for el, e, n in zip(el_list, e_list, n_nodes)
                    ]
        a_list = a_e_list
        # if edge features, use this: a_list, e_list = list(zip(*a_e_list))

        # set dataset target (classification / regression)
        y_list = self._get_y_list()
        labels = np.load(self.path + "/" + "graph_labels.npy")

        # limited to True positives only if needed
        if self.p_only:
            # Convert to Graph
            print("Successfully loaded {}.".format(self.name))
            return [
                Graph(x=x, a=a, y=y)
                for x, a, y, label in zip(x_list, a_list, y_list, labels) if label
            ]

        # Convert to Graph
        print("Successfully loaded {}.".format(self.name))
        return [
            Graph(x=x, a=a, y=y)
            for x, a, y in zip(x_list, a_list, labels)
        ]

    def _get_x_list(self, n_nodes_cum):
        # Node features
        x_attr = np.load(self.path + "/" + "node_attributes.npy")
        if self.norm_x is None:
            self.norm_x = self._get_standardization(x_attr)
        self._standardize(x_attr, self.norm_x)
        x_list = np.split(x_attr, n_nodes_cum[1:])

        return x_list

    def _get_e_list(self, n_edges_cum):
        e_attr = np.load(self.path + "/" + "edge_attributes.npy")  # ["arr_0"]
        if self.norm_e is None:
            self.norm_e = self._get_standardization(e_attr)
        self._standardize(e_attr, self.norm_e)
        e_list = np.split(e_attr, n_edges_cum)
        return e_list

    def _get_y_list(self):
        if self.reg_type is not None:
            graph_attributes = np.load(self.path + "/" + "graph_attributes.npy")
            if self.reg_type == "Energy":
                y_list = graph_attributes[:, :2]
            elif self.reg_type == "Position":
                y_list = graph_attributes[:, 2:]
            else:
                print("Warning: Regression type not set correctly")
                return None

        else:
            # return class labels
            y_list = np.load(self.path + "/" + "graph_labels.npy")
        return y_list

    def get_classweight_dict(self):
        labels = np.load(self.path + "/" + "graph_labels.npy")  # ["arr_0"]

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

        ary_norm = np.zeros(shape=(x.shape[1], 2))
        ary_norm[:, 0] = np.mean(x, axis=0)
        ary_norm[:, 1] = np.std(x, axis=0)

        return ary_norm

    @staticmethod
    def _standardize(x, ary_norm):
        for i in range(x.shape[1]):
            x[:, i] -= ary_norm[i, 0]
            x[:, i] /= ary_norm[i, 1]

    @property
    def sp(self):
        sp = np.load(self.path + "/" + "graph_sp.npy")
        return sp

    @property
    def pe(self):
        pe = np.load(self.path + "/" + "graph_pe.npy")
        return pe
