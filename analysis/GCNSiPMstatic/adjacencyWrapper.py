import tensorflow as tf
import numpy as np
import spektral as spk

from adjacency import load_adj


class ConcatAdj(tf.keras.layers.Layer):
    """Custom Layer to store loadable/reinitializable version of the adjacency matrix.
    If node positions are passed the adjacency matrix is calculated based on additional arguments.
    In latter case 'normalized_laplacian' is applied in addition.

    Args:
        adj: numpy array (N,N). Adjacency matrix. Not needed if pos and adjArgs are passed.
        filterFnc: str. keyword specifying the filter function used to normalize the adjacency
                        matrix. (See Kipf and Welling paper and spektral documentation)

    Input:
        previous layer.
    Return:
        List which contains input and adjacency matrix.
    """

    def __init__(self, adj=None,
                 filterFnc="gcn_filter",
                 **kwargs):
        super(ConcatAdj, self).__init__(**kwargs)
        self.adj = adj
        self.filterFnc = filterFnc

        adj = load_adj().astype(np.float32)
        if self.filterFnc == "normalized_laplacian":
            adj = spk.utils.normalized_laplacian(adj)
        elif self.filterFnc == "gcn_filter":
            adj = spk.utils.gcn_filter(adj)
        elif self.filterFnc == "mirrored_normalized_laplacian":
            adj = np.eye(adj.shape[-1], dtype=adj.dtype) + spk.utils.normalized_adjacency(adj,
                                                                                          symmetric=True)
        elif self.filterFnc == "normalized_adjacency":
            adj = spk.utils.normalized_adjacency(adj, symmetric=True)

        # Set the tf tensor
        self.adj_tensor = tf.constant(adj)

    def call(self, input):
        return [input, self.adj_tensor]

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "adj": self.adj,
                "filterFnc": self.filterFnc}
