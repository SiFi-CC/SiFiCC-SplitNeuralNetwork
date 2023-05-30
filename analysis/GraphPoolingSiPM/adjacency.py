import numpy as np

import sklearn as sk


def calcWeightedAdj(pos,
                    kNeighbor,
                    kr,
                    includeSelf,
                    r0,
                    mode):
    r""" Calculate weighted adjacency matrix.
    Node positions are used to calculate an adjaceny matrix
    based on the spatial distance and weights its entries.

    Args:
        pos: 2D ndarray (N, 3); node positions in cartesian coordinates.
        kNeighbor: bool; use k nearest neighbors approach or nearest neighbors
                         in a ceartain radius.
        kr: int or float; number of nearest neighbors or radius.
        includeSelf: bool; includ self loops of the nodes.
        r0: float; scale parameter for weighting function.
        mode: string ('r_squared','exp','gaussian'); choose weighting function
                                            for the adjacency matrix elements.

    Returns:
        A: ndarray (N, N); Adjacency matrix based on the nearest neighbors given
                           by the node positions.
    """
    if kNeighbor:
        A = sk.neighbors.kneighbors_graph(pos,
                                          kr,
                                          mode="distance",
                                          metric="euclidean",
                                          include_self=includeSelf).toarray()
    else:
        A = sk.neighbors.radius_neighbors_graph(pos,
                                                kr,
                                                mode="distance",
                                                metric="euclidean",
                                                include_self=includeSelf).toarray()

    A = np.maximum(A, A.T)

    nonZeroMask = np.logical_and(~np.identity(A.shape[0], dtype=bool),
                                 A != 0)  # Avoid division by zero and weighting self loops
    if mode == "r_squared":
        A = np.where(nonZeroMask, 1. / (1 + (A / r0) ** 2), A)
    elif mode == "exp":
        A = np.where(nonZeroMask, np.exp(-(A / r0)), A)
    elif mode == "gaussian":
        A = np.where(nonZeroMask, np.exp(-(A / r0) ** 2), A)
    else:
        pass
    return A
