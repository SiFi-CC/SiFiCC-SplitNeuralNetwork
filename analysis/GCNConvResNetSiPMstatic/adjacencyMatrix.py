import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import radius_neighbors_graph

from SiFiCCNN.EventDisplay import EDBuilder
from SiFiCCNN.utils.adjacency import gen_adj_pos_4to1

def sipm_id_to_tensor(sipm_id):
    # determine y
    y = sipm_id // 368
    # remove third dimension
    sipm_id -= (y * 368)
    # x and z in scatterer
    if sipm_id < 112:
        x = sipm_id // 28
        z = (sipm_id % 28) + 2
    # x and z in absorber
    else:
        x = (sipm_id + 16) // 32
        z = (sipm_id + 16) % 32
    return int(x), int(y), int(z)


def sipm_id_to_position(id):
    # Detector dimensions
    # TODO: soft code detector dimensions into method
    scat_xpos = 150
    scat_ypos = 0
    scat_zpos = 0
    scat_xdim = 14
    scat_ydim = 100
    scat_zdim = 110

    abs_xpos = 270
    abs_ypos = 0
    abs_zpos = 0
    abs_xdim = 30
    abs_ydim = 100
    abs_zdim = 126

    # determine y
    y = id // 368
    if y == 0:
        y = 10
    else:
        y = -10

    # remove third dimension
    id -= ((id // 368) * 368)

    # x and z in scatterer
    if id < 112:
        x = id // 28
        z = (id % 28) + 2
    # x and z in absorber
    else:
        x = (id + 16) // 32 + 10
        z = (id + 16) % 32

    return x, y, z


####################################################################################################
# adjacency matrix calculations

# check if file exist, else generate it
if not os.path.isfile("adj_positions_4to1.txt"):
    gen_adj_pos_4to1()

# load file
ary_id_pos = np.loadtxt("adj_positions_4to1.txt")

# modify y position to allow fibre connections
# y_mod = 15: no fibre connection
y_mod = 5.0
ary_id_pos[:, 1] /= 51.0
ary_id_pos[:, 1] *= y_mod / 2

adj = radius_neighbors_graph(ary_id_pos, radius=6.0)

####################################################################################################

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for row, col in zip(adj.nonzero()[0], adj.nonzero()[1]):

    if (112 <= row < 368) or (368 + 112 <= row):
        continue

    """
    if row != 60:
        continue
    """

    ax.plot3D([ary_id_pos[row, 0], ary_id_pos[col, 0]],
              [ary_id_pos[row, 1], ary_id_pos[col, 1]],
              [ary_id_pos[row, 2], ary_id_pos[col, 2]],
              color="blue")

for i in range(len(ary_id_pos)):

    if (112 <= i < 368) or (368 + 112 <= i):
        continue

    ax.plot3D(ary_id_pos[i, 0], ary_id_pos[i, 1], ary_id_pos[i, 2],
              "o", color="lime", markersize="5")
"""
for i in [60]:
    list_sipm_edges = EDBuilder.get_edges(ary_positions[i, 0],
                                          ary_positions[i, 1],
                                          ary_positions[i, 2],
                                          4.0, 0, 4.0)
    for j in range(len(list_sipm_edges)):
        ax.plot3D(list_sipm_edges[j][0], list_sipm_edges[j][1],
                  list_sipm_edges[j][2], color="darkgreen")
"""
plt.show()
