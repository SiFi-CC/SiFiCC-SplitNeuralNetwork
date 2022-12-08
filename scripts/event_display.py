def edge_list(x, y, z, xdim, ydim, zdim):
    list_edges = [[[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
                  [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
                  [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
                  [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
                  [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
                  [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
                  [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
                  [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
                  [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
                  [[x - xdim / 2, x - xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
                  [[x + xdim / 2, x + xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
                  [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]

    return list_edges


def event_display(RootParser, n=1):
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from itertools import product, combinations

    from classes import Event

    # get event from position n
    event = RootParser.get_event(n)

    list_edge_scatterer = edge_list(RootParser.scatterer.pos.x,
                                    RootParser.scatterer.pos.y,
                                    RootParser.scatterer.pos.z,
                                    RootParser.scatterer.dimx,
                                    RootParser.scatterer.dimy,
                                    RootParser.scatterer.dimz)

    list_edge_absorber = edge_list(RootParser.absorber.pos.x,
                                   RootParser.absorber.pos.y,
                                   RootParser.absorber.pos.z,
                                   RootParser.absorber.dimx,
                                   RootParser.absorber.dimy,
                                   RootParser.absorber.dimz)

    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(list_edge_scatterer)):
        ax.plot3D(list_edge_scatterer[i][0], list_edge_scatterer[i][1], list_edge_scatterer[i][2], color="blue")
        ax.plot3D(list_edge_absorber[i][0], list_edge_absorber[i][1], list_edge_absorber[i][2], color="blue")

    # plot source axis
    ax.plot3D([0, 270 + 46.8/2], [0, 0], [0, 0], color="black")
    ax.set_title("Cube")
    plt.show()
