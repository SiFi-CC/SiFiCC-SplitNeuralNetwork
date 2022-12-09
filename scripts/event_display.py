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

    from classes.utilities import print_event_summary

    # get event from position n
    event = RootParser.get_event(n)

    # print event summary
    print_event_summary(RootParser, n)

    # ideal compton event tag
    if event.is_ideal_compton:
        ic_tag = "Ideal Compton"
    else:
        ic_tag = "Background"

    # get detector edges
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

    # get detector hits
    list_cluster_x = []
    list_cluster_y = []
    list_cluster_z = []
    for cl in event.RecoClusterPosition:
        list_cluster_x.append(cl.x)
        list_cluster_y.append(cl.y)
        list_cluster_z.append(cl.z)

    # plotting
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-100, 400)
    ax.set_ylim(-100, 100)
    ax.set_zlim(-100, 100)
    ax.set_xlabel("x-axis (Detector-direction)")
    ax.set_ylabel("y-axis (Fibre-direction)")
    ax.set_zlabel("z-axis (Shift-direction)")
    # plot detector edges
    for i in range(len(list_edge_scatterer)):
        ax.plot3D(list_edge_scatterer[i][0], list_edge_scatterer[i][1], list_edge_scatterer[i][2], color="blue")
        ax.plot3D(list_edge_absorber[i][0], list_edge_absorber[i][1], list_edge_absorber[i][2], color="blue")
    # plot cluster hits
    b = 5  # markersize scaling factor
    for i in range(len(list_cluster_x)):
        ax.plot3D(list_cluster_x[i], list_cluster_y[i], list_cluster_z[i],
                  "X", color="orange", markersize=event.RecoClusterEnergies_values[i] * b)
    # plot compton event
    ax.plot3D(event.MCPosition_e_first.x, event.MCPosition_e_first.y, event.MCPosition_e_first.z,
              "x", color="red", markersize=event.MCEnergy_e * b)
    ax.plot3D(event.MCPosition_p_first.x, event.MCPosition_p_first.y, event.MCPosition_p_first.z,
              "x", color="red", markersize=event.MCEnergy_p * b)
    # plot compton event path
    ax.plot3D(event.MCPosition_p.x, event.MCPosition_p.y, event.MCPosition_p.z, color="red", linestyle="--")
    ax.plot3D(event.MCPosition_e.x, event.MCPosition_e.y, event.MCPosition_e.z, color="green", linestyle="--")
    # plot source path
    a = 250  # path length scaling factor
    ax.plot3D([event.MCPosition_source.x, (event.MCPosition_source.x + a * event.MCDirection_source.x)],
              [event.MCPosition_source.y, (event.MCPosition_source.y + a * event.MCDirection_source.y)],
              [event.MCPosition_source.z, (event.MCPosition_source.z + a * event.MCDirection_source.z)],
              color="red")

    # plot source axis
    ax.plot3D([0, 270 + 46.8 / 2], [0, 0], [0, 0], color="black", linestyle="--")
    ax.set_title("Display: Event {} ({},\nType {},\nEnergy {:.2f})".format(n, ic_tag, event.MCSimulatedEventType, event.MCEnergy_Primary))
    # plt.legend()
    plt.show()
