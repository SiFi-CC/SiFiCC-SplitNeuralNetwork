import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from itertools import product, combinations
from src.utilities import print_event_summary


def get_digits(number):
    list_digits = []
    str_digit = str(number)
    for i in range(1, len(str_digit) + 1):
        list_digits.append(int(str_digit[-i]))
    if len(list_digits) == 1:
        list_digits.append(0)
    return list_digits


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


def surface_list(x, y, z, xdim, ydim, zdim):
    one = np.ones(4).reshape(2, 2)
    list_surface = [[[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z - zdim / 2) * one],
                    [[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z + zdim / 2) * one],
                    [[x - xdim / 2, x + xdim / 2], (y - ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
                    [[x - xdim / 2, x + xdim / 2], (y + ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
                    [(x - xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
                    [(x + xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]
    return list_surface


# ----------------------------------------------------------------------------------------------------------------------
# Main event display

def event_display(RootParser, event_position=None, event_id=None):
    # grab event from either root file position or event number (might be slow)
    if event_position is not None:
        event = RootParser.get_event(position=event_position)
        print_event_summary(RootParser, event_position)
    elif event_id is not None:
        for i, it_event in enumerate(RootParser.iterate_events(n=None)):
            if it_event.EventNumber == event_id:
                event = RootParser.get_event(position=i)
                print_event_summary(RootParser, i)
    else:
        print("Invalid event position/ID!")
        return

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
    ax.set_xlim(-10, 300)
    ax.set_ylim(-60, 60)
    ax.set_zlim(-60, 60)
    ax.set_xlabel("x-axis [mm]")
    ax.set_ylabel("y-axis [mm]")
    ax.set_zlabel("z-axis [mm]")
    # plot detector edges
    for i in range(len(list_edge_scatterer)):
        ax.plot3D(list_edge_scatterer[i][0], list_edge_scatterer[i][1], list_edge_scatterer[i][2], color="blue")
        ax.plot3D(list_edge_absorber[i][0], list_edge_absorber[i][1], list_edge_absorber[i][2], color="blue")
    # plot source axis
    ax.plot3D([0, 270 + 46.8 / 2], [0, 0], [0, 0], color="black", linestyle="--")
    # plot fiber hits + cluster hits
    b = 5  # marker-size scaling factor
    for i in range(len(list_cluster_x)):
        """
        # fiber hits
        list_surface = surface_list(list_cluster_x[i], 0, list_cluster_z[i], 1.3, 100.0, 1.3)
        for j in range(len(list_surface)):
            ax.plot_wireframe(*list_surface[i], alpha=0.5, color="green")
        """
        # cluster hits
        ax.plot3D(list_cluster_x[i], list_cluster_y[i], list_cluster_z[i],
                  "X", color="orange", markersize=event.RecoClusterEnergies_values[i] * b)

    # plot primary gamma trajectory
    ax.plot3D([event.MCPosition_source.x, event.MCComptonPosition.x],
              [event.MCPosition_source.y, event.MCComptonPosition.y],
              [event.MCPosition_source.z, event.MCComptonPosition.z],
              color="red")
    a = 250
    ax.plot3D([event.MCComptonPosition.x, event.MCComptonPosition.x + a * event.MCDirection_scatter.x],
              [event.MCComptonPosition.y, event.MCComptonPosition.y + a * event.MCDirection_scatter.y],
              [event.MCComptonPosition.z, event.MCComptonPosition.z + a * event.MCDirection_scatter.z],
              color="red")
    # electron interaction plotting
    list_e_interaction = [[]]
    counter = 0
    list_tmp = []
    for i in range(len(event.MCInteractions_e)):
        list_digit = get_digits(event.MCInteractions_e[i])
        if list_digit[1] == 1:
            list_e_interaction[0].append(i)
    list_e_interaction[0] = np.array(list_e_interaction[0])
    for i in range(2, 10):
        for j in range(len(event.MCInteractions_e)):
            list_digit = get_digits(event.MCInteractions_e[j])
            if list_digit[1] == i:
                list_tmp.append(j)
            elif len(list_tmp) != 0:
                if event.MCInteractions_e[list_tmp[0] - 1] > event.MCInteractions_e[list_tmp[0]]:
                    list_e_interaction[counter] = np.concatenate(
                        [list_e_interaction[counter], np.array(list_tmp)])
                else:
                    list_e_interaction.append(np.arange(list_tmp[0] - 1, list_tmp[-1] + 1, 1.0, dtype=int))
                    counter += 1
                list_tmp = []

    # plot secondary electron reaction chain
    for i in range(len(list_e_interaction)):
        for j in range(1, len(list_e_interaction[i])):
            ax.plot3D(
                [event.MCPosition_e.x[list_e_interaction[i][j - 1]], event.MCPosition_e.x[list_e_interaction[i][j]]],
                [event.MCPosition_e.y[list_e_interaction[i][j - 1]], event.MCPosition_e.y[list_e_interaction[i][j]]],
                [event.MCPosition_e.z[list_e_interaction[i][j - 1]], event.MCPosition_e.z[list_e_interaction[i][j]]],
                color="green", linestyle="--")

    # photon interaction plotting
    list_p_interaction = [[]]
    counter = 0
    list_tmp = []
    for i in range(len(event.MCInteractions_p)):
        if 0 < event.MCInteractions_p[i] < 10:
            list_p_interaction[0].append(i)
    list_p_interaction[0] = np.array(list_p_interaction[0])
    for i in range(1, 10):
        for j in range(len(event.MCInteractions_p)):
            list_digit = get_digits(event.MCInteractions_p[j])
            if list_digit[1] == i:
                list_tmp.append(j)
            elif len(list_tmp) != 0:
                if event.MCInteractions_p[list_tmp[0] - 1] > event.MCInteractions_p[list_tmp[0]]:
                    list_p_interaction[counter] = np.concatenate(
                        [list_p_interaction[counter], np.array(list_tmp)])
                else:
                    list_p_interaction.append(np.arange(list_tmp[0] - 1, list_tmp[-1] + 1, 1.0, dtype=int))
                    counter += 1
                list_tmp = []

    # plot secondary photon reaction chain
    for i in range(len(list_p_interaction)):
        for j in range(1, len(list_p_interaction[i])):
            ax.plot3D(
                [event.MCPosition_p.x[list_p_interaction[i][j - 1]], event.MCPosition_p.x[list_p_interaction[i][j]]],
                [event.MCPosition_p.y[list_p_interaction[i][j - 1]], event.MCPosition_p.y[list_p_interaction[i][j]]],
                [event.MCPosition_p.z[list_p_interaction[i][j - 1]], event.MCPosition_p.z[list_p_interaction[i][j]]],
                color="purple", linestyle="--")

    # plot compton event
    ax.plot3D(event.MCPosition_e_first.x, event.MCPosition_e_first.y, event.MCPosition_e_first.z,
              "x", color="red", markersize=event.MCEnergy_e * b)
    ax.plot3D(event.MCPosition_p_first.x, event.MCPosition_p_first.y, event.MCPosition_p_first.z,
              "x", color="red", markersize=event.MCEnergy_p * b)

    # title string
    dict_type = {2: "Real Coincidence",
                 3: "Random Coincidence",
                 5: "Real Coincidence + Pile-Up",
                 6: "Random Coincidence + Pile-Up"}

    ax.set_title(
        "Display: Event {} (Id: {})\nType: {}, {}\nEnergy e/p: {:.2f} MeV / {:.2f} MeV\nPrimary Energy: {:.2f} MeV\nTotal cluster energy: {:.2f} MeV".format(
            event_position,
            event.EventNumber, dict_type[event.MCSimulatedEventType], "TAGGING",
            event.MCEnergy_e, event.MCEnergy_p, event.MCEnergy_Primary, np.sum(event.RecoClusterEnergies_values)))
    # plt.legend()
    plt.show()
