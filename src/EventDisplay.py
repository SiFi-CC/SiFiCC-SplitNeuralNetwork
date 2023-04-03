import numpy as np
import matplotlib.pyplot as plt

from src.utilities import print_event_summary

from scipy.spatial.transform import Rotation as R
from uproot_methods.classes.TVector3 import TVector3


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


def unit_vec(vec):
    return vec / np.sqrt(np.dot(vec, vec))


def vec_angle(vec1, vec2):
    return np.arccos(np.clip(np.dot(unit_vec(vec1), unit_vec(vec2)), -1.0, 1.0))


def cone_point(vec_ax1, vec_ax2, theta, offset, sr=8):
    # correct angle theta
    theta = np.pi - theta

    # define rotation axis in reference system of compton scattering (vec_ax1) as origin
    rot_axis = vec_ax2 - vec_ax1

    # rotate reference vector around scattering angle theta
    ref_vec = np.array([1, 0, 0])
    rotation_y = R.from_rotvec((rot_axis.theta - np.pi / 2 - theta) * np.array([0, 1, 0]))
    rotation_z = R.from_rotvec(rot_axis.phi * np.array([0, 0, 1]))
    ref_vec = rotation_y.apply(ref_vec)
    ref_vec = rotation_z.apply(ref_vec)

    # rotate reference vector around axis vector to sample cone edges
    list_cone_vec = []
    rot_axis_ary = np.array([rot_axis.x, rot_axis.y, rot_axis.z])
    # phi angle sampling (not the same as scattering angle theta!)
    list_phi = np.linspace(0, 360, sr)
    for angle in list_phi:
        vec_temp = ref_vec
        rot_vec = np.radians(angle) * rot_axis_ary / np.sqrt(np.dot(rot_axis_ary, rot_axis_ary))
        rot_M = R.from_rotvec(rot_vec)
        vec_temp = rot_M.apply(vec_temp)
        list_cone_vec.append(vec_temp)

    # scale each cone vector to hit the final canvas
    # shift them to correct final position

    for i in range(len(list_cone_vec)):
        a = -offset / list_cone_vec[i][0]
        list_cone_vec[i] *= a
        list_cone_vec[i] = np.array([list_cone_vec[i][0] + vec_ax1.x,
                                     list_cone_vec[i][1] + vec_ax1.y,
                                     list_cone_vec[i][2] + vec_ax1.z])

    return list_cone_vec


def define_cone_points(vec_init, axis, sr=8):
    list_angles = np.linspace(0, 360, sr)
    list_points = []
    for angle in list_angles:
        vec = vec_init
        rot_vec = np.radians(angle) * axis / np.sqrt(np.dot(axis, axis))
        rotation = R.from_rotvec(rot_vec)
        rotated_vec = rotation.apply(vec)
        list_points.append(rotated_vec)
    return list_points


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
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(-300, 300)
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
    ax.plot3D(event.MCPosition_source.x, event.MCPosition_source.y, event.MCPosition_source.z,
              "o", color="red", markersize=4)

    # Compton cone definition
    # REFERENCE VECTOR
    v1_u = np.array([event.MCDirection_source.x, event.MCDirection_source.y, event.MCDirection_source.z])
    v2_u = np.array([event.MCDirection_scatter.x, event.MCDirection_scatter.y, event.MCDirection_scatter.z])
    dir_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    vec_ax1 = event.MCPosition_e_first
    vec_ax2 = event.MCPosition_p_first
    rot_axis = vec_ax2 - vec_ax1
    offset = event.MCPosition_e_first.x - event.MCPosition_source.x

    list_cone = cone_point(vec_ax1, vec_ax2, event.theta, offset, sr=64)
    for i in range(1, len(list_cone)):
        ax.plot3D([list_cone[i - 1][0], list_cone[i][0]],
                  [list_cone[i - 1][1], list_cone[i][1]],
                  [list_cone[i - 1][2], list_cone[i][2]],
                  color="black")
    for i in [8, 16, 32, 64]:
        ax.plot3D([vec_ax1.x, list_cone[i - 1][0]],
                  [vec_ax1.y, list_cone[i - 1][1]],
                  [vec_ax1.z, list_cone[i - 1][2]],
                  color="black")

    # Compton cone definition, defined by reco cluster positions
    vec_ax1, _ = event.get_electron_position()
    vec_ax2, _ = event.get_photon_position()
    e1, _ = event.get_electron_energy()
    e2, _ = event.get_photon_energy()
    reco_theta = event.calculate_theta(e1, e2)
    rot_axis = vec_ax2 - vec_ax1
    offset = vec_ax1.x

    list_cone = cone_point(vec_ax1, vec_ax2, reco_theta, offset, sr=64)
    for i in range(1, len(list_cone)):
        ax.plot3D([list_cone[i - 1][0], list_cone[i][0]],
                  [list_cone[i - 1][1], list_cone[i][1]],
                  [list_cone[i - 1][2], list_cone[i][2]],
                  color="orange", linestyle="--")
    for i in [8, 16, 32, 64]:
        ax.plot3D([vec_ax1.x, list_cone[i - 1][0]],
                  [vec_ax1.y, list_cone[i - 1][1]],
                  [vec_ax1.z, list_cone[i - 1][2]],
                  color="orange", linestyle="--")
    ax.plot3D([vec_ax1.x, vec_ax2.x],
              [vec_ax1.y, vec_ax2.y],
              [vec_ax1.z, vec_ax2.z],
              color="orange", linestyle="--")

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
