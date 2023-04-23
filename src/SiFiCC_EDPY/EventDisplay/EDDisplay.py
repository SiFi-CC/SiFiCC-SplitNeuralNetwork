import numpy as np
import matplotlib.pyplot as plt

from src.SiFiCC_EDPY.EventDisplay import EDBuilder


# ----------------------------------------------------------------------------------------------------------------------


def get_digits(number):
    list_digits = []
    str_digit = str(number)
    for i in range(1, len(str_digit) + 1):
        list_digits.append(int(str_digit[-i]))
    if len(list_digits) == 1:
        list_digits.append(0)
    return list_digits


def display(event, detector):
    # ------------------------------------------------------------------------------------------------------------------
    # Main plotting, general settings of 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 300)
    ax.set_ylim(-155, 155)
    ax.set_zlim(-155, 155)
    ax.set_xlabel("x-axis [mm]")
    ax.set_ylabel("y-axis [mm]")
    ax.set_zlabel("z-axis [mm]")

    # ------------------------------------------------------------------------------------------------------------------
    # detector edges, orientation axis, (fiber hits)
    # get detector edges
    list_edge_scatterer = EDBuilder.get_edges(detector.scatterer.pos.x,
                                              detector.scatterer.pos.y,
                                              detector.scatterer.pos.z,
                                              detector.scatterer.dimx,
                                              detector.scatterer.dimy,
                                              detector.scatterer.dimz)
    list_edge_absorber = EDBuilder.get_edges(detector.absorber.pos.x,
                                             detector.absorber.pos.y,
                                             detector.absorber.pos.z,
                                             detector.absorber.dimx,
                                             detector.absorber.dimy,
                                             detector.absorber.dimz)

    # get detector hits
    list_cluster_x = []
    list_cluster_y = []
    list_cluster_z = []
    for cl in event.RecoClusterPosition:
        list_cluster_x.append(cl.x)
        list_cluster_y.append(cl.y)
        list_cluster_z.append(cl.z)

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

    # ------------------------------------------------------------------------------------------------------------------
    # plot primary gamma trajectory
    a = 250
    ax.plot3D([event.MCPosition_source.x, event.MCComptonPosition.x],
              [event.MCPosition_source.y, event.MCComptonPosition.y],
              [event.MCPosition_source.z, event.MCComptonPosition.z],
              color="red")
    ax.plot3D([event.MCComptonPosition.x, event.MCComptonPosition.x + a * event.MCDirection_scatter.x],
              [event.MCComptonPosition.y, event.MCComptonPosition.y + a * event.MCDirection_scatter.y],
              [event.MCComptonPosition.z, event.MCComptonPosition.z + a * event.MCDirection_scatter.z],
              color="red")
    # True source direction as control plot
    """
    ax.plot3D([event.MCPosition_source.x, event.MCPosition_source.x + a * event.MCDirection_source.x],
              [event.MCPosition_source.y, event.MCPosition_source.y + a * event.MCDirection_source.y],
              [event.MCPosition_source.z, event.MCPosition_source.z + a * event.MCDirection_source.z],
              color="pink")
    """
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    # Marker for MC-Truth (Later definition standard for Neural Network)
    ax.plot3D(event.MCPosition_e_first.x, event.MCPosition_e_first.y, event.MCPosition_e_first.z,
              "x", color="red", markersize=event.MCEnergy_e * b)
    ax.plot3D(event.MCPosition_p_first.x, event.MCPosition_p_first.y, event.MCPosition_p_first.z,
              "x", color="red", markersize=event.MCEnergy_p * b)
    ax.plot3D(event.MCPosition_source.x, event.MCPosition_source.y, event.MCPosition_source.z,
              "o", color="red", markersize=4)

    # ------------------------------------------------------------------------------------------------------------------
    # MC-Truth and CB-Reco compton cone
    # Grab Compton scattering angle from source and scattering directions as energy calculations are not good enough
    # on MC-Truth level
    v1_u = np.array([event.MCDirection_source.x, event.MCDirection_source.y, event.MCDirection_source.z])
    v2_u = np.array([event.MCDirection_scatter.x, event.MCDirection_scatter.y, event.MCDirection_scatter.z])
    dir_angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # Main vectors needed for cone calculations
    vec_ax1 = event.MCPosition_e_first
    vec_ax2 = event.MCPosition_p_first - event.MCPosition_e_first
    vec_src = event.MCPosition_source

    list_cone = EDBuilder.get_compton_cone(vec_ax1, vec_ax2, vec_src, dir_angle, sr=128)
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
    reco_theta = event.calc_theta_energy(e1, e2)
    offset = vec_ax1.x

    list_cone = EDBuilder.get_compton_cone(vec_ax1, vec_ax2 - vec_ax1, vec_src, reco_theta, sr=128)
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

    # ------------------------------------------------------------------------------------------------------------------
    # Control prints
    print("\nControl: ")
    print("True E Position: ({:.3f}, {:.3f}, {:.3f})".format(event.MCPosition_e_first.x,
                                                             event.MCPosition_e_first.y,
                                                             event.MCPosition_e_first.z))
    print("True P Position: ({:.3f}, {:.3f}, {:.3f})".format(event.MCPosition_p_first.x,
                                                             event.MCPosition_p_first.y,
                                                             event.MCPosition_p_first.z))
    print("\nCompton Scattering Angle theta:")
    print("theta (Energy):",
          "{:5.3f} [rad] | {:5.1f} [deg]".format(event.theta_energy, event.theta_energy * 360 / 2 / np.pi))
    print("theta (Vector):", "{:5.3f} [rad] | {:5.1f} [deg]".format(dir_angle, dir_angle * 360 / 2 / np.pi))

    # print("RETURNER:", event.check_absorber_interaction())

    # ------------------------------------------------------------------------------------------------------------------
    # title string
    dict_type = {2: "Real Coincidence",
                 3: "Random Coincidence",
                 5: "Real Coincidence + Pile-Up",
                 6: "Random Coincidence + Pile-Up"}
    str_tagging = str(event.is_real_coincidence * 1) + str(event.is_compton * 1) + str(
        event.is_compton_pseudo_complete * 1) + str(event.is_compton_pseudo_distributed * 1) + str(
        event.is_compton_distributed * 1) + str(event.is_ideal_compton * 1)

    ax.set_title(
        "Display: Event {} (Id: {})\nType: {}, {}\nEnergy e/p: {:.2f} MeV / {:.2f} MeV\nPrimary Energy: {:.2f} MeV\nTotal cluster energy: {:.2f} MeV".format(
            0,
            event.EventNumber, dict_type[event.MCSimulatedEventType], str_tagging,
            event.MCEnergy_e, event.MCEnergy_p, event.MCEnergy_Primary, np.sum(event.RecoClusterEnergies_values)))
    # plt.legend()
    plt.tight_layout()
    plt.show()
