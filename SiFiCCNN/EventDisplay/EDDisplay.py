import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from SiFiCCNN.EventDisplay import EDBuilder
from SiFiCCNN.root import RootLogger


def display(event):
    # grab needed attributes from event object
    target_energy_e, target_energy_p = event.get_target_energy()
    target_position_e, target_position_p = event.get_target_position()

    # ----------------------------------------------------------------------------------------------
    # Main plotting, general settings of 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect=(3, 1, 1))

    ax.set_xlim3d(-10, 300)
    ax.set_ylim3d(-55, 55)
    ax.set_zlim3d(-55, 55)
    ax.set_xlabel("x-axis [mm]")
    ax.set_ylabel("y-axis [mm]")
    ax.set_zlabel("z-axis [mm]")

    # ----------------------------------------------------------------------------------------------
    # detector edges, orientation axis, (fiber hits)
    # get detector edges
    list_edge_scatterer = EDBuilder.get_edges(event.scatterer.pos.x,
                                              event.scatterer.pos.y,
                                              event.scatterer.pos.z,
                                              event.scatterer.dimx,
                                              event.scatterer.dimy,
                                              event.scatterer.dimz)
    list_edge_absorber = EDBuilder.get_edges(event.absorber.pos.x,
                                             event.absorber.pos.y,
                                             event.absorber.pos.z,
                                             event.absorber.dimx,
                                             event.absorber.dimy,
                                             event.absorber.dimz)

    for i in range(len(list_edge_scatterer)):
        ax.plot3D(list_edge_scatterer[i][0], list_edge_scatterer[i][1],
                  list_edge_scatterer[i][2], color="blue")
        ax.plot3D(list_edge_absorber[i][0], list_edge_absorber[i][1],
                  list_edge_absorber[i][2], color="blue")
    # plot reference axis
    ax.plot3D([0, 270 + 46.8 / 2], [0, 0], [0, 0], color="black",
              linestyle="--")

    # ----------------------------------------------------------------------------------------------
    # plot primary gamma trajectory
    a = 250
    ax.plot3D([event.MCPosition_source.x, event.MCComptonPosition.x],
              [event.MCPosition_source.y, event.MCComptonPosition.y],
              [event.MCPosition_source.z, event.MCComptonPosition.z],
              color="red")
    """    
    ax.plot3D([event.MCPosition_source.x,
               event.MCPosition_source.x + a * event.MCDirection_source.x],
              [event.MCPosition_source.y,
               event.MCPosition_source.y + a * event.MCDirection_source.y],
              [event.MCPosition_source.z,
               event.MCPosition_source.z + a * event.MCDirection_source.z],
              color="purple")
    """
    ax.plot3D([event.MCComptonPosition.x,
               event.MCComptonPosition.x + a * event.MCDirection_scatter.x],
              [event.MCComptonPosition.y,
               event.MCComptonPosition.y + a * event.MCDirection_scatter.y],
              [event.MCComptonPosition.z,
               event.MCComptonPosition.z + a * event.MCDirection_scatter.z],
              color="red")
    # True source direction as control plot
    """
    ax.plot3D([event.MCPosition_source.x, event.MCPosition_source.x + a * event.MCDirection_source.x],
              [event.MCPosition_source.y, event.MCPosition_source.y + a * event.MCDirection_source.y],
              [event.MCPosition_source.z, event.MCPosition_source.z + a * event.MCDirection_source.z],
              color="pink")
    """
    # ----------------------------------------------------------------------------------------------
    # electron interaction plotting
    list_e_interaction, list_p_interaction = EDBuilder.get_interaction(
        event.MCInteractions_e, event.MCInteractions_p)

    # plot secondary electron reaction chain
    for i in range(len(list_e_interaction)):
        for j in range(1, len(list_e_interaction[i])):
            ax.plot3D(
                [event.MCPosition_e.x[list_e_interaction[i][j - 1]],
                 event.MCPosition_e.x[list_e_interaction[i][j]]],
                [event.MCPosition_e.y[list_e_interaction[i][j - 1]],
                 event.MCPosition_e.y[list_e_interaction[i][j]]],
                [event.MCPosition_e.z[list_e_interaction[i][j - 1]],
                 event.MCPosition_e.z[list_e_interaction[i][j]]],
                color="green", linestyle="--")
    # plot secondary photon reaction chain
    for i in range(len(list_p_interaction)):
        for j in range(1, len(list_p_interaction[i])):
            ax.plot3D(
                [event.MCPosition_p.x[list_p_interaction[i][j - 1]],
                 event.MCPosition_p.x[list_p_interaction[i][j]]],
                [event.MCPosition_p.y[list_p_interaction[i][j - 1]],
                 event.MCPosition_p.y[list_p_interaction[i][j]]],
                [event.MCPosition_p.z[list_p_interaction[i][j - 1]],
                 event.MCPosition_p.z[list_p_interaction[i][j]]],
                color="purple", linestyle="--")

    # ----------------------------------------------------------------------------------------------
    # Marker for MC-Truth (Later definition standard for Neural Network
    ax.plot3D(target_position_e.x, target_position_e.y, target_position_e.z,
              "x", color="red", markersize=event.MCEnergy_e * 10)
    ax.plot3D(target_position_p.x, target_position_p.y, target_position_p.z,
              "x", color="red", markersize=event.MCEnergy_p * 10)
    ax.plot3D(event.MCPosition_source.x, event.MCPosition_source.y,
              event.MCPosition_source.z,
              "o", color="red", markersize=4)

    # ----------------------------------------------------------------------------------------------
    # SiPM and Fibre hits
    # Only drawn if event file contains SiPM and Fibre information

    if event.bsipm:
        # fibre hits plus boxes
        for i in range(len(event.fibre_position)):
            ax.plot3D(event.fibre_position.x[i], event.fibre_position.y[i],
                      event.fibre_position.z[i], "o",
                      color="lime")
            list_fibre_edges = EDBuilder.get_edges(event.fibre_position.x[i], 0,
                                                   event.fibre_position.z[i],
                                                   1.94, 100, 1.94)
            for j in range(len(list_fibre_edges)):
                ax.plot3D(list_fibre_edges[j][0], list_fibre_edges[j][1],
                          list_fibre_edges[j][2], color="lime")

        for i in range(len(event.SiPM_position)):
            list_sipm_edges = EDBuilder.get_edges(event.SiPM_position.x[i],
                                                  event.SiPM_position.y[i],
                                                  event.SiPM_position.z[i],
                                                  4.0, 0, 4.0)
            for j in range(len(list_sipm_edges)):
                ax.plot3D(list_sipm_edges[j][0], list_sipm_edges[j][1],
                          list_sipm_edges[j][2], color="darkgreen")

    # ----------------------------------------------------------------------------------------------
    # True and reconstructed compton cone

    if event.bglobal:
        # Grab Compton scattering angle from source and scattering directions as energy calculations
        # are not good enough on MC-Truth level

        # Main vectors needed for cone calculations
        vec_ax1 = target_position_e
        vec_ax2 = target_position_p - target_position_e
        vec_src = event.MCPosition_source

        list_cone = EDBuilder.get_compton_cone(vec_ax1, vec_ax2, vec_src,
                                               event.theta_dotvec, sr=128)
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

    if event.bcluster:
        # Compton cone definition, defined by reco cluster positions
        vec_ax1, _ = event.get_electron_position()
        vec_ax2, _ = event.get_photon_position()
        e1, _ = event.get_electron_energy()
        e2, _ = event.get_photon_energy()
        reco_theta = event.calc_theta_energy(e1, e2)
        vec_src = event.MCPosition_source

        list_cone = EDBuilder.get_compton_cone(vec_ax1, vec_ax2 - vec_ax1,
                                               vec_src, reco_theta, sr=128)
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
    # get detector hits
    if event.bcluster:
        list_cluster_x = []
        list_cluster_y = []
        list_cluster_z = []
        for cl in event.RecoClusterPosition:
            list_cluster_x.append(cl.x)
            list_cluster_y.append(cl.y)
            list_cluster_z.append(cl.z)

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
                      "X", color="orange",
                      markersize=event.RecoClusterEnergies_values[i] * b)
    # print("RETURNER:", event.check_absorber_interaction())
    """
    # ----------------------------------------------------------------------------------------------
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
    """

    plt.tight_layout()
    plt.show()
