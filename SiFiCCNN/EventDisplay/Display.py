import matplotlib.pyplot as plt
import numpy as np

from uproot_methods import TVector3
from ..utils.physics import compton_scattering_angle, vector_angle

from SiFiCCNN.root import RootParser
from SiFiCCNN.EventDisplay import Utils


class Display:

    def __init__(self, root_file):
        self.root_parser = RootParser.RootParser(root_file)
        self.event = None
        self.nnreco = None

        # control switches
        self.show_box_scatterer = True
        self.show_box_absorber = True
        self.show_reference_axis = True
        self.show_gamma_trajectory = True
        self.show_targets = True
        self.show_interaction_e = False
        self.show_interaction_p = False

        self.show_true_cone = False
        self.show_cbreco_cone = False
        self.show_added_cone = False

        self.show_cluster_hits = False
        self.show_sipm_hits = False

    def selector_index(self, idx):
        self.event = self.root_parser.get_event(idx)

    def set_nnreco(self, ary_reco):
        self.nnreco = ary_reco

    def summary(self):
        # check if a valid event was selected
        if self.event is None:
            print("Error! No Event selected for display!")
            return

        self.event.summary()

    def show(self):
        # check if a valid event was selected
        if self.event is None:
            print("Error! No Event selected for display!")
            return

        # ------------------------------------------
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

        # ----------------------------------------------
        # detector edges, orientation axis, (fiber hits)
        # get detector edges
        if self.show_box_scatterer:
            list_edge_scatterer = Utils.get_edges(self.event.scatterer.pos.x,
                                                  self.event.scatterer.pos.y,
                                                  self.event.scatterer.pos.z,
                                                  self.event.scatterer.dimx,
                                                  self.event.scatterer.dimy,
                                                  self.event.scatterer.dimz)
            for i in range(len(list_edge_scatterer)):
                ax.plot3D(list_edge_scatterer[i][0], list_edge_scatterer[i][1],
                          list_edge_scatterer[i][2], color="blue")

        if self.show_box_absorber:
            list_edge_absorber = Utils.get_edges(self.event.absorber.pos.x,
                                                 self.event.absorber.pos.y,
                                                 self.event.absorber.pos.z,
                                                 self.event.absorber.dimx,
                                                 self.event.absorber.dimy,
                                                 self.event.absorber.dimz)
            for i in range(len(list_edge_absorber)):
                ax.plot3D(list_edge_absorber[i][0], list_edge_absorber[i][1],
                          list_edge_absorber[i][2], color="blue")

        # ----------------------------------------------
        # plot reference axis
        if self.show_reference_axis:
            ax.plot3D([0, 270 + 46.8 / 2], [0, 0], [0, 0], color="black",
                      linestyle="--")

        # ---------------------------------------------
        # plot primary gamma trajectory
        if self.show_gamma_trajectory:
            a = 250
            ax.plot3D([self.event.MCPosition_source.x, self.event.MCComptonPosition.x],
                      [self.event.MCPosition_source.y, self.event.MCComptonPosition.y],
                      [self.event.MCPosition_source.z, self.event.MCComptonPosition.z],
                      color="red")

            ax.plot3D([self.event.MCComptonPosition.x,
                       self.event.MCComptonPosition.x + a * self.event.MCDirection_scatter.x],
                      [self.event.MCComptonPosition.y,
                       self.event.MCComptonPosition.y + a * self.event.MCDirection_scatter.y],
                      [self.event.MCComptonPosition.z,
                       self.event.MCComptonPosition.z + a * self.event.MCDirection_scatter.z],
                      color="red")

        # -----------------------------------------------------------------
        # electron and photon interactions
        if self.show_interaction_e:
            for pos in self.event.MCPosition_e:
                ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

        if self.show_interaction_p:
            for pos in self.event.MCPosition_p:
                ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

        # -----------------------------------------------------------------
        # Marker for MC-Truth (Later definition standard for Neural Network
        if self.show_targets:
            # target_energy_e, target_energy_p = self.event.get_target_energy()
            target_position_e, target_position_p = self.event.get_target_position()

            ax.plot3D(target_position_e.x, target_position_e.y, target_position_e.z,
                      "x", color="red", markersize=15)
            ax.plot3D(target_position_p.x, target_position_p.y, target_position_p.z,
                      "x", color="red", markersize=15)
            ax.plot3D(self.event.MCPosition_source.x, self.event.MCPosition_source.y,
                      self.event.MCPosition_source.z,
                      "o", color="red", markersize=4)

        # -----------------------------------
        # True and reconstructed Compton cone
        if self.show_true_cone:
            # Main vectors needed for cone calculations
            target_energy_e, target_energy_p = self.event.get_target_energy()
            target_position_e, target_position_p = self.event.get_target_position()
            vec_ax1 = target_position_e
            vec_ax2 = target_position_p - target_position_e
            vec_src = self.event.MCPosition_source
            theta = vector_angle(vec_ax1 - vec_src, vec_ax2)

            list_cone = Utils.get_compton_cone(vec_ax1, vec_ax2, vec_src, theta, sr=128)
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

        if self.show_cbreco_cone:
            reco_energy_e, reco_energy_p = self.event.get_reco_energy()
            reco_position_e, reco_position_p = self.event.get_reco_position()
            vec_ax1 = reco_position_e
            vec_ax2 = reco_position_p
            theta = compton_scattering_angle(reco_energy_e, reco_energy_p)
            vec_src = self.event.MCPosition_source

            list_cone = Utils.get_compton_cone(vec_ax1, vec_ax2 - vec_ax1, vec_src, theta, sr=128)
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

        # --------------------------------------
        # cluster detector hits
        if self.show_cluster_hits:
            list_cluster_x = []
            list_cluster_y = []
            list_cluster_z = []
            for cl in self.event.RecoClusterPosition:
                list_cluster_x.append(cl.x)
                list_cluster_y.append(cl.y)
                list_cluster_z.append(cl.z)

            # plot fiber hits + cluster hits
            b = 10  # marker-size scaling factor
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
                          markersize=b)

        # ----------------------------------------------
        # sipm and fibre hits

        if self.show_sipm_hits:
            # fibre hits plus boxes
            for i in range(len(self.event.fibre_position)):
                ax.plot3D(self.event.fibre_position.x[i],
                          self.event.fibre_position.y[i],
                          self.event.fibre_position.z[i],
                          "o",
                          color="lime")
                list_fibre_edges = Utils.get_edges(self.event.fibre_position.x[i],
                                                   0,
                                                   self.event.fibre_position.z[i],
                                                   1.94,
                                                   100,
                                                   1.94)
                for j in range(len(list_fibre_edges)):
                    ax.plot3D(list_fibre_edges[j][0],
                              list_fibre_edges[j][1],
                              list_fibre_edges[j][2],
                              color="lime")

            for i in range(len(self.event.SiPM_position)):
                list_sipm_edges = Utils.get_edges(self.event.SiPM_position.x[i],
                                                  self.event.SiPM_position.y[i],
                                                  self.event.SiPM_position.z[i],
                                                  4.0,
                                                  0,
                                                  4.0)
                for j in range(len(list_sipm_edges)):
                    ax.plot3D(list_sipm_edges[j][0],
                              list_sipm_edges[j][1],
                              list_sipm_edges[j][2],
                              color="darkgreen")

        plt.show()
