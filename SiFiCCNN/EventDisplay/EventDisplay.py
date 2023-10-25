import matplotlib.pyplot as plt
import numpy as np

from uproot_methods import TVector3
from ..utils.physics import compton_scattering_angle, vector_angle

from SiFiCCNN.root import RootParser
from SiFiCCNN.EventDisplay import Utils


class EventDisplay:

    def __init__(self, ph_method="TRUE", coordinate_system="AACHEN"):
        # main plotting objects
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.event = None

        # internal variables
        self.ph_method = ph_method
        self.coordinate_system = coordinate_system

        # generate canvas
        self._generate_canvas()

    def set_ph_method(self, ph_method):
        self.ph_method = ph_method

    def set_coordinate_system(self, coordinate_system):
        self.coordinate_system = coordinate_system
        self._generate_canvas()

    def _generate_canvas(self):
        if self.coordinate_system == "CRACOW":
            self.ax.set_box_aspect(aspect=(3, 1, 1))
            self.ax.set_xlim3d(-10, 300)
            self.ax.set_ylim3d(-55, 55)
            self.ax.set_zlim3d(-55, 55)
            self.ax.set_xlabel("x-axis [mm]")
            self.ax.set_ylabel("y-axis [mm]")
            self.ax.set_zlabel("z-axis [mm]")

        if self.coordinate_system == "AACHEN":
            self.ax.set_box_aspect(aspect=(1, 1, 3))
            self.ax.set_xlim3d(-55, 55)
            self.ax.set_ylim3d(-55, 55)
            self.ax.set_zlim3d(-10, 300)
            self.ax.set_xlabel("x-axis [mm]")
            self.ax.set_ylabel("y-axis [mm]")
            self.ax.set_zlabel("z-axis [mm]")

    def load_event(self, event):
        self.event = event

    def _verify_event(self):
        if self.event is None:
            raise TypeError

    def draw_detector(self, color="blue"):
        self._verify_event()

        list_edge_scatterer = Utils.get_edges(self.event.scatterer.pos.x,
                                              self.event.scatterer.pos.y,
                                              self.event.scatterer.pos.z,
                                              self.event.scatterer.dimx,
                                              self.event.scatterer.dimy,
                                              self.event.scatterer.dimz)
        for i in range(len(list_edge_scatterer)):
            self.ax.plot3D(list_edge_scatterer[i][0],
                           list_edge_scatterer[i][1],
                           list_edge_scatterer[i][2],
                           color=color)
        list_edge_absorber = Utils.get_edges(self.event.absorber.pos.x,
                                             self.event.absorber.pos.y,
                                             self.event.absorber.pos.z,
                                             self.event.absorber.dimx,
                                             self.event.absorber.dimy,
                                             self.event.absorber.dimz)
        for i in range(len(list_edge_absorber)):
            self.ax.plot3D(list_edge_absorber[i][0],
                           list_edge_absorber[i][1],
                           list_edge_absorber[i][2],
                           color="blue")

    def draw_reference_axis(self):
        self._verify_event()

        endpoint = [0, 0, 0]
        if self.coordinate_system == "CRACOW":
            endpoint = [270 + 46.8 / 2, 0, 0]
        if self.coordinate_system == "AACHEN":
            endpoint = [0, 0, 270 + 46.8 / 2]

        self.ax.plot3D([0, endpoint[0]], [0, endpoint[1]], [0, endpoint[2]],
                       color="black",
                       linestyle="--")

    def draw_promptgamma(self):
        self._verify_event()

        a = 250
        self.ax.plot3D([self.event.MCPosition_source.x, self.event.MCComptonPosition.x],
                       [self.event.MCPosition_source.y, self.event.MCComptonPosition.y],
                       [self.event.MCPosition_source.z, self.event.MCComptonPosition.z],
                       color="red")

        self.ax.plot3D([self.event.MCComptonPosition.x,
                        self.event.MCComptonPosition.x + a * self.event.MCDirection_scatter.x],
                       [self.event.MCComptonPosition.y,
                        self.event.MCComptonPosition.y + a * self.event.MCDirection_scatter.y],
                       [self.event.MCComptonPosition.z,
                        self.event.MCComptonPosition.z + a * self.event.MCDirection_scatter.z],
                       color="red")

    def draw_interactions(self):
        self._verify_event()

        for pos in self.event.MCPosition_e:
            self.ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

        for pos in self.event.MCPosition_p:
            self.ax.plot3D(pos.x, pos.y, pos.z, ".", color="limegreen", markersize=10)

    def draw_cone_targets(self):
        self._verify_event()
        # target_energy_e, target_energy_p = self.event.get_target_energy()
        target_position_e, target_position_p = self.event.get_target_position(
            ph_method=self.ph_method)

        self.ax.plot3D(target_position_e.x, target_position_e.y, target_position_e.z,
                       "x", color="red", markersize=15, zorder=10)

        self.ax.plot3D(target_position_p.x, target_position_p.y, target_position_p.z,
                       "x", color="red", markersize=15, zorder=10)

        self.ax.plot3D(self.event.MCPosition_source.x, self.event.MCPosition_source.y,
                       self.event.MCPosition_source.z,
                       "o", color="red", markersize=4)

    def draw_cone_true(self):
        self._verify_event()

        # Main vectors needed for cone calculations
        target_energy_e, target_energy_p = self.event.get_target_energy()
        target_position_e, target_position_p = self.event.get_target_position()
        vec_ax1 = target_position_e
        vec_ax2 = target_position_p - target_position_e
        vec_src = self.event.MCPosition_source
        theta = vector_angle(vec_ax1 - vec_src, vec_ax2)

        list_cone = []
        if self.coordinate_system == "CRACOW":
            list_cone = Utils.get_compton_cone_cracow(vec_ax1, vec_ax2, vec_src, theta, sr=128)
        if self.coordinate_system == "AACHEN":
            list_cone = Utils.get_compton_cone_aachen(vec_ax1, vec_ax2, vec_src, theta, sr=128)

        for i in range(1, len(list_cone)):
            self.ax.plot3D([list_cone[i - 1][0], list_cone[i][0]],
                           [list_cone[i - 1][1], list_cone[i][1]],
                           [list_cone[i - 1][2], list_cone[i][2]],
                           color="black")
        for i in [8, 16, 32, 64]:
            self.ax.plot3D([vec_ax1.x, list_cone[i - 1][0]],
                           [vec_ax1.y, list_cone[i - 1][1]],
                           [vec_ax1.z, list_cone[i - 1][2]],
                           color="black")

    def draw_cluster_hits(self):
        self._verify_event()

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
            self.ax.plot3D(list_cluster_x[i], list_cluster_y[i], list_cluster_z[i],
                           "X", color="orange",
                           markersize=b)

    def draw_fibre_hits(self):
        self._verify_event()
        # fibre hits plus boxes
        for i in range(len(self.event.fibre_position)):
            self.ax.plot3D(self.event.fibre_position.x[i],
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
                self.ax.plot3D(list_fibre_edges[j][0],
                               list_fibre_edges[j][1],
                               list_fibre_edges[j][2],
                               color="lime")

    def draw_sipm_hits(self):
        self._verify_event()

        for i in range(len(self.event.SiPM_position)):
            list_sipm_edges = Utils.get_edges(self.event.SiPM_position.x[i],
                                              self.event.SiPM_position.y[i],
                                              self.event.SiPM_position.z[i],
                                              4.0,
                                              0,
                                              4.0)
            for j in range(len(list_sipm_edges)):
                self.ax.plot3D(list_sipm_edges[j][0],
                               list_sipm_edges[j][1],
                               list_sipm_edges[j][2],
                               color="darkgreen")

    @staticmethod
    def show():
        plt.show()
