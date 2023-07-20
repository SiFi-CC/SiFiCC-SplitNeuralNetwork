import matplotlib.pyplot as plt

from uproot_methods import TVector3
from ..utils.physics import compton_scattering_angle

from SiFiCCNN.EventDisplay import Utils


class Display:

    def __init__(self):
        # Main plotting, general settings of 3D plot
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_box_aspect(aspect=(3, 1, 1))

        self.ax.set_xlim3d(-10, 300)
        self.ax.set_ylim3d(-55, 55)
        self.ax.set_zlim3d(-55, 55)
        self.ax.set_xlabel("x-axis [mm]")
        self.ax.set_ylabel("y-axis [mm]")
        self.ax.set_zlabel("z-axis [mm]")


