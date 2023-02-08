import numpy as np
import matplotlib.pyplot as plt
from uproot_methods import TVector3


class ComptonCone:
    def __init__(self, e1, e2, x1, y1, z1, x2, y2, z2):
        self.theta = calculate_theta(e1, e2)
        self.axis = connect_points(TVector3(x1, y1, z1), TVector3(x2, y2, z2))
        self.apex = TVector3(x1, y1, z1)

    # TODO: energy check
    # TODO: angle check
    # TODO: axis check


def connect_points(vec1, vec2):
    """
    Coordinates of vector connecting two given points.
    Used to find compton cones axis. Point 1 represents the interaction in the scatterer and
    point 2 is the interaction in the absorber.

    Args:
        vec1 (TVector3): coordinates first point
        vec2 (TVector3): coordinates second point

    """

    vec = vec2 - vec1
    vec /= vec.mag
    return vec


def calculate_theta(e1, e2):
    """
    Calculate scattering angle theta in radiants.

    Args:
         e1 (double): Initial gamma energy
         e2 (double): Gamma energy after compton scattering
    """
    kMe = 0.510999  # MeV/c^2
    costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / (e1 + e2))

    # TODO: reformat python numpy arcos function
    theta = np.arccos(costheta)  # rad
    return theta


def reconstruct_image(ary_e1, ary_e2, ary_x1, ary_y1, ary_z1, ary_x2, ary_y2, ary_z2):
    """
    Method for image reconstruction. 2D histogram will be set as final image.
    Reconstructed image cone will be checked for intersection with image pixels
    of the image plane.

    Summary of the algorithm:
    Source: https://bragg.if.uj.edu.pl/gccbwiki//images/0/0e/KR_20170222_CCandCarbonLine.pdf
    Author: Katarzyna Rusiecka

    Vector of image pixel center ad cone apex will be calculated. If angle between vector
    and cone axis is equal to scattering angle theta, then pixel value will be increased. Else continue.
    Repeated for all pixels
    """
    entries = len(ary_e1)

    # histogram settings (in this case a 2d-array)
    # detector dimensions are hardcoded at the moment!
    scatz = 60.0
    scaty = 30.0
    nbinsz = int(scatz)
    nbinsy = int(scaty)
    zlimit = scatz / 2.0
    ylimit = scaty / 2.0
    widthz = zlimit * 2 / nbinsz
    widthy = ylimit * 2 / nbinsy

    A = 1
    D = 150

    # histogram
    ary_image = np.zeros(shape=(nbinsz, nbinsy))

    for i in range(entries):
        if ary_e1[i] == 0.0 or ary_e2[i] == 0.0:
            continue

        # print("processing event " + str(i))
        cone = ComptonCone(ary_e1[i],
                           ary_e2[i],
                           ary_x1[i],
                           ary_y1[i],
                           ary_z1[i],
                           ary_x2[i],
                           ary_y2[i],
                           ary_z2[i])

        for z in range(nbinsz):
            for y in range(nbinsy):
                pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy), -zlimit + widthz / 2 + (z * widthz))
                linkingVector = connect_points(pixelCenter, cone.apex)
                angle = cone.axis.angle(linkingVector)

                resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))

                if abs(cone.theta - angle) <= resolution:
                    ary_image[z, y] += 1

    return ary_image


def plot_backprojection(image, figure_name):
    # rotate origional image by 90 degrees
    image = np.rot90(image)
    proj = np.sum(image, axis=0)

    xticks = np.arange(0, image.shape[1], 5.0)
    xlabels = xticks - image.shape[1] / 2
    yticks = np.arange(0, image.shape[0], 5.0)
    ylabels = yticks - image.shape[0] / 2

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    axs[0].imshow(image)
    axs[0].set_yticks(yticks, ylabels)
    axs[0].set_aspect('auto')
    # axs[0].set(xlim=(0, image.shape[1]), ylim=(0, image.shape[0]))
    plt.tick_params('x', labelbottom=False)

    axs[1].set_xticks(xticks, xlabels)
    axs[1].set_aspect('auto')
    # axs[1].set(xlim=(0 , image.shape[1]), ylim=(0, max(proj)))
    axs[1].set_xlabel("z-position [mm]")
    axs[1].plot(proj, color="black")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
