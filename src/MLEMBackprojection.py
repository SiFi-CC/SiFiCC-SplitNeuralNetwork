import numpy as np
import matplotlib.pyplot as plt
from uproot_methods import TVector3
from src import CBSelector

from matplotlib import colors


def tmath_acos(x):
    """
    Alternative version to np.arccos, as it is unclear how it handles
    parameter outside the defined arccos range. This definition is equivalent
    to the root tmath version
    """
    if x < -1:
        return np.pi
    if x > 1:
        return 0
    return np.arccos(x)


class ComptonCone:
    def __init__(self, e1, e2, x1, y1, z1, x2, y2, z2, theta=None):
        # constructing cone positions
        self.axis = connect_points(TVector3(x1, y1, z1), TVector3(x2, y2, z2))
        self.apex = TVector3(x1, y1, z1)
        # constructing cone angle
        if theta is None:
            self.theta = calculate_theta(e1, e2)
        else:
            self.theta = theta


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
    Calculate scattering angle theta in radiant.

    Args:
         e1 (double): Initial gamma energy
         e2 (double): Gamma energy after compton scattering
    """
    kMe = 0.510999  # MeV/c^2
    costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / (e1 + e2))

    # define physical exceptions
    if e1 == 0.0 or e2 == 0.0:
        return 0.0

    if abs(costheta) > 1:
        return 0.0

    # theta returned wia root tmath::acos argument
    theta = tmath_acos(costheta)  # rad

    return theta


def reconstruct_image(ary_e1, ary_e2, ary_x1, ary_y1, ary_z1, ary_x2, ary_y2, ary_z2,
                      scatz=100.0, scaty=40.0,
                      ary_theta=None,
                      apply_filter=False):
    """
    Method for image reconstruction. 2D histogram will be set as final image.
    Reconstructed image cone will be checked for intersection with image pixels
    of the image plane.

    Summary of the algorithm:
    Source: https://bragg.if.uj.edu.pl/gccbwiki//images/0/0e/KR_20170222_CCandCarbonLine.pdf

    Vector of image pixel center ad cone apex will be calculated. If angle between vector
    and cone axis is equal to scattering angle theta, then pixel value will be increased. Else continue.
    Repeated for all pixels

    Args:
        ary_e1      (numpy array): energy electron
        ary_e2      (numpy array): energy photon
        ary_x1      (numpy array): x position of electron
        ary_y1      (numpy array): y position of electron
        ary_z1      (numpy array): z position of electron
        ary_x2      (numpy array): x position of photon
        ary_y2      (numpy array): y position of photon
        ary_z2      (numpy array): z position of photon
        ary_theta   (numpy array): theta angle of compton scattering. If given, overrides energy ary parameters
                                   for cone class
        apply_filter    (Boolean): If true, events are filtered by compton kinematics and DAC filter
    Return:
        ary_image: (nbinsy, nbinsz) dimensional array
    """

    # histogram settings (in this case a 2d-array)
    # detector dimensions are hardcoded at the moment!
    entries = len(ary_e1)
    nbinsz = int(scatz)
    nbinsy = int(scaty)
    zlimit = scatz / 2.0
    ylimit = scaty / 2.0
    widthz = zlimit * 2 / nbinsz
    widthy = ylimit * 2 / nbinsy

    # Quantities defining scatterer plane
    A = 1
    D = 150

    # histogram
    ary_image = np.zeros(shape=(nbinsz, nbinsy))

    for i in range(entries):
        if apply_filter:
            if not CBSelector.check_compton_arc(ary_e1[i], ary_e2[i]):
                continue
            if not CBSelector.check_compton_kinematics(ary_e1[i], ary_e2[i]):
                continue
            if not CBSelector.beam_origin(ary_e1[i], ary_e2[i], ary_x1[i], ary_y1[i], ary_z1[i], ary_x2[i], ary_y2[i],
                                          ary_z2[i], beam_diff=20, inverse=False):
                continue

        # create cone object based on theta parameter
        if ary_theta is None:
            cone = ComptonCone(ary_e1[i], ary_e2[i],
                               ary_x1[i], ary_y1[i], ary_z1[i],
                               ary_x2[i], ary_y2[i], ary_z2[i],
                               theta=None)
        else:
            cone = ComptonCone(ary_e1[i], ary_e2[i],
                               ary_x1[i], ary_y1[i], ary_z1[i],
                               ary_x2[i], ary_y2[i], ary_z2[i],
                               theta=ary_theta[i])

        # grab exceptions from scattering angle
        if cone.theta == 0.0:
            continue

        for z in range(nbinsz):
            for y in range(nbinsy):
                pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy), -zlimit + widthz / 2 + (z * widthz))
                linkingVector = connect_points(pixelCenter, cone.apex)
                angle = cone.axis.angle(linkingVector)

                resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))

                if abs(cone.theta - angle) <= resolution:
                    ary_image[z, y] += 1

    return ary_image


def reconstruct_image_optimized(ary_e1, ary_e2, ary_x1, ary_y1, ary_z1, ary_x2, ary_y2, ary_z2,
                                scatz=100.0, scaty=40.0,
                                ary_theta=None,
                                apply_filter=False):
    # histogram settings (in this case a 2d-array)
    # detector dimensions are hardcoded at the moment!
    entries = len(ary_e1)
    nbinsz = int(scatz)
    nbinsy = int(scaty)
    zlimit = scatz / 2.0
    ylimit = scaty / 2.0
    widthz = zlimit * 2 / nbinsz
    widthy = ylimit * 2 / nbinsy

    # Quantities defining scatterer plane
    A = 1
    D = 150

    # histogram
    ary_image = np.zeros(shape=(nbinsz, nbinsy))

    for i in range(entries):
        if apply_filter:
            if not CBSelector.check_compton_arc(ary_e1[i], ary_e2[i]):
                continue
            if not CBSelector.check_compton_kinematics(ary_e1[i], ary_e2[i]):
                continue
            if not CBSelector.beam_origin(ary_e1[i], ary_e2[i], ary_x1[i], ary_y1[i], ary_z1[i], ary_x2[i], ary_y2[i],
                                          ary_z2[i], beam_diff=20, inverse=False):
                continue

        # create cone object based on theta parameter
        if ary_theta is None:
            cone = ComptonCone(ary_e1[i], ary_e2[i],
                               ary_x1[i], ary_y1[i], ary_z1[i],
                               ary_x2[i], ary_y2[i], ary_z2[i],
                               theta=None)
        else:
            cone = ComptonCone(ary_e1[i], ary_e2[i],
                               ary_x1[i], ary_y1[i], ary_z1[i],
                               ary_x2[i], ary_y2[i], ary_z2[i],
                               theta=ary_theta[i])

        # grab exceptions from scattering angle
        if cone.theta == 0.0:
            continue

        # Here: new optimized image reconstruction method
        #       only the minimal amount of pixels needed are scanned
        list_pixel_cache = []
        ary_image_temp = np.zeros(shape=(nbinsz, nbinsy))
        ary_map = np.zeros(shape=(nbinsz, nbinsy))

        # optimized sampling of z-dimension
        # zbin_sampling = np.arange(0, nbinsz, 4, dtype=int)
        zbin_sampling = []
        for i in range(nbinsz):
            zbin_step = int(nbinsz / 2 + (i + 1) ** 2)
            if zbin_step < nbinsz:
                zbin_sampling.append(int(nbinsz / 2 + (i + 1) ** 2))
                zbin_sampling.append(int(nbinsz / 2 - (i + 1) ** 2))
            else:
                break

        for z in zbin_sampling:
            for y in range(nbinsy):
                ary_map[z, y] = 1
                pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy), -zlimit + widthz / 2 + (z * widthz))
                linkingVector = connect_points(pixelCenter, cone.apex)
                angle = cone.axis.angle(linkingVector)
                resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))

                if abs(cone.theta - angle) <= resolution:
                    ary_image[z, y] += 1
                    ary_image_temp[z, y] += 1
                    break_cond = True
                    list_pixel_cache.append((z, y))

        for pixel in list_pixel_cache:
            zmax_range = min(pixel[0] + 2, nbinsz)
            ymax_range = min(pixel[1] + 2, nbinsy)
            for z in range(pixel[0] - 1, zmax_range):
                if not 0 <= z < nbinsz:
                    continue
                for y in range(pixel[1] - 1, ymax_range):
                    if ary_map[z, y] == 1:
                        continue
                    if not 0 <= y <= nbinsy:
                        continue
                    else:
                        ary_map[z, y] = 1
                        pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy),
                                               -zlimit + widthz / 2 + (z * widthz))
                        linkingVector = connect_points(pixelCenter, cone.apex)
                        angle = cone.axis.angle(linkingVector)
                        resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))

                        if abs(cone.theta - angle) <= resolution:
                            ary_image[z, y] += 1
                            ary_image_temp[z, y] += 1
                            list_pixel_cache.append((z, y))
        """
        cmap = colors.ListedColormap(['white', 'red'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, axs = plt.subplots(2, 1)
        axs[0].imshow(ary_image_temp)
        axs[1].imshow(ary_map, cmap=cmap, norm=norm)
        plt.show()
        """

    return ary_image


def get_projection(image):
    # rotate original image by 90 degrees
    # This is purely done for presentational purpose
    image = np.rot90(image)
    ary_proj = np.sum(image, axis=0)
    return ary_proj


# ----------------------------------------------------------------------------------------------------------------------
# Back-projection scripts

def get_backprojection_cbreco(ary_cb_reco, ary_tagging, f_sample=1.0, n_subsample=1, scatz=100.0, scaty=40.0,
                              verbose=0):
    # Grab the total number of positive events
    # Apply the scaling factor to the positive events to generate a sub-sample
    # repeat for n_subsample  and average the back-projection of every sub-sample
    n_pos = np.sum(ary_tagging)
    n_pos_subsample = int(n_pos * f_sample)
    ary_proj = np.zeros(shape=(int(scatz),))

    for i in range(n_subsample):
        ary_idx = np.arange(0, len(ary_cb_reco), 1.0, dtype=int)
        ary_idx_pos = ary_idx[ary_tagging == 1]

        rng = np.random.default_rng(42)
        rng.shuffle(ary_idx_pos)

        # generate back-projection image
        ary_image = reconstruct_image(ary_cb_reco[ary_idx_pos[:n_pos_subsample], 1],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 2],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 3],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 4],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 5],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 6],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 7],
                                      ary_cb_reco[ary_idx_pos[:n_pos_subsample], 8],
                                      scatz=scatz, scaty=scaty,
                                      apply_filter=True)
        ary_proj += get_projection(ary_image)
        if verbose == 1:
            print("Back-projection done for {} events of sub-sample {}".format(n_pos_subsample, i))

    # mean value of all sub-samples
    ary_proj /= n_subsample
    return ary_proj


def get_backprojection_cbreco_optimized(ary_cb_reco, ary_tagging, f_sample=1.0, n_subsample=1, scatz=100.0, scaty=40.0,
                                        verbose=0):
    # Grab the total number of positive events
    # Apply the scaling factor to the positive events to generate a sub-sample
    # repeat for n_subsample  and average the back-projection of every sub-sample
    n_pos = np.sum(ary_tagging)
    n_pos_subsample = int(n_pos * f_sample)
    ary_proj = np.zeros(shape=(int(scatz),))

    for i in range(n_subsample):
        ary_idx = np.arange(0, len(ary_cb_reco), 1.0, dtype=int)
        ary_idx_pos = ary_idx[ary_tagging == 1]

        rng = np.random.default_rng(42)
        rng.shuffle(ary_idx_pos)

        # generate back-projection image
        ary_image = reconstruct_image_optimized(ary_cb_reco[ary_idx_pos[:n_pos_subsample], 1],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 2],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 3],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 4],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 5],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 6],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 7],
                                                ary_cb_reco[ary_idx_pos[:n_pos_subsample], 8],
                                                scatz=scatz, scaty=scaty,
                                                apply_filter=True)
        ary_proj += get_projection(ary_image)
        if verbose == 1:
            print("Back-projection done for {} events of sub-sample {}".format(n_pos_subsample, i))

    # mean value of all sub-samples
    ary_proj /= n_subsample
    return ary_proj


def get_backprojection_nnpred_optimized(ary_nn_pred, ary_score, theta=0.5, f_sample=1.0, n_subsample=1,
                                        scatz=100.0, scaty=40.0,
                                        verbose=0):
    # Grab the total number of positive events
    # Apply the scaling factor to the positive events to generate a sub-sample
    # repeat for n_subsample  and average the back-projection of every sub-sample
    n_pos = np.sum((ary_score > theta) * 1)
    n_pos_subsample = int(n_pos * f_sample)
    ary_proj = np.zeros(shape=(int(scatz),))

    for i in range(n_subsample):
        ary_idx = np.arange(0, len(ary_nn_pred), 1.0, dtype=int)
        ary_idx_pos = ary_idx[ary_score > theta]

        rng = np.random.default_rng(42)
        rng.shuffle(ary_idx_pos)

        # generate back-projection image
        ary_image = reconstruct_image_optimized(ary_nn_pred[ary_idx_pos[:n_pos_subsample], 1],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 2],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 3],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 4],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 5],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 6],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 7],
                                                ary_nn_pred[ary_idx_pos[:n_pos_subsample], 8],
                                                scatz=scatz, scaty=scaty,
                                                apply_filter=True)
        ary_proj += get_projection(ary_image)
        if verbose == 1:
            print("Back-projection done for {} events of sub-sample {}".format(n_pos_subsample, i))

    # mean value of all sub-samples
    ary_proj /= n_subsample
    return ary_proj


# ----------------------------------------------------------------------------------------------------------------------
# plotting

# TODO: Summarize this and stacked plot

def plot_backprojection_dual(list_proj1, list_proj2,
                             list_labels,
                             figure_title, figure_name, ):
    xticks = np.arange(0, len(list_proj1[0]) + 10.0, 10.0)
    xlabels = xticks - len(list_proj1[0]) / 2
    list_colors = ["black", "blue", "green", "red", "orange", "purple", "grey"]

    plt.figure()
    plt.title(figure_title)
    # axs[1].set(xlim=(0 , image.shape[1]), ylim=(0, max(proj)))
    plt.xlabel("z-position [mm]")
    plt.xticks(xticks, xlabels)
    for i in range(len(list_proj1)):
        plt.plot(list_proj1[i], label=list_labels[i], color=list_colors[i])
        plt.plot(list_proj2[i], color=list_colors[i], linestyle="--")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_backprojection(image, figure_title, figure_name):
    # rotate original image by 90 degrees
    image = np.rot90(image)
    proj = np.sum(image, axis=0)

    xticks = np.arange(0, image.shape[1] + 10.0, 10.0)
    xlabels = xticks - image.shape[1] / 2
    yticks = np.arange(0, image.shape[0] + 5.0, 5.0)
    ylabels = yticks - image.shape[0] / 2

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    axs[0].set_title(figure_title)
    axs[0].imshow(image)
    axs[0].set_aspect('auto')
    axs[0].set_yticks(yticks, ylabels)
    # axs[0].set(xlim=(0, image.shape[1]), ylim=(0, image.shape[0]))
    plt.tick_params('x', labelbottom=False)

    axs[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axs[1].set_ylim(max(proj) * 0.5, max(proj) * 1.5)
    axs[1].set_aspect('auto')
    # axs[1].set(xlim=(0 , image.shape[1]), ylim=(0, max(proj)))
    axs[1].set_xlabel("z-position [mm]")
    axs[1].set_xticks(xticks, xlabels)
    axs[1].plot(proj, color="black")

    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_backprojection_image(image, figure_title, figure_name):
    plt.rcParams.update({'font.size': 14})
    # rotate original image by 90 degrees
    image = np.rot90(image)
    proj = np.sum(image, axis=0)

    xticks = np.arange(0, image.shape[1] + 10.0 + 0.1, 10.0) - 0.5
    xlabels = xticks - image.shape[1] / 2 + 0.5
    yticks = np.arange(0, image.shape[0] + 5.0 + 1.0, 5.0) - 0.5
    ylabels = yticks - image.shape[0] / 2 + 0.5

    plt.figure(figsize=(8, 4))
    plt.title(figure_title)
    plt.xticks(xticks, xlabels)
    plt.yticks(yticks, ylabels)
    plt.xlabel("z-position [mm]")
    plt.ylabel("y-position [mm]")
    plt.imshow(image)
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


"""
def plot_backprojection_dual(image_0mm, image_5mm, figure_title, figure_name):
    plt.rcParams.update({'font.size': 14})
    # rotate original image by 90 degrees
    image_0mm = np.rot90(image_0mm)
    proj_0mm = np.sum(image_0mm, axis=0)

    image_5mm = np.rot90(image_5mm)
    proj_5mm = np.sum(image_5mm, axis=0)

    xticks = np.arange(0, image_0mm.shape[1] + 10.0 + 0.1, 10.0) - 0.5
    xlabels = xticks - image_0mm.shape[1] / 2 + 0.5
    yticks = np.arange(0, image_0mm.shape[0] + 0.1, 10.0) - 0.5
    ylabels = yticks - image_0mm.shape[0] / 2 + 0.5

    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0)
    axs = gs.subplots(sharex=True)

    axs[0].set_title(figure_title)
    axs[0].imshow(image_0mm)
    axs[0].set_aspect('auto')
    axs[0].set_yticks(yticks, ylabels)
    # axs[0].set(xlim=(0, image.shape[1]), ylim=(0, image.shape[0]))
    plt.tick_params('x', labelbottom=False)

    axs[1].xaxis.set_tick_params(which='both', labelbottom=True)
    axs[1].set_aspect('auto')
    # axs[1].set(xlim=(0 , image.shape[1]), ylim=(0, max(proj)))
    axs[1].set_xlabel("z-position [mm]")
    axs[1].set_xticks(xticks, xlabels)
    axs[1].plot(proj_0mm, color="black", label="0mm")
    axs[1].plot(proj_5mm, color="red", label="5mm")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(figure_name + ".png")
"""


def plot_backprojection_stacked(list_image, list_labels, figure_title, figure_name):
    list_proj = []
    for i in range(len(list_image)):
        list_image[i] = np.rot90(list_image[i])
        proj = np.sum(list_image[i], axis=0)
        list_proj.append(proj)

    xticks = np.arange(0, list_image[0].shape[1] + 10.0, 10.0)
    xlabels = xticks - list_image[0].shape[1] / 2
    list_colors = ["black", "blue", "green", "red", "orange", "purple", "grey"]

    plt.figure()
    plt.title(figure_title)
    plt.xlabel("z-position [mm]")
    plt.xticks(xticks, xlabels)
    for i in range(len(list_image)):
        plt.plot(list_proj[i], label=list_labels[i], color=list_colors[i])
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_backprojection_stacked_dual(list_image_0mm, list_image_5mm, list_labels, figure_title, figure_name):
    list_proj_0mm = []
    list_proj_5mm = []
    for i in range(len(list_image_0mm)):
        list_image_0mm[i] = np.rot90(list_image_0mm[i])
        proj_0mm = np.sum(list_image_0mm[i], axis=0)
        list_proj_0mm.append(proj_0mm)

        list_image_5mm[i] = np.rot90(list_image_5mm[i])
        proj_5mm = np.sum(list_image_5mm[i], axis=0)
        list_proj_5mm.append(proj_5mm)

    xticks = np.arange(0, list_image_0mm[0].shape[1] + 10.0, 10.0)
    xlabels = xticks - list_image_0mm[0].shape[1] / 2
    list_colors = ["black", "blue", "green", "red", "orange", "purple", "grey"]

    plt.figure()
    plt.title(figure_title)
    # axs[1].set(xlim=(0 , image.shape[1]), ylim=(0, max(proj)))
    plt.xlabel("z-position [mm]")
    plt.xticks(xticks, xlabels)
    for i in range(len(list_image_0mm)):
        plt.plot(list_proj_0mm[i], label=list_labels[i], color=list_colors[i])
        plt.plot(list_proj_5mm[i], color=list_colors[i], linestyle="--")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(figure_name + ".png")
