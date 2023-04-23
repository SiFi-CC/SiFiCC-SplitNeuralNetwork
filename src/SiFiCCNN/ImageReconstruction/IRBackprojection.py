import numpy as np
from uproot_methods import TVector3

from src.SiFiCCNN.ImageReconstruction import IRComptonCone
from src.SiFiCCNN.ImageReconstruction import IRVeto


def backprojection(ary_e1,
                   ary_e2,
                   ary_x1,
                   ary_y1,
                   ary_z1,
                   ary_x2,
                   ary_y2,
                   ary_z2,
                   ary_theta,
                   scatz=100.0,
                   scaty=40.0,
                   use_theta="DOTVEC",
                   optimized=False,
                   veto=False):
    """
    Method for back-projection. 2D histogram will be set as final image.
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
        ary_theta   (numpy array): theta angle of compton scattering
        scaty       (Int): y-dimension of final image (1 pixel == 1 mm)
        scatz       (Int): z-dimension of final image (1 pixel == 1 mm)
        use_theta   (string): If "DOTVEC", dotvec angle will be use for theta
                                "ENERGY" if energy vector should be used
        optimized   (Boolean): If true, optimized back-projection will be used, final image might be inaccurate
        veto        (Boolean): If true, Cut-Based filter will be applied to veto out events

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

    # main image reconstruction algorythm
    for i in range(entries):
        if veto:
            if not IRVeto.check_valid_prediction(ary_e1[i],
                                                 ary_e2[i],
                                                 ary_x1[i],
                                                 ary_y1[i],
                                                 ary_z1[i],
                                                 ary_x2[i],
                                                 ary_y2[i],
                                                 ary_z2[i],
                                                 ary_theta[i]):
                continue
            if not IRVeto.check_compton_arc(ary_e1[i], ary_e2[i]):
                continue
            if not IRVeto.check_DAC(ary_e1[i],
                                    ary_e2[i],
                                    ary_x1[i],
                                    ary_y1[i],
                                    ary_z1[i],
                                    ary_x2[i],
                                    ary_y2[i],
                                    ary_z2[i],
                                    beam_diff=20,
                                    inverse=False):
                continue

        # create cone object
        cone = IRComptonCone.ComptonCone(ary_e1[i],
                                         ary_e2[i],
                                         ary_x1[i],
                                         ary_y1[i],
                                         ary_z1[i],
                                         ary_x2[i],
                                         ary_y2[i],
                                         ary_z2[i],
                                         ary_theta[i])

        # grab exceptions from scattering angle
        theta = 0
        if use_theta == "DOTVEC":
            theta = cone.theta_dotvec
        if use_theta == "ENERGY":
            theta = cone.theta_energy
        if theta == 0.0:
            continue

        if optimized:
            # Optimized algorithm
            list_pixel_cache = []
            ary_image_temp = np.zeros(shape=(nbinsz, nbinsy))
            ary_map = np.zeros(shape=(nbinsz, nbinsy))

            # optimized sampling of z-dimension
            # zbin_sampling = np.arange(0, nbinsz, 4, dtype=int)
            zbin_sampling = np.arange(0, nbinsz, 4, dtype=int)
            """            
            for i in range(nbinsz):
                zbin_step = int(nbinsz / 2 + (i + 1) ** 2)
                if zbin_step < nbinsz:
                    zbin_sampling.append(int(nbinsz / 2 + (i + 1) ** 2))
                    zbin_sampling.append(int(nbinsz / 2 - (i + 1) ** 2))
                else:
                    break
            """
            for z in zbin_sampling:
                for y in range(nbinsy):
                    ary_map[z, y] = 1
                    pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy),
                                           -zlimit + widthz / 2 + (z * widthz))
                    axis = TVector3(ary_x1[i], ary_y1[i], ary_z1[i]) - TVector3(ary_x2[i], ary_y2[i], ary_z2[i])
                    linkingVector = pixelCenter - TVector3(ary_x1[i], ary_y1[i], ary_z1[i])
                    angle = axis.angle(linkingVector)
                    resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))
                    if abs(cone.theta - angle) <= resolution:
                        ary_image[z, y] += 1

                        ary_image_temp[z, y] += 1
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
                            axis = TVector3(ary_x1[i], ary_y1[i], ary_z1[i]) - TVector3(ary_x2[i], ary_y2[i], ary_z2[i])
                            linkingVector = pixelCenter - TVector3(ary_x1[i], ary_y1[i], ary_z1[i])
                            angle = axis.angle(linkingVector)

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

        if not optimized:
            # Base algorithm
            for z in range(nbinsz):
                for y in range(nbinsy):
                    # calculation done in uproot TVector3 for optimization
                    pixelCenter = TVector3(0.0, -ylimit + widthy / 2 + (y * widthy),
                                           -zlimit + widthz / 2 + (z * widthz))
                    axis = TVector3(ary_x1[i], ary_y1[i], ary_z1[i]) - TVector3(ary_x2[i], ary_y2[i], ary_z2[i])
                    linkingVector = pixelCenter - TVector3(ary_x1[i], ary_y1[i], ary_z1[i])
                    angle = axis.angle(linkingVector)

                    resolution = np.arctan(0.5 * widthz * np.sqrt(2) / (D / A))

                    if abs(theta - angle) <= resolution:
                        ary_image[z, y] += 1

    return ary_image


# ----------------------------------------------------------------------------------------------------------------------
# methods for easy execution of back projection

def get_projection(image):
    # rotate original image by 90 degrees
    # This is purely done for presentational purpose
    image = np.rot90(image)
    ary_proj = np.sum(image, axis=0)
    return ary_proj


def get_backprojection(ary_score,
                       ary_e1,
                       ary_e2,
                       ary_x1,
                       ary_y1,
                       ary_z1,
                       ary_x2,
                       ary_y2,
                       ary_z2,
                       ary_theta,
                       use_theta="DOTVEC",
                       optimized=False,
                       veto=False,
                       f_sample=1.0,
                       n_subsample=1,
                       scatz=60.0,
                       scaty=40.0,
                       threshold=0.5,
                       verbose=0):
    # Grab the total number of positive events
    # Apply the scaling factor to the positive events to generate a sub-sample
    # repeat for n_subsample  and average the back-projection of every sub-sample
    n_pos = np.sum((ary_score > threshold) * 1)
    n_pos_subsample = int(n_pos * f_sample)
    ary_proj = np.zeros(shape=(n_subsample, int(scatz)))

    for i in range(n_subsample):
        ary_idx = np.arange(0, len(ary_score), 1.0, dtype=int)
        ary_idx_pos = ary_idx[ary_score > threshold]

        rng = np.random.default_rng()
        rng.shuffle(ary_idx_pos)

        # generate back-projection image
        ary_image = backprojection(ary_e1[ary_idx_pos[:n_pos_subsample]],
                                   ary_e2[ary_idx_pos[:n_pos_subsample]],
                                   ary_x1[ary_idx_pos[:n_pos_subsample]],
                                   ary_y1[ary_idx_pos[:n_pos_subsample]],
                                   ary_z1[ary_idx_pos[:n_pos_subsample]],
                                   ary_x2[ary_idx_pos[:n_pos_subsample]],
                                   ary_y2[ary_idx_pos[:n_pos_subsample]],
                                   ary_z2[ary_idx_pos[:n_pos_subsample]],
                                   ary_theta[ary_idx_pos[:n_pos_subsample]],
                                   scatz=scatz,
                                   scaty=scaty,
                                   use_theta=use_theta,
                                   optimized=optimized,
                                   veto=veto)
        ary_proj[i, :] = get_projection(ary_image)
        if verbose == 1:
            print("Back-projection done for {} events of sub-sample {}".format(n_pos_subsample, i))

    # mean value of all sub-samples
    proj_mean = np.mean(ary_proj, axis=0)
    proj_std = np.std(ary_proj, axis=0)

    return proj_mean, proj_std
