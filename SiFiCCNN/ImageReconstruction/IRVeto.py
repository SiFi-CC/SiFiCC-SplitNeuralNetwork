import numpy as np
from uproot_methods import TVector3


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


def check_valid_prediction(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz, theta):
    """
    Check if regressed event topology of a compton event makes physical sense.
    -> Energies are positive and non-zero
    -> Position are inside the detector dimensions
    """
    if e == 0.0 or p == 0.0:
        return False
    # TODO: check if position prediction inside detector dimension

    return True


def check_compton_arc(e, p):
    """
    Check if compton scattering angle is in the valid parameter range
    """
    ELECTRON_MASS = 0.511

    arc_base = np.abs(1 - ELECTRON_MASS * (1 / p - 1 / (e + p)))
    if arc_base <= 1:
        return True
    else:
        return False


def check_compton_kinematics(e, p, ee=0.0, ep=0.0, compton=True):
    if compton:
        ELECTRON_MASS = 0.511

        event_energy = e + p
        event_energy_uncertainty = np.sqrt(ee ** 2 + ep ** 2)

        compton_edge = ((event_energy / (1 + ELECTRON_MASS / (2 * event_energy))),
                        event_energy * (ELECTRON_MASS + event_energy) / (
                                ELECTRON_MASS / 2 + event_energy) / (
                                ELECTRON_MASS / 2 + event_energy) * event_energy_uncertainty)

        # print(electron_energy_value - electron_energy_uncertainty, compton_edge[0] + compton_edge[1])
        if e + ee > compton_edge[0] + compton_edge[1]:
            return False
    return True


def check_DAC(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz, beam_diff, inverse, return_dac=False):
    ELECTRON_MASS = 0.511

    # pseudo-wrapper
    # grab Tvectors and transform into new coordinate system
    electron_energy = e
    photon_energy = p
    electron_position = TVector3(p_ez, p_ey, p_ex)
    photon_position = TVector3(p_pz, p_py, p_px)

    coneaxis = electron_position - photon_position
    coneapex = electron_position

    f1 = -coneapex.z / coneaxis.z
    crosspoint = coneapex + (f1 * coneaxis)

    sign = 1
    if crosspoint.y > 0:
        sign = -1

    n_x = sign / np.sqrt((coneaxis.x * coneaxis.x / coneaxis.z / coneaxis.z) + 1)
    n_z = np.sqrt(1 - (n_x * n_x))
    rotvec = TVector3(n_x, 0, n_z)
    theta = tmath_acos(
        1 - ELECTRON_MASS * (1 / (photon_energy) - 1 / (photon_energy + electron_energy)))

    if inverse:
        theta = np.pi - theta

    # generate rotation matrix
    # uproot methods does not support matrices
    # matrix multiplication will be done in numpy
    a1 = [(rotvec.x * rotvec.x * (1 - np.cos(theta))) + np.cos(theta),
          (rotvec.x * rotvec.y * (1 - np.cos(theta))) - (rotvec.z * np.sin(theta)),
          (rotvec.x * rotvec.z * (1 - np.cos(theta))) + (rotvec.y * np.sin(theta))]

    a2 = [(rotvec.y * rotvec.x * (1 - np.cos(theta))) + (rotvec.z * np.sin(theta)),
          (rotvec.y * rotvec.y * (1 - np.cos(theta))) + np.cos(theta),
          (rotvec.y * rotvec.z * (1 - np.cos(theta))) - (rotvec.x * np.sin(theta))]

    a3 = [(rotvec.z * rotvec.x * (1 - np.cos(theta))) - (rotvec.y * np.sin(theta)),
          (rotvec.z * rotvec.y * (1 - np.cos(theta))) + (rotvec.x * np.sin(theta)),
          (rotvec.z * rotvec.z * (1 - np.cos(theta))) + np.cos(theta)]

    rotmat = np.array([a1, a2, a3])
    ary_coneaxis = np.array([coneaxis.x, coneaxis.y, coneaxis.z])
    ary_endvec = rotmat.dot(ary_coneaxis)
    endvec = TVector3(ary_endvec[0], ary_endvec[1], ary_endvec[2])

    f2 = -coneapex.z / endvec.z
    m0 = (coneapex + (f2 * endvec))

    if return_dac:
        if crosspoint.y > 0 and m0.y < 0:
            return 0.0
        elif crosspoint.y < 0 and m0.y > 0:
            return 0.0

        distance = m0.y
        return abs(distance)

    if crosspoint.y > 0 and m0.y < 0:
        return True
    elif crosspoint.y < 0 and m0.y > 0:
        return True

    distance = m0.y
    if abs(distance) < beam_diff:
        return True

    return False
