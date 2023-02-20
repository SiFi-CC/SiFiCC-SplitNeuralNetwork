# ------------------------------------------------------------------------------
# General script which aim to imitate the classical cut-based approach
# event selection for SiFi-CC event identification developed in Jonas PhD thesis.

import numpy as np
import os
from uproot_methods import TVector3

from src import RootParser
from src import root_files


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


def check_valid_prediction(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz):
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
                        event_energy * (ELECTRON_MASS + event_energy) / (ELECTRON_MASS / 2 + event_energy) / (
                                ELECTRON_MASS / 2 + event_energy) * event_energy_uncertainty)

        # print(electron_energy_value - electron_energy_uncertainty, compton_edge[0] + compton_edge[1])
        if e + ee > compton_edge[0] + compton_edge[1]:
            return False
    return True


def beam_origin(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz, beam_diff, inverse):
    #
    #
    #

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
    theta = tmath_acos(1 - ELECTRON_MASS * (1 / (photon_energy) - 1 / (photon_energy + electron_energy)))

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

    if crosspoint.y > 0 and m0.y < 0:
        return True
    elif crosspoint.y < 0 and m0.y > 0:
        return True

    distance = m0.y
    if abs(distance) < beam_diff:
        return True

    return False


########################################################################################################################

def select_event(event):
    # Settings
    is_first = False
    is_sum = True
    beam_diff = 20
    is_compton = True
    is_inversion = True

    # Event Class IDs
    # 0 - S1A1-EP (electron scatterer, photon absorber)
    # 1 - S1A1-PE (photon-scatterer, electron absorber)
    #   2 clusters one in scatterer one in absorber
    #
    # 2 - S1AX
    #   1 cluster in scatterer and x in absorber
    # procedure for x:
    #          biggest deposition is photon position and all energy is photon energy

    identified = 0
    tag_class = ""
    tag_reason = ""

    # sort cluster by index
    idx_scatterer, idx_absorber = event.sort_clusters_by_module(use_energy=True)

    # Possible classes are
    # S1A1
    # S1AX
    # if cluster in scatter > 1 not identified

    if not len(idx_scatterer) == 1:
        identified = 0
        tag_reason = "TOMUCHSCATCLUSTER"
        tag_class = "BACKGROUND"

    else:
        # S1A1 class no handle on backscattering)
        if len(idx_absorber) == 1:
            identified = 1
            tag_class = "S1A1"
            tag_reason = "IDENTIFIED"
        else:
            # handle S1AX class
            identified = 1
            tag_class = "S1AX"
            tag_reason = "IDENTIFIED"

        # handle on event kinematics
        if identified != 0:
            if check_compton_kinematics(event, is_compton):
                if not beam_origin(event, beam_diff):
                    identified = 0
                    tag_reason = "DAACWRONG"
            elif is_inversion:
                # check if inversion of IddClass==0 leads to good IDClass==1)

                if not check_compton_kinematics(event, is_compton):
                    identified = 0
                    tag_reason = "COMPTONWRONG"
                else:
                    tag_class = "S1A1-PE"
                    tag_reason = "IDENTIFIED"
                    if not beam_origin(event, beam_diff, inverse=True):
                        identified = 0
                        tag_reason = "DAACWRONG"
                    else:
                        tag_reason = "COMTPONWRONG"
            else:
                tag_reason = "COMPTONWRONG"

    return identified


def main():
    # TODO: define wrapper for event object
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"

    root_data = RootParser(dir_main + root_files.OptimisedGeometry_BP0mm_2e10protons_offline)

    n = 100000
    positives = 0.0

    for i, event in enumerate(root_data.iterate_events(n=n)):
        cb_true_tag = event.Identified

        cb_identified = select_event(event)

        if cb_identified != 0.0 and cb_true_tag != 0.0:
            positives += 1.0
        if cb_identified == 0.0 and cb_true_tag == 0.0:
            positives += 1.0

    print("Positive Rate: {:.3f}%".format(positives / n * 100))
    print(n, positives)
