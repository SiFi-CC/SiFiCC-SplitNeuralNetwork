import numpy as np
import uproot
from uproot_methods import TVector3


def tmath_acos(x):
    if (x < -1):
        return np.pi
    if (x > 1):
        return 0
    return np.arccos(x)


def check_valid_prediction(energy_e, energy_p, pos_pz):
    if pos_pz == 0.0:
        return False
    if energy_e == 0.0 or energy_p == 0.0:
        return False

    return True


def check_compton_arc(e_e, e_p):
    ELECTRON_MASS = 0.511

    arc_base = np.abs(1 - ELECTRON_MASS * (1 / e_p - 1 / (e_e + e_p)))
    if arc_base <= 1:
        return True
    else:
        return False


def check_compton_kinematics(energy_e, energy_p, energy_e_uncertainty=0.0, energy_p_uncertainty=0.0, is_compton=True):
    if is_compton:
        # TODO: check correctness
        ELECTRON_MASS = 0.511

        event_energy = energy_e + energy_p
        event_energy_uncertainty = np.sqrt(energy_e_uncertainty ** 2 + energy_p_uncertainty ** 2)

        compton_edge = ((event_energy / (1 + ELECTRON_MASS / (2 * event_energy))),
                        event_energy * (ELECTRON_MASS + event_energy) / (ELECTRON_MASS / 2 + event_energy) / (
                                ELECTRON_MASS / 2 + event_energy) * event_energy_uncertainty)

        # print(electron_energy_value - electron_energy_uncertainty, compton_edge[0] + compton_edge[1])
        if energy_e - energy_e_uncertainty > compton_edge[0] + compton_edge[1]:
            return False
    return True


def beam_origin(e_e, e_p, pos_ex, pos_ey, pos_ez, pos_px, pos_py, pos_pz, beam_diff, is_inverse=False):
    ELECTRON_MASS = 0.511

    # pseudo-wrapper
    # grab Tvectors and transform into new coordinate system
    electron_energy = e_e
    photon_energy = e_p
    electron_position = TVector3(pos_ez, pos_ey, pos_ex)
    photon_position = TVector3(pos_pz, pos_py, pos_px)

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

    if is_inverse:
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


def export_mlem(ary_e, ary_p, ary_ex, ary_ey, ary_ez, ary_px, ary_py, ary_pz, filename="MLEM_export"):
    # Global settings
    beam_diff = 20  # in [mm]
    is_compton = True
    is_cut_based = False

    # By default all events are accepted
    ary_identified = np.ones(shape=(len(ary_e),))
    error_valid_prediction = 0
    err0r_compton_kinematics = 0
    error_beam_origin = 0

    print(np.sum(ary_identified), " events pre-selected")

    for i in range(len(ary_e)):
        # define event quantities:
        identified = 1

        e_e = ary_e[i]
        e_p = ary_p[i]
        pos_ex = ary_ex[i]
        pos_ey = ary_ey[i]
        pos_ez = ary_ez[i]
        pos_px = ary_px[i]
        pos_py = ary_py[i]
        pos_pz = ary_pz[i]

        if is_cut_based:
            # check validity of predictions
            if not check_valid_prediction(e_e, e_p, pos_pz):
                # print("failed valid prediction")
                ary_identified[i] = 0
                error_valid_prediction += 1
                continue
            # check compton kinematics
            if not check_compton_kinematics(e_e, e_p, is_compton=is_compton):
                # print("failed compton kinematics")
                ary_identified[i] = 0
                err0r_compton_kinematics += 1
                continue
            # check DAC-filter
            if not beam_origin(e_e, e_p, pos_ex, pos_ey, pos_ez, pos_px, pos_py, pos_pz, beam_diff):
                # print("failed DAAC")
                ary_identified[i] = 0
                error_beam_origin += 1
                continue

        ary_identified[i] = identified

    print(np.sum(ary_identified), " events after cuts")

    # required fields for the root file
    entries = np.sum(ary_identified)
    print(entries)
    zeros = np.zeros(shape=(int(entries),))
    event_number = zeros
    event_type = zeros

    ary_identified = ary_identified == 1
    e_energy = ary_e[ary_identified]
    p_energy = ary_p[ary_identified]
    total_energy = e_energy + p_energy

    e_pos_x = ary_ey[ary_identified]
    e_pos_y = -ary_ez[ary_identified]
    e_pos_z = -ary_ex[ary_identified]
    p_pos_x = ary_py[ary_identified]
    p_pos_y = -ary_pz[ary_identified]
    p_pos_z = -ary_px[ary_identified]

    arc = np.arccos(1 - 0.511 * (1 / p_energy - 1 / total_energy))

    # create root file
    file_name = filename + ".root"
    file = uproot.recreate(file_name, compression=None)

    print(len(arc), "events exported")

    # defining the branch
    branch = {
        'GlobalEventNumber': 'int32',  # event sequence in the original simulation file
        'v_x': 'float32',  # electron position
        'v_y': 'float32',
        'v_z': 'float32',
        'v_unc_x': 'float32',
        'v_unc_y': 'float32',
        'v_unc_z': 'float32',
        'p_x': 'float32',  # vector pointing from e pos to p pos
        'p_y': 'float32',
        'p_z': 'float32',
        'p_unc_x': 'float32',
        'p_unc_y': 'float32',
        'p_unc_z': 'float32',
        'E0Calc': 'float32',  # total energy
        'E0Calc_unc': 'float32',
        'arc': 'float32',  # formula
        'arc_unc': 'float32',
        'E1': 'float32',  # e energy
        'E1_unc': 'float32',
        'E2': 'float32',  # p energy
        'E2_unc': 'float32',
        'E3': 'float32',  # 0
        'E3_unc': 'float32',
        'ClassID': 'int32',  # 0
        'EventType': 'int32',  # 2-correct  1-pos  0-wrong
        'EnergyBinID': 'int32',  # 0
        'x_1': 'float32',  # electron position
        'y_1': 'float32',
        'z_1': 'float32',
        'x_2': 'float32',  # photon position
        'y_2': 'float32',
        'z_2': 'float32',
        'x_3': 'float32',  # 0
        'y_3': 'float32',
        'z_3': 'float32',
    }

    file['ConeList'] = uproot.newtree(branch, title='Neural network cone list')

    # filling the branch
    file['ConeList'].extend({
        'GlobalEventNumber': event_number,
        'v_x': e_pos_x,
        'v_y': e_pos_y,
        'v_z': e_pos_z,
        'v_unc_x': zeros,
        'v_unc_y': zeros,
        'v_unc_z': zeros,
        'p_x': p_pos_x - e_pos_x,
        'p_y': p_pos_y - e_pos_y,
        'p_z': p_pos_z - e_pos_z,
        'p_unc_x': zeros,
        'p_unc_y': zeros,
        'p_unc_z': zeros,
        'E0Calc': total_energy,
        'E0Calc_unc': zeros,
        'arc': arc,
        'arc_unc': zeros,
        'E1': e_energy,
        'E1_unc': zeros,
        'E2': p_energy,
        'E2_unc': zeros,
        'E3': zeros,
        'E3_unc': zeros,
        'ClassID': zeros,
        'EventType': event_type,
        'EnergyBinID': zeros,
        'x_1': e_pos_x,
        'y_1': e_pos_y,
        'z_1': e_pos_z,
        'x_2': p_pos_x,
        'y_2': p_pos_y,
        'z_2': p_pos_z,
        'x_3': zeros,
        'y_3': zeros,
        'z_3': zeros,
    })

    # defining the settings branch
    branch2 = {
        'StartEvent': 'int32',
        'StopEvent': 'int32',
        'TotalSimNev': 'int32'
    }

    file['TreeStat'] = uproot.newtree(branch2, title='Evaluated events details')

    # filling the branch
    file['TreeStat'].extend({
        'StartEvent': [0],
        'StopEvent': [entries],
        'TotalSimNev': [0 - entries]
    })

    # closing the root file
    file.close()
