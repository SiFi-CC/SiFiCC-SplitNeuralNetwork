import os
import numpy as np
import uproot

from SiFiCCNN.ImageReconstruction import IRVeto


def export_CC6(ary_e,
               ary_p,
               ary_ex,
               ary_ey,
               ary_ez,
               ary_px,
               ary_py,
               ary_pz,
               ary_theta=None,
               filename="MLEM_export",
               use_theta="DOTVEC",
               veto=True,
               verbose=0):
    # TODO: handle theta angle

    # Define verbose statistic on event rejection
    ary_identified = np.ones(shape=(len(ary_e),))
    reject_valid = 0
    reject_arc = 0
    reject_kinematics = 0
    reject_DAC = 0

    if veto:
        for i in range(len(ary_e)):
            # define event quantities:
            identified = 1

            e = ary_e[i]
            p = ary_p[i]
            p_ex = ary_ex[i]
            p_ey = ary_ey[i]
            p_ez = ary_ez[i]
            p_px = ary_px[i]
            p_py = ary_py[i]
            p_pz = ary_pz[i]

            if ary_theta is None:
                theta = None
            else:
                theta = ary_theta[i]

            if not IRVeto.check_valid_prediction(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz, theta):
                ary_identified[i] = 0
                reject_valid += 1
                continue

            if not IRVeto.check_compton_arc(e, p):
                ary_identified[i] = 0
                reject_arc += 1
                continue

            if not IRVeto.check_compton_kinematics(e, p, ee=0, ep=0, compton=True):
                # print("failed compton kinematics")
                ary_identified[i] = 0
                reject_kinematics += 1
                continue

            if not IRVeto.check_DAC(e, p, p_ex, p_ey, p_ez, p_px, p_py, p_pz, 20, inverse=False):
                # print("failed DAAC")
                ary_identified[i] = 0
                reject_DAC += 1
                continue

    # print MLEM export statistics
    if verbose == 1:
        print("\n# MLEM export statistics: ")
        print("Number of total events: ", len(ary_e))
        print("Number of events after cuts: ", np.sum(ary_identified))
        print("Number of cut events: ", len(ary_e) - np.sum(ary_identified))
        print("    - Valid prediction: ", reject_valid)
        print("    - Compton arc: ", reject_arc)
        print("    - Compton kinematics: ", reject_kinematics)
        print("    - Beam Origin: ", reject_DAC)

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
    print("file created at: ", os.getcwd() + file_name)

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
