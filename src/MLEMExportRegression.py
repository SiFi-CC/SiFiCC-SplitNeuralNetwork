import numpy as np
import uproot


def export_mlem(ary_e,
                ary_p,
                ary_ex,
                ary_ey,
                ary_ez,
                ary_px,
                ary_py,
                ary_pz,
                filename="MLEM_export"):

    for i in range(len(ary_e)):
        if ary_e[i] == 0.0 or ary_p[i] == 0.0:
            print(ary_e[i], ary_p[i], ary_ex[i], ary_ey[i], ary_ez[i], ary_px[i], ary_py[i], ary_pz[i])

    # identify events with invalid compton cones
    me = 0.510999
    ary_arc_base = np.abs(1 - me * (1 / ary_p - 1 / (ary_e + ary_p)))
    ary_arc_base_valid = ary_arc_base <= 1

    # required fields for the root file
    entries = np.sum(ary_arc_base_valid * 1)
    zeros = np.zeros(shape=(entries,))
    event_number = zeros
    event_type = zeros

    e_energy = ary_e[ary_arc_base_valid]
    p_energy = ary_p[ary_arc_base_valid]
    total_energy = e_energy + p_energy

    e_pos_x = ary_ey[ary_arc_base_valid]
    e_pos_y = -ary_ez[ary_arc_base_valid]
    e_pos_z = -ary_ex[ary_arc_base_valid]

    p_pos_x = ary_py[ary_arc_base_valid]
    p_pos_y = -ary_pz[ary_arc_base_valid]
    p_pos_z = -ary_px[ary_arc_base_valid]

    arc = np.arccos(1 - me * (1 / p_energy - 1 / total_energy))

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
