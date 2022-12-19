def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import numpy as np
    import os
    import copy
    from fastROCAUC import fastROCAUC
    import uproot

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.features)
    y_true = DataCluster.targets

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC(y_pred, y_true, return_score=True)

    # best optimal threshold
    threshold = theta

    # evaluate important metrics
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_pred[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            FN += 1

    if (TP + FN) == 0:
        efficiency = 0
    else:
        efficiency = TP / (TP + FN)
    if (TP + FP) == 0:
        purity = 0
    else:
        purity = TP / (TP + FP)

    print("\nFull dataset Classification results: ")
    print("AUC score: {:.3f}".format(auc))
    print("Threshold: {:.3f}".format(theta))
    print("Accuracy: {:.1f}".format((TP + TN) / (TP + TN + FP + FN) * 100))
    print("Efficiency: {:.1f}%".format(efficiency * 100))
    print("Purity: {:.1f}%".format(purity * 100))
    print("TP: {} | TN: {} | FP: {} | FN: {}".format(TP, TN, FP, FN))

    ####################################################################################################################
    # Cut-Based approach based energy and position regression
    for i in range(DataCluster.features.shape[1]):
        DataCluster.features[:, i] *= DataCluster.list_std[i]
        DataCluster.features[:, i] += DataCluster.list_mean[i]

    # select all neural network identified events
    identified = [i for i in range(DataCluster.features.shape[0]) if y_pred[i] == 1.0]
    DataCluster.features = DataCluster.features[identified, :]

    # TODO: remove hard-coding of energy and position indexing
    # grab sum of absorber energy
    # identify events with invalid compton cones
    p = np.sum(DataCluster.features[:, [16, 21, 26, 31, 36]], axis=1)
    e = DataCluster.features[:, 1]

    print(p.shape, p)
    print(e.shape, e)

    me = 0.510999
    arc_base = np.abs(1 - me * (1 / p - 1 / (e + p)))
    valid_arc = arc_base <= 1

    # filter out invalid events
    DataCluster.features = DataCluster.features[valid_arc, :]

    # required fields for the root file
    zeros = np.zeros(DataCluster.features.shape[0])
    event_number = zeros
    event_type = zeros

    e_energy = DataCluster.features[:, 1]
    p_energy = np.sum(DataCluster.features[:, [16, 21, 26, 31, 36]], axis=1)
    total_energy = e_energy + p_energy

    e_pos_x = DataCluster.features[:, 3]
    e_pos_y = -DataCluster.features[:, 4]
    e_pos_z = -DataCluster.features[:, 2]

    p_pos_x = DataCluster.features[:, 18]
    p_pos_y = -DataCluster.features[:, 19]
    p_pos_z = -DataCluster.features[:, 17]

    arc = np.arccos(1 - me * (1 / p_energy - 1 / total_energy))

    # create root file
    # TODO: add root file name tag to final file name
    file_name = SiFiCCNN.model_name + SiFiCCNN.model_tag + "MLEMInput.root"
    file = uproot.recreate(file_name, compression=None)

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
        'StopEvent': [DataCluster.features.shape[0]],
        'TotalSimNev': [0 - DataCluster.features.shape[0]]
    })

    # closing the root file
    file.close()

