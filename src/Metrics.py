import numpy as np


def is_event_correct(y_class_pred,
                     y_energy_pred,
                     y_position_predict,
                     y_class_true,
                     y_energy_true,
                     y_position_true,
                     f_energy=2 * 0.06,
                     f_position_x=1.3 * 2,
                     f_position_y=10.0 * 2,
                     f_position_z=1.3 * 2):
    """
    Return boolean if event is correct in terms of class and regression prediction.
    This is valid only for signal events.
    """

    if not (y_class_pred == 1 and y_class_true == 1):
        return False
    if np.abs(float(y_energy_pred[0]) - float(y_energy_true[0])) > f_energy * float(y_energy_true[0]):
        return False
    if np.abs(float(y_energy_pred[1]) - float(y_energy_true[1])) > f_energy * float(y_energy_true[1]):
        return False
    if np.abs(y_position_predict[0] - y_position_true[0]) > f_position_x:
        return False
    if np.abs(y_position_predict[1] - y_position_true[1]) > f_position_y:
        return False
    if np.abs(y_position_predict[2] - y_position_true[2]) > f_position_z:
        return False
    if np.abs(y_position_predict[3] - y_position_true[3]) > f_position_x:
        return False
    if np.abs(y_position_predict[4] - y_position_true[4]) > f_position_y:
        return False
    if np.abs(y_position_predict[5] - y_position_true[5]) > f_position_z:
        return False

    return True


def get_global_effpur(n_positive_total,
                      ary_class_pred,
                      ary_energy_pred,
                      ary_position_pred,
                      ary_class_true,
                      ary_energy_true,
                      ary_position_true,
                      f_energy=2 * 0.06,
                      f_position_x=1.3 * 2,
                      f_position_y=10.0 * 2,
                      f_position_z=1.3 * 2
                      ):
    n_correct = 0
    for i in range(len(ary_class_pred)):
        if is_event_correct(ary_class_pred[i],
                            ary_energy_pred[i],
                            ary_position_pred[i],
                            ary_class_true[i],
                            ary_energy_true[i],
                            ary_position_true[i],
                            f_energy,
                            f_position_x,
                            f_position_y,
                            f_position_z):
            n_correct += 1

    efficiency = n_correct / n_positive_total
    purity = n_correct / np.sum(ary_class_true)

    return efficiency, purity



def get_classifier_metrics(y_scores, y_true, threshold, weighted=False):
    # pre-define
    y_pred = np.zeros(shape=(len(y_true, )))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_scores[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1

    if (tp + fn) == 0:
        efficiency = 0
    else:
        efficiency = tp / (tp + fn)
    if (tp + fp) == 0:
        purity = 0
    else:
        purity = tp / (tp + fp)

    if weighted:
        # set sample weights to class weights
        _, counts = np.unique(y_pred, return_counts=True)
        class_weights = [len(y_pred) / (2 * counts[0]), len(y_pred) / (2 * counts[1])]

        accuracy = ((tp * class_weights[1]) + (tn * class_weights[0])) / (
                ((tp + fp) * class_weights[1]) + ((tn + fn) * class_weights[0]))
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    return accuracy, efficiency, purity, (tp, fp, tn, fn)
