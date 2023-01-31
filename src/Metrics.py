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
