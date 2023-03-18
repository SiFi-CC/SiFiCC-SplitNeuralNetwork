import numpy as np


# ----------------------------------------------------------------------------------------------------------------------

def event_match(clas_pred, ee_pred, ep_pred, xe_pred, ye_pred, ze_pred, xp_pred, yp_pred, zp_pred,
                clas_true, ee_true, ep_true, xe_true, ye_true, ze_true, xp_true, yp_true, zp_true,
                f_energy=2 * 0.06,
                f_position_x=1.3 * 2,
                f_position_y=10.0 * 2,
                f_position_z=1.3 * 2):
    """
    Return boolean if event is correct in terms of class and regression prediction.
    This is valid only for signal events.
    """

    if not (clas_pred > 0.5 and clas_true == 1):
        return False
    if np.abs(ee_pred - ee_true) > f_energy * ee_true:
        return False
    if np.abs(ep_pred - ep_true) > f_energy * ep_true:
        return False
    if np.abs(xe_pred - xe_true) > f_position_x:
        return False
    if np.abs(ye_pred - ye_true) > f_position_y:
        return False
    if np.abs(ze_pred - ze_true) > f_position_z:
        return False
    if np.abs(xp_pred - xp_true) > f_position_x:
        return False
    if np.abs(yp_pred - yp_true) > f_position_y:
        return False
    if np.abs(zp_pred - zp_true) > f_position_z:
        return False

    return True


def get_global_effpur(ary_pred,
                      ary_true,
                      f_energy=2 * 0.06,
                      f_position_x=1.3 * 2,
                      f_position_y=10.0 * 2,
                      f_position_z=1.3 * 2):
    n_correct = 0
    for i in range(ary_pred.shape[0]):
        if event_match(*ary_pred[i, :],
                       *ary_true[i, :],
                       f_energy=2 * 0.06,
                       f_position_x=1.3 * 2,
                       f_position_y=10.0 * 2,
                       f_position_z=1.3 * 2):
            n_correct += 1

    efficiency = n_correct / (np.sum(ary_true[:, 0]))
    purity = n_correct / np.sum((ary_pred[:, 0] > 0.5) * 1)

    return efficiency, purity
