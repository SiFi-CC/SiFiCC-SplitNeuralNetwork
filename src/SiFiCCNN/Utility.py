import numpy as np


def get_scattering_angle_energy(e1, e2):
    """
    Calculate scattering angle theta in radiant from Compton scattering formula.

    Args:
         e1 (double): Initial gamma energy
         e2 (double): Gamma energy after compton scattering
    """
    if e1 == 0.0 or e2 == 0.0:
        return 0.0

    kMe = 0.510999  # MeV/c^2
    costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / (e1 + e2))

    if abs(costheta) > 1:
        return 0.0
    else:
        theta = np.arccos(costheta)  # rad
        return theta
