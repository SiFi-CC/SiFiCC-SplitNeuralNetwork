import numpy as np


def compton_scattering_angle(e1, e2):
    """Calculates the compton scattering angle from given energies.

    Energies are given as primary gamma energy and gamma energy after
    scattering. Compton scattering angle is derived from the formula:

        cos(\theta) = (1.0 - kMe) * (\frac{1}{E_2} - \frac{1}{E_1}

    Args:
         e1: float; initial gamma energy
         e2: float; gamma energy after scattering

    Return:
        theta: float; compton scattering angle in rad
    """

    if e1 == 0.0 or e2 == 0.0:
        return 0.0

    kMe = 0.510999  # MeV/c^2
    costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / e1)

    if abs(costheta) > 1:
        return 0.0
    else:
        theta = np.arccos(costheta)  # rad
        return theta


def vector_angle(vec1, vec2):
    """Calculates the angle between two given vectors.

    Vectors are given as root TVector3 or numpy arrays. Vectors are normalized
    and their angle is calculated from the vector dot product.

    Args:
         vec1: TVector3 or ndarray (3,); 3-dim origin vector
         vec2: TVector3 or ndarray (3,); 3-dim origin vector

    Returns:
        theta: float; Angle between vectors in rad
    """
    # exception input vectors are ndarrays
    if type(vec1) is np.ndarray and type(vec2) is np.ndarray:

        vec1 /= np.sqrt(np.dot(vec1, vec1))
        vec2 /= np.sqrt(np.dot(vec2, vec2))

        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    # exception input vectors are TVector2 objects
    else:
        if vec1.mag == 0 or vec2.mag == 0:
            return 0.0

        ary_vec1 = np.array([vec1.x, vec1.y, vec1.z])
        ary_vec2 = np.array([vec2.x, vec2.y, vec2.z])

        ary_vec1 /= np.sqrt(np.dot(ary_vec1, ary_vec1))
        ary_vec2 /= np.sqrt(np.dot(ary_vec2, ary_vec2))

        return np.arccos(np.clip(np.dot(ary_vec1, ary_vec2), -1.0, 1.0))
