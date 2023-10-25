import numpy as np
from scipy.spatial.transform import Rotation as R


def get_edges(x, y, z, xdim, ydim, zdim):
    """
    Calculates a list of all edges of a cuboid. Used to feed matplotlib methods.

    Args:
        x (float): x-coordinate of center
        y (float): y-coordinate of center
        z (float): z-coordinate of center
        xdim (float): length of cuboid in x-dimension
        ydim (float): length of cuboid in y-dimension
        zdim (float): length of cuboid in z-dimension

    Return:
        list of edges (list of 3-length lists)
    """
    list_edges = [
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z - zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z + zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x - xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y - ydim / 2, y - ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [[x + xdim / 2, x + xdim / 2], [y + ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]

    return list_edges


def get_surface(x, y, z, xdim, ydim, zdim):
    """
    Calculates a list of all surfaces of a cuboid. Used to feed matplotlib methods.

    Args:
        x (float): x-coordinate of center
        y (float): y-coordinate of center
        z (float): z-coordinate of center
        xdim (float): length of cuboid in x-dimension
        ydim (float): length of cuboid in y-dimension
        zdim (float): length of cuboid in z-dimension

    Return:
        list of edges (list of 3-length lists)
    """

    one = np.ones(4).reshape(2, 2)
    list_surface = [
        [[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z - zdim / 2) * one],
        [[x - xdim / 2, x + xdim / 2], [y - ydim / 2, y + ydim / 2], (z + zdim / 2) * one],
        [[x - xdim / 2, x + xdim / 2], (y - ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
        [[x - xdim / 2, x + xdim / 2], (y + ydim / 2) * one, [z - zdim / 2, z + zdim / 2]],
        [(x - xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]],
        [(x + xdim / 2) * one, [y - ydim / 2, y + ydim / 2], [z - zdim / 2, z + zdim / 2]]]
    return list_surface


def unit_vec(vec):
    """
    Returns the unit vector of a given vector.

    Args:
        vec (TVector3): vector

    Returns:
        unit vector (TVector3)

    """
    return vec / np.sqrt(np.dot(vec, vec))


def vec_angle(vec1, vec2):
    """
    Calculates the vector between two given angles.

    Args:
        vec1 (TVector3): vector 1
        vec2 (TVector3): vector 2

    Return:
         angle between vectors
    """
    return np.arccos(np.clip(np.dot(unit_vec(vec1), unit_vec(vec2)), -1.0, 1.0))


def get_compton_cone_cracow(vec_apex, vec_axis, vec_origin, theta, sr=8):
    """
    Computes the compton cone of a reconstructed event.

    Args:
        vec_apex    (TVector3): origin vector of true scatterer interaction
        vec_axis    (TVector3): vector pointing from true absorber interaction to true scatterer
                                interaction
        vec_origin  (TVector3): origin vector of true source position
        theta:
        sr:

    Returns:

    """
    # TODO: All vector rotations are done via scipy transformation library
    # TODO: Uproot vector rotations should be better suited
    # Correct angle theta (stems from the definition of the axis vector as it is flipped)
    theta = np.pi - theta

    # Rotate reference vector around scattering angle theta
    ref_vec = np.array([1, 0, 0])
    rotation_y = R.from_rotvec((vec_axis.theta - np.pi / 2 - theta) * np.array([0, 1, 0]))
    rotation_z = R.from_rotvec(vec_axis.phi * np.array([0, 0, 1]))
    ref_vec = rotation_y.apply(ref_vec)
    ref_vec = rotation_z.apply(ref_vec)

    # Rotate reference vector around axis vector to sample cone edges
    list_cone_vec = []
    rot_axis_ary = np.array([vec_axis.x, vec_axis.y, vec_axis.z])
    # Phi angle sampling (not the same as scattering angle theta!)
    list_phi = np.linspace(0, 360, sr)
    for angle in list_phi:
        vec_temp = ref_vec
        rot_vec = np.radians(angle) * rot_axis_ary / np.sqrt(np.dot(rot_axis_ary, rot_axis_ary))
        rot_M = R.from_rotvec(rot_vec)
        vec_temp = rot_M.apply(vec_temp)
        list_cone_vec.append(vec_temp)

    # scale each cone vector to hit the final canvas
    # shift them to correct final position

    for i in range(len(list_cone_vec)):
        a = -(vec_apex.x - vec_origin.x) / list_cone_vec[i][0]
        list_cone_vec[i] *= a
        list_cone_vec[i] = np.array([list_cone_vec[i][0] + vec_apex.x,
                                     list_cone_vec[i][1] + vec_apex.y,
                                     list_cone_vec[i][2] + vec_apex.z])

    return list_cone_vec


def get_compton_cone_aachen(vec_apex, vec_axis, vec_origin, theta, sr=8):
    """
    Computes the compton cone of a reconstructed event.

    Args:
        vec_apex    (TVector3): origin vector of true scatterer interaction
        vec_axis    (TVector3): vector pointing from true absorber interaction to true scatterer
                                interaction
        vec_origin  (TVector3): origin vector of true source position
        theta:
        sr:

    Returns:

    """
    # TODO: All vector rotations are done via scipy transformation library
    # TODO: Uproot vector rotations should be better suited
    # Correct angle theta (stems from the definition of the axis vector as it is flipped)
    theta = np.pi - theta

    # Rotate reference vector around scattering angle theta
    ref_vec = np.array([0, 0, 1])
    rotation_y = R.from_rotvec((vec_axis.theta - np.pi / 2 - theta) * np.array([0, 1, 0]))
    rotation_x = R.from_rotvec(vec_axis.phi * np.array([1, 0, 0]))
    ref_vec = rotation_y.apply(ref_vec)
    ref_vec = rotation_x.apply(ref_vec)

    # Rotate reference vector around axis vector to sample cone edges
    list_cone_vec = []
    rot_axis_ary = np.array([vec_axis.x, vec_axis.y, vec_axis.z])
    # Phi angle sampling (not the same as scattering angle theta!)
    list_phi = np.linspace(0, 360, sr)
    for angle in list_phi:
        vec_temp = ref_vec
        rot_vec = np.radians(angle) * rot_axis_ary / np.sqrt(np.dot(rot_axis_ary, rot_axis_ary))
        rot_M = R.from_rotvec(rot_vec)
        vec_temp = rot_M.apply(vec_temp)
        list_cone_vec.append(vec_temp)

    # scale each cone vector to hit the final canvas
    # shift them to correct final position
    for i in range(len(list_cone_vec)):
        a = -(vec_apex.z - vec_origin.z) / list_cone_vec[i][0]
        list_cone_vec[i] *= a
        list_cone_vec[i] = np.array([list_cone_vec[i][0] + vec_apex.x,
                                     list_cone_vec[i][1] + vec_apex.y,
                                     list_cone_vec[i][2] + vec_apex.z])

    return list_cone_vec
