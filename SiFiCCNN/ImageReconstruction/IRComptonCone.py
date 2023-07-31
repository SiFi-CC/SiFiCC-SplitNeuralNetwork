import numpy as np


class ComptonCone:
    def __init__(self, e1, e2, x1, y1, z1, x2, y2, z2):
        # parameter
        self.vec1 = np.array([x1, y1, z1])
        self.vec2 = np.array([x2, y2, z2])
        self.e1 = e1
        self.e2 = e2

        self.theta_dotvec = self.get_theta_dotvec(x1, y1, z1, x2, y2, z2)
        self.theta_energy = ComptonCone.get_theta_energy(e1, e2)

        # constructing cone positions
        self.axis = ComptonCone.connect_points(x1, y1, z1, x2, y2, z2)
        self.apex = np.array([x1, y1, z1])

    @staticmethod
    def connect_points(x1, y1, z1, x2, y2, z2):
        """
        Coordinates of vector connecting two given points.
        Used to find compton cones axis. Point 1 represents the interaction in the scatterer and
        point 2 is the interaction in the absorber.

        Args:
             x1 (Float): x-coordinate of first vector
             y1 (Float): y-coordinate of first vector
             z1 (Float): z-coordinate of first vector
             x2 (Float): x-coordinate of second vector
             y2 (Float): y-coordinate of second vector
             z2 (Float): z-coordinate of second vector
        """

        return np.array([x2 - x1, y2 - y1, z2 - z1])

    @staticmethod
    def tmath_acos(x):
        """
        Alternative version to np.arccos, as it is unclear how it handles
        parameter outside the defined arccos range. This definition is equivalent
        to the root tmath version
        """
        if x < -1:
            return np.pi
        if x > 1:
            return 0
        return np.arccos(x)

    @staticmethod
    def get_theta_energy(e1, e2):
        """
        Calculate scattering angle theta in radiant.

        Args:
             e1 (double): Initial gamma energy
             e2 (double): Gamma energy after compton scattering
        """
        kMe = 0.510999  # MeV/c^2
        costheta = 1.0 - kMe * (1.0 / e2 - 1.0 / (e1 + e2))

        # define physical exceptions
        if e1 == 0.0 or e2 == 0.0:
            return 0.0

        if abs(costheta) > 1:
            return 0.0

        # theta returned via root tmath::acos argument
        theta = ComptonCone.tmath_acos(costheta)  # rad

        return theta

    @staticmethod
    def get_theta_dotvec(x1, y1, z1, x2, y2, z2):
        """
        Calculate scattering angle theta in radiant from the dot product of 2 vectors.

        Args:
             x1 (Float): x-coordinate of first vector
             y1 (Float): y-coordinate of first vector
             z1 (Float): z-coordinate of first vector
             x2 (Float): x-coordinate of second vector
             y2 (Float): y-coordinate of second vector
             z2 (Float): z-coordinate of second vector
        """
        ary_vec1 = np.array([x1, y1, z1])
        ary_vec2 = np.array([x2, y2, z2])

        mag_vec1 = np.sqrt(np.dot(ary_vec1, ary_vec1))
        mag_vec2 = np.sqrt(np.dot(ary_vec2, ary_vec2))

        # exception: if vector magnitude is zero, return zero as vector is invalid
        if mag_vec1 == 0.0 or mag_vec2 == 0.0:
            return 0.0

        ary_vec1 /= mag_vec1
        ary_vec2 /= mag_vec2

        return np.arccos(np.clip(np.dot(ary_vec1, ary_vec2), -1.0, 1.0))
