class Detector:
    """object containing detector module dimensions, used in Rootdata.

    Attributes:
        pos (TVec3): vector pointing towards the middle of the detector module
        dimx (float): detector thickness in x-dimension
        dimy (float): detector thickness in y-dimension
        dimz (float): detector thickness in z-dimension

    """

    def __init__(self, pos, dimx, dimy, dimz):
        self.dimx = dimx
        self.dimy = dimy
        self.dimz = dimz
        # position is defined by the Detector-Module middle
        self.pos = pos

    def is_vec_in_module(self, input_vec):
        """Checks if a vector points inside the module.

        Args:
            input_vec (TVector3) or (list<TVector3>): If a list type of vectors is given, True will be returned
                                                      if at least one of the vectors is inside the detector

        Return:
            True if vector points inside module, False otherwise

        """
        # check type of parameter vec
        # TODO: do proper type check

        # some events are right on the border of the detector
        # the factor "a" adds a small buffer to compensate for float uncertainties
        a = 0.001

        try:
            for vec in input_vec:
                # check if vector is inside detector boundaries
                if (abs(self.pos.x - vec.x) <= self.dimx / 2 + a
                        and abs(self.pos.y - vec.y) <= self.dimy / 2 + a
                        and abs(self.pos.z - vec.z) <= self.dimz / 2 + a):
                    return True
            return False
        # if input_vec is not iterable
        except TypeError:
            if (abs(self.pos.x - input_vec.x) <= self.dimx / 2 + a
                    and abs(self.pos.y - input_vec.y) <= self.dimy / 2 + a
                    and abs(self.pos.z - input_vec.z) <= self.dimz / 2 + a):
                return True
            else:
                return False
