import numpy as np
from uproot_methods.classes.TVector3 import TVector3


# ----------------------------------------------------------------------------------------------------------------------

class Event:
    """
    Represents a single event of a root tree. For detailed description of the attributes consult the gccb-wiki
    Attributes:

        EventNumber (int)
        MCEnergy_Primary (double)
        MCEnergy_e (double)
        MCEnergy_p (double)
        MCPosition_source (TVector3)
        MCSimulatedEventType (int)
        MCDirection_source (TVector3)
        MCComptonPosition (TVector3)
        MCDirection_scatter (TVector3)
        MCPosition_e (vector<TVector3>)
        MCInteractions_e (vector<int>)
        MCPosition_p (vector<TVector3>)
        MCInteractions_p (vector<int>)

        Identified (int)

        RecoClusterPosition (vector<TVector3>)
        RecoClusterPosition_uncertainty (vector<TVector3>)
        RecoClusterEnergies_values (vector<PhysicVar>)
        RecoClusterEnergies_uncertainty (vector<PhysicVar>)
        RecoClusterEntries (vector<int>)

        scatterer (obj:SiFiCC_module)
        absorber (obj:SiFiCC_module)

    """

    def __init__(self,
                 bglobal,
                 breco,
                 bcluster,
                 bsipm,
                 EventNumber,
                 MCSimulatedEventType,
                 MCEnergy_Primary,
                 MCEnergy_e,
                 MCEnergy_p,
                 MCPosition_source,
                 MCDirection_source,
                 MCComptonPosition,
                 MCDirection_scatter,
                 MCPosition_e,
                 MCInteractions_e,
                 MCPosition_p,
                 MCInteractions_p,
                 Identified,
                 RecoClusterPosition,
                 RecoClusterPosition_uncertainty,
                 RecoClusterEnergies_values,
                 RecoClusterEnergies_uncertainty,
                 RecoClusterEntries,
                 RecoClusterTimestamps,
                 SiPM_triggertime,
                 SiPM_qdc,
                 SiPM_position,
                 SiPM_id,
                 fibre_time,
                 fibre_energy,
                 fibre_position,
                 fibre_id,
                 scatterer,
                 absorber):

        # Information level tags
        self.bglobal = bglobal
        self.breco = breco
        self.bcluster = bcluster
        self.bsipm = bsipm

        # Global information
        self.EventNumber = EventNumber
        self.MCSimulatedEventType = MCSimulatedEventType
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCEnergy_e = MCEnergy_e
        self.MCEnergy_p = MCEnergy_p
        self.MCPosition_source = MCPosition_source
        self.MCDirection_source = MCDirection_source
        self.MCComptonPosition = MCComptonPosition
        self.MCDirection_scatter = MCDirection_scatter
        self.MCPosition_e = MCPosition_e
        self.MCInteractions_e = MCInteractions_e
        self.MCPosition_p = MCPosition_p
        self.MCInteractions_p = MCInteractions_p

        # LEGACY FEATURES
        # self.MCEventStartTime = MCEventStartTime
        # self.MCComptonTime = MCComptonTime

        # Reco information (Cut-Based Reconstruction)
        self.Identified = Identified
        # Cut-Based reco data can not be accessed in python due to the entries being branches!

        # Cluster information
        self.RecoClusterPosition = RecoClusterPosition
        self.RecoClusterPosition_uncertainty = RecoClusterPosition_uncertainty
        self.RecoClusterEnergies_values = RecoClusterEnergies_values
        self.RecoClusterEnergies_uncertainty = RecoClusterEnergies_uncertainty
        self.RecoClusterEntries = RecoClusterEntries
        self.RecoClusterTimestamps = RecoClusterTimestamps
        if self.bcluster:
            self.RecoClusterTimestamps_relative = RecoClusterTimestamps - min(RecoClusterTimestamps)

        # SiPM and Fibre information
        self.SiPM_triggertime = SiPM_triggertime
        self.SiPM_qdc = SiPM_qdc
        self.SiPM_position = SiPM_position
        self.SiPM_id = SiPM_id
        self.fibre_time = fibre_time
        self.fibre_energy = fibre_energy
        self.fibre_position = fibre_position
        self.fibre_id = fibre_id

        # Detector modules
        self.scatterer = scatterer
        self.absorber = absorber

        # --------------------------------------------------------------------------------------------------------------
        # Temporary corrections to MC-Truth with additional control tagging

        # correction of MCDirection_source quantity
        vec_ref = self.MCComptonPosition - self.MCPosition_source
        # print(vec_ref.theta, self.MCDirection_source.theta)
        if not abs(vec_ref.phi - self.MCDirection_source.phi) < 0.1 or not abs(
                vec_ref.theta - self.MCDirection_source.theta) < 0.1:
            self.MCDirection_source = self.MCComptonPosition - self.MCPosition_source
            self.MCDirection_source /= self.MCDirection_source.mag

        # scattering angle
        # calculated from energy and vector dot product of direction vectors given by simulation output
        self.theta_energy = self.calc_theta_energy(self.MCEnergy_e, self.MCEnergy_p)
        self.theta_dotvec = self.calc_theta_dotvec(self.MCDirection_source, self.MCDirection_scatter)

        # correction of photon absorber position

        # --------------------------------------------------------------------------------------------------------------
        # Event tagging and deep learning targets

        # initialize neural network targets
        self.target_position_e = TVector3(0, 0, 0)
        self.target_position_p = TVector3(0, 0, 0)
        self.target_energy_e = 0.0
        self.target_energy_p = 0.0
        self.target_angle_theta = 0.0
        self.compton_tag = False

        # set correct targets
        self.set_target_positions()
        self.set_target_energy()
        self.set_target_angle()
        self.set_compton_tag()

    # ------------------------------------------------------------------------------------------------------------------
    # neural network targets

    def set_target_positions(self):
        """
        Set scatterer and absorber target interaction position for neural network regression

        return:
            None
        """

        self.target_position_e = self.MCComptonPosition

        # scan for first absorber interaction that has the correct scattering direction
        for idx in range(0, len(self.MCInteractions_p)):
            if 0 <= self.MCInteractions_p[idx] < 20 and self.absorber.is_vec_in_module(
                    self.MCPosition_p[idx]):

                # check additionally if the interaction is in the scattering direction
                tmp_angle = self.calc_theta_dotvec(self.MCPosition_p[idx] - self.MCComptonPosition,
                                                   self.MCDirection_scatter)
                if tmp_angle < 1e-3:
                    self.target_position_p = self.MCPosition_p[idx]
                    break

    def set_target_energy(self):
        """

        return:
            None
        """

        self.target_energy_e = self.MCEnergy_e
        self.target_energy_p = self.MCEnergy_p

    def set_target_angle(self):
        """

        return:
            None
        """

        self.target_angle_theta = self.theta_dotvec

    def set_compton_tag(self):
        """
        Scans if a given event is an ideal Compton event and sets the corresponding tag. This tag is used as a neural
        network classification target.

        Ideal Compton event:
            - Compton energy stored in event
            - Interaction of primary gamma in scatterer and at least one interaction in the absorber (any)

        return:
            None
        """
        self.compton_tag = False
        # check for electron energy (compton event took place)
        if self.MCEnergy_e > 0.0:
            # check if primary gamma interacted at least 2 times
            if len(self.MCPosition_p) >= 2:
                if self.scatterer.is_vec_in_module(self.target_position_e) and self.absorber.is_vec_in_module(
                        self.target_position_p):
                    self.compton_tag = True

    # ------------------------------------------------------------------------------------------------------------------
    # SiPM and fibre feature map generation
    # Code manly written by Philippe Clement from NN fibre identification
    @staticmethod
    def sipm_id_to_position(sipm_id):
        # determine y
        y = sipm_id // 368
        # remove third dimension
        sipm_id -= (y * 368)
        # x and z in scatterer
        if sipm_id < 112:
            x = sipm_id // 28
            z = (sipm_id % 28) + 2
        # x and z in absorber
        else:
            x = (sipm_id + 16) // 32
            z = (sipm_id + 16) % 32
        return int(x), int(y), int(z)

    def get_sipm_feature_map(self, padding=2, gap_padding=4):
        # hardcoded detector size
        dimx = 12
        dimy = 2
        dimz = 32

        ary_feature = np.zeros(shape=(dimx + 2 * padding + gap_padding, dimy + 2 * padding, dimz + 2 * padding, 2))

        for i, sipm_id in enumerate(self.SiPM_id):
            x, y, z = self.sipm_id_to_position(sipm_id=sipm_id)
            x_final = x + padding if x < 4 else x + padding + gap_padding
            y_final = y + padding
            z_final = z + padding

            ary_feature[x_final, y_final, z_final, 0] = self.SiPM_qdc[i]
            ary_feature[x_final, y_final, z_final, 1] = self.SiPM_triggertime[i]

        return ary_feature

    # --------------------------------------------------------------------------
    # Graph generation methods

    def get_edge_features(self, idx1, idx2):
        vec = self.RecoClusterPosition[idx2] - self.RecoClusterPosition[idx1]
        r = vec.mag
        phi = vec.phi
        theta = vec.theta

        return r, phi, theta


    # --------------------------------------------------------------------------

    def calc_theta_energy(self, e1, e2):
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

    def calc_theta_dotvec(self, vec1, vec2):
        """
        Calculate scattering angle theta in radiant from the dot product of 2 vectors.

        Args:
             vec1 (TVector3): 3-dim origin vector, direction vector of source
             vec2 (TVector3): 3-dim origin vector, direction vector of compton scattering
        """
        if vec1.mag == 0 or vec2.mag == 0:
            return 0.0

        ary_vec1 = np.array([vec1.x, vec1.y, vec1.z])
        ary_vec2 = np.array([vec2.x, vec2.y, vec2.z])

        ary_vec1 /= np.sqrt(np.dot(ary_vec1, ary_vec1))
        ary_vec2 /= np.sqrt(np.dot(ary_vec2, ary_vec2))

        return np.arccos(np.clip(np.dot(ary_vec1, ary_vec2), -1.0, 1.0))

    def get_electron_energy(self):
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterEnergies_values[idx_scatterer[0]], self.RecoClusterEnergies_uncertainty[idx_scatterer[0]]

    def get_photon_energy(self):
        _, idx_absorber = self.sort_clusters_by_module(use_energy=True)
        photon_energy_value = np.sum(self.RecoClusterEnergies_values[idx_absorber])
        photon_energy_uncertainty = np.sqrt(np.sum(self.RecoClusterEnergies_uncertainty[idx_absorber] ** 2))
        return photon_energy_value, photon_energy_uncertainty

    def get_electron_position(self):
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterPosition[idx_scatterer[0]], self.RecoClusterPosition_uncertainty[idx_scatterer[0]]

    def get_photon_position(self):
        _, idx_absorber = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterPosition[idx_absorber[0]], self.RecoClusterPosition_uncertainty[idx_absorber[0]]

    def sort_clusters_energy(self):
        """ sort events by highest energy in descending order

        return: sorted array idx

        """
        return np.flip(np.argsort(self.RecoClusterEnergies_values))

    def sort_clusters_position(self):
        """ sort events by lowest x position in ascending order

        return: sorted array idx

        """
        return np.argsort(self.RecoClusterPosition.x)

    def sort_clusters_by_module(self, use_energy=True):
        """ sort clusters (sorted by energy) by corresponding module only creates list of array idx's.

        Args:
            use_energy (bool): True if clusters are sorted by energy before, else sorted by position

        return: sorted array idx scatterer, absorber

        """
        RecoCluster_idx_scatterer = []
        RecoCluster_idx_absorber = []

        # check energy tag
        if use_energy:
            idx_sort = self.sort_clusters_energy()
        else:
            idx_sort = np.arange(0, len(self.RecoClusterPosition), 1)

        # sort cluster
        for idx in idx_sort:
            if self.scatterer.is_vec_in_module(self.RecoClusterPosition[idx]):
                RecoCluster_idx_scatterer.append(idx)
            if self.absorber.is_vec_in_module(self.RecoClusterPosition[idx]):
                RecoCluster_idx_absorber.append(idx)

        return RecoCluster_idx_scatterer, RecoCluster_idx_absorber

    def argmatch_cluster(self, tvec3, indexing=None, a=1):
        """ takes a point and finds the first cluster matching the point within the cluster uncertainty.

        Args:
            tvec3 (TVector3): vector pointing to the cluster
            indexing (list): list of cluster indices to define an iteration order
            a: sigma range (factor multiplied to sigma)

        return: idx if cluster is matched, else -1

        """
        if indexing is None:
            # iterate all cluster positions + uncertainty
            for i in range(len(self.RecoClusterPosition)):
                tcluster = self.RecoClusterPosition[i]
                tcluster_unc = self.RecoClusterPosition_uncertainty[i]
                # check if absolute x,y,z difference is smaller than absolute uncertainty
                if (abs(tvec3.x - tcluster.x) <= a * abs(tcluster_unc.x)
                        and abs(tvec3.y - tcluster.y) <= a * abs(tcluster_unc.y)
                        and abs(tvec3.z - tcluster.z) <= a * abs(tcluster_unc.z)):
                    return i
            else:
                return -1
        else:
            # iterate all cluster positions + uncertainty
            for i, idx in enumerate(indexing):
                tcluster = self.RecoClusterPosition[idx]
                tcluster_unc = self.RecoClusterPosition_uncertainty[idx]
                # check if absolute x,y,z difference is smaller than absolute uncertainty
                if (abs(tvec3.x - tcluster.x) <= abs(tcluster_unc.x)
                        and abs(tvec3.y - tcluster.y) <= abs(tcluster_unc.y)
                        and abs(tvec3.z - tcluster.z) <= abs(tcluster_unc.z)):
                    return i
            else:
                return -1

    def get_prime_vector(self):
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterPosition[idx_scatterer[0]]

    def get_relative_vector(self, tvec3, subtract_prime=True):
        tvec3_prime = self.get_prime_vector()
        # subtract prime vector
        if subtract_prime:
            tvec3 = tvec3 - tvec3_prime
        # rotate tvec3 so that the prime vector aligns with the x-axis
        tvec3 = tvec3.rotatez(-tvec3_prime.phi).rotatey(-tvec3_prime.theta + np.pi / 2)

        return tvec3

    # ------------------------------------------------------------------------------------------------------------------
    # correction functions

    def check_absorber_interaction(self):
        """
        Checks the type of interaction in the absorber

        return: (int) based on results
            0 - check failed
            1 - intended interaction
            2 - no correct interaction found
            3 - delayed correct interaction
            4 - non-primary correct interaction
        """
        list_interact = []
        returner = 0
        for i in range(len(self.MCInteractions_p)):
            if self.absorber.is_vec_in_module(self.MCPosition_p[i]):
                if 0 < self.MCInteractions_p[i] < 10:
                    tmp_angle = self.calc_theta_dotvec(self.MCPosition_p[i] - self.MCComptonPosition,
                                                       self.MCDirection_scatter)
                    if tmp_angle < 0.01:
                        if returner in [2, 4]:
                            returner = 3
                        else:
                            returner = 1
                        break

                    else:
                        returner = 2
                if 10 <= self.MCInteractions_p[i]:
                    tmp_angle = self.calc_theta_dotvec(self.MCPosition_p[i] - self.MCComptonPosition,
                                                       self.MCDirection_scatter)
                    if tmp_angle < 0.01:
                        returner = 4
                    else:
                        returner = 2
                # print("DEBUG: RETURNER", returner, self.MCInteractions_p[i], tmp_angle)
            else:
                continue

        return returner
