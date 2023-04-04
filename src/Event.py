import numpy as np
from uproot_methods.classes.TVector3 import TVector3


########################################################################################################################
class Event:
    """represents a single event of a root tree.

    Attributes:
        ### For detailed description of the attributes consult the gccb-wiki
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
        RecoEnergy_e (vector<PhysicVar>)
        RecoEnergy_p (vector<PhysicVar>)
        RecoPosition_e (vector<PhysicVec>)
        RecoPosition_p (vector<PhysicVec>)
        RecoDirection_scatter (vector<PhysicVec>)

        RecoClusterPosition (vector<TVector3>)
        RecoClusterPosition_uncertainty (vector<TVector3>)
        RecoClusterEnergies_values (vector<PhysicVar>)
        RecoClusterEnergies_uncertainty (vector<PhysicVar>)
        RecoClusterEntries (vector<int>)

        scatterer (obj:SiFiCC_module)
        absorber (obj:SiFiCC_module)

    """

    def __init__(self,
                 EventNumber,
                 MCEnergy_Primary,
                 MCEnergy_e,
                 MCEnergy_p,
                 MCPosition_source,
                 MCSimulatedEventType,
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
                 MCEventStartTime,
                 MCComptonTime,
                 scatterer,
                 absorber):

        # Monte-Carlo information
        self.EventNumber = EventNumber
        self.MCEnergy_Primary = MCEnergy_Primary
        self.MCEnergy_e = MCEnergy_e
        self.MCEnergy_p = MCEnergy_p
        self.MCPosition_source = MCPosition_source
        self.MCSimulatedEventType = MCSimulatedEventType
        self.MCDirection_source = MCDirection_source
        self.MCComptonPosition = MCComptonPosition
        self.MCDirection_scatter = MCDirection_scatter
        self.MCPosition_e = MCPosition_e
        self.MCInteractions_e = MCInteractions_e
        self.MCPosition_p = MCPosition_p
        self.MCInteractions_p = MCInteractions_p
        self.MCEventStartTime = MCEventStartTime
        self.MCComptonTime = MCComptonTime

        # Cut-Based information
        self.Identified = Identified
        # Cut-Based reco data can not be accessed in python due to the entries being branches

        # Cluster information
        self.RecoClusterPosition = RecoClusterPosition
        self.RecoClusterPosition_uncertainty = RecoClusterPosition_uncertainty
        self.RecoClusterEnergies_values = RecoClusterEnergies_values
        self.RecoClusterEnergies_uncertainty = RecoClusterEnergies_uncertainty
        self.RecoClusterEntries = RecoClusterEntries

        # timing information
        self.RecoClusterTimestamps = RecoClusterTimestamps
        # get position sorted indices
        idx_position = self.sort_clusters_position()
        self.RecoClusterTimestamps_relative = RecoClusterTimestamps - min(RecoClusterTimestamps)

        # scattering angle
        self.theta = self.calculate_theta(self.MCEnergy_e, self.MCEnergy_p)

        self.scatterer = scatterer
        self.absorber = absorber

        # correction of MCDirection_source quantity
        vec_ref = self.MCComptonPosition - self.MCPosition_source
        # print(vec_ref.theta, self.MCDirection_source.theta)
        if not abs(vec_ref.phi - self.MCDirection_source.phi) < 0.1 or not abs(
                vec_ref.theta - self.MCDirection_source.theta) < 0.1:
            self.MCDirection_source = self.MCComptonPosition - self.MCPosition_source
            self.MCDirection_source /= self.MCDirection_source.mag

        # ---------------------------------------------------------------------------------------------
        # Event tagging
        # event identification steps:
        #
        # - compton event: A compton event took place on Monte Carlo level
        # - full compton event: A compton event with Monte Carlo entries of electron in scatterer and
        #                       absorber entries of scattered photon
        # - ideal compton event: Full compton event, with single scattering in scatterer and next interaction in
        #                        absorber

        # check if simulated event type is a real coincidence ( + pileup)
        self.is_real_coincidence = True if self.MCSimulatedEventType in [2, 5] else False

        # check if the event is a Compton event
        # Compton events have a positive MC electron energy
        self.is_compton = False
        if self.is_real_coincidence:
            if self.MCEnergy_e != 0:
                self.is_compton = True

        # check if event is distributed
        # distributed = electron interaction in scatterer and photon interaction in absorber
        self.is_compton_distributed = False
        self.is_compton_pseudo_distributed = False
        self.is_compton_pseudo_complete = False
        self.MCPosition_p_first = TVector3(0, 0, 0)
        self.MCPosition_e_first = TVector3(0, 0, 0)

        # compton distributed
        if self.is_compton:
            if len(self.MCPosition_p) >= 2 and len(self.MCPosition_e) >= 1:
                if (self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20):
                    if scatterer.is_vec_in_module(self.MCPosition_e):
                        if ((self.MCInteractions_p[1:] > 0) & (self.MCInteractions_p[1:] < 10)).any():
                            if scatterer.is_vec_in_module(self.MCPosition_p[0]) \
                                    and absorber.is_vec_in_module(self.MCPosition_p[1]):
                                self.is_compton_distributed = True
                                for idx in range(0, len(self.MCInteractions_e)):
                                    if 10 <= self.MCInteractions_e[idx] < 20 and scatterer.is_vec_in_module(
                                            self.MCPosition_e[idx]):
                                        self.MCPosition_e_first = self.MCPosition_e[idx]
                                        break

                                for idx in range(1, len(self.MCInteractions_p)):
                                    if 0 < self.MCInteractions_p[idx] < 10 and absorber.is_vec_in_module(
                                            self.MCPosition_p[idx]):
                                        self.MCPosition_p_first = self.MCPosition_p[idx]
                                        break

        # compton pseudo distributed
        if self.is_compton:
            if len(self.MCPosition_p) >= 2 and len(self.MCPosition_e) >= 1:
                if (self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20):
                    if scatterer.is_vec_in_module(self.MCPosition_e):
                        if ((self.MCInteractions_p[1:] > 0) & (self.MCInteractions_p[1:] < 10)).any():
                            if scatterer.is_vec_in_module(self.MCPosition_p[0]) \
                                    and absorber.is_vec_in_module(self.MCPosition_p[1:]):
                                self.is_compton_pseudo_distributed = True
                                for idx in range(0, len(self.MCInteractions_e)):
                                    if 10 <= self.MCInteractions_e[idx] < 20 and scatterer.is_vec_in_module(
                                            self.MCPosition_e[idx]):
                                        self.MCPosition_e_first = self.MCPosition_e[idx]
                                        break

                                for idx in range(1, len(self.MCInteractions_p)):
                                    if 0 < self.MCInteractions_p[idx] < 10 and absorber.is_vec_in_module(
                                            self.MCPosition_p[idx]):
                                        self.MCPosition_p_first = self.MCPosition_p[idx]
                                        break

        # compton pseudo complete
        if self.is_compton:
            if len(self.MCPosition_p) >= 2 and len(self.MCPosition_e) >= 1:
                if (self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20):
                    if scatterer.is_vec_in_module(self.MCPosition_e):
                        if scatterer.is_vec_in_module(self.MCPosition_p[0]) \
                                and absorber.is_vec_in_module(self.MCPosition_p[1:]):
                            self.is_compton_pseudo_complete = True
                            for idx in range(0, len(self.MCInteractions_e)):
                                if 10 <= self.MCInteractions_e[idx] < 20 and scatterer.is_vec_in_module(
                                        self.MCPosition_e[idx]):
                                    self.MCPosition_e_first = self.MCPosition_e[idx]
                                    break

                            for idx in range(1, len(self.MCInteractions_p)):
                                if absorber.is_vec_in_module(self.MCPosition_p[idx]):
                                    self.MCPosition_p_first = self.MCPosition_p[idx]
                                    break

        # OVERWRITING TRUE ELECTRON POSITION WITH THE TRUE COMPTON SCATTERING POSITION
        self.MCPosition_e_first = self.MCComptonPosition

        # new better super optimal tagging
        self.is_ideal_compton = False
        if self.is_compton:
            # baseline conditions:
            # - event is a compton event
            # - at least 2 photon interactions, first one in scatterer, at least one in absorber
            # - at least 1 electron interaction, in scatterer
            if len(self.MCPosition_p) >= 2:
                # set compton scattering position
                if ((self.MCInteractions_p > 0) & (self.MCInteractions_p < 10)).any():
                    for idx in range(0, len(self.MCInteractions_p)):
                        if 0 <= self.MCInteractions_p[idx] < 10 and scatterer.is_vec_in_module(
                                self.MCPosition_p[idx]):
                            self.MCPosition_e_first = self.MCComptonPosition
                            break
                # set absorption position
                for idx in range(0, len(self.MCInteractions_p)):
                    if 0 <= self.MCInteractions_p[idx] < 20 and absorber.is_vec_in_module(
                            self.MCPosition_p[idx]):
                        # check additionally if the interaction is in the scattering direction
                        vec1 = np.array([self.MCPosition_p.x[idx] - self.MCComptonPosition.x,
                                         self.MCPosition_p.y[idx] - self.MCComptonPosition.y,
                                         self.MCPosition_p.z[idx] - self.MCComptonPosition.z])
                        vec2 = np.array([self.MCDirection_scatter.x,
                                         self.MCDirection_scatter.y,
                                         self.MCDirection_scatter.z])
                        tmp_angle = self.vec_angle(vec1, vec2)
                        if tmp_angle < 0.01:
                            self.MCPosition_p_first = self.MCPosition_p[idx]
                            self.is_ideal_compton = True
                            break
                        break

        """
        # check if the event is a Compton event
        # Compton events have a positive MC electron energy
        self.is_compton = True if self.MCEnergy_e != 0 else False

        # check if the event is a full compton event
        # TODO: need to do this nicer
        self.e_pos = False
        self.p_pos = False
        self.MCPosition_p_first = TVector3(0, 0, 0)
        self.MCPosition_e_first = TVector3(0, 0, 0)

        if self.is_compton:
            # check if enough interaction of electron and photon took place
            if len(self.MCPosition_p) >= 2 and len(self.MCPosition_e) >= 1:
                # check for the correct type of events
                if ((self.MCInteractions_p[1:] > 0) & (self.MCInteractions_p[1:] < 10)).any() and (
                        (self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20)):
                    # initialize the compton scattering and photon absorption interation positions
                    # condition on first interaction position initialization:
                    # - electron interaction is in scatterer
                    # - photon interaction is in absorber
                    for idx in range(0, len(self.MCInteractions_e)):
                        if 10 <= self.MCInteractions_e[idx] < 20 and scatterer.is_vec_in_module(self.MCPosition_e[idx]):
                            self.MCPosition_e_first = self.MCPosition_e[idx]
                            self.e_pos = True
                            break

                    for idx in range(1, len(self.MCInteractions_p)):
                        if 0 < self.MCInteractions_p[idx] < 10 and absorber.is_vec_in_module(self.MCPosition_p[idx]):
                            self.MCPosition_p_first = self.MCPosition_p[idx]
                            self.p_pos = True
                            break

        if self.e_pos and self.p_pos:
            self.is_fullcompton = True
        else:
            self.is_fullcompton = False

        # check if the event is a complete Compton event
        # complete Compton event= Compton event + both e and p go through a second interation in which
        # 0 < p interaction < 10
        # 10 <= e interaction < 20
        # Note: first interaction of p is the compton event
        if (self.is_compton
                and len(self.MCPosition_p) >= 2
                and len(self.MCPosition_e) >= 1
                and ((self.MCInteractions_p[1:] > 0) & (self.MCInteractions_p[1:] < 10)).any()
                and ((self.MCInteractions_e[0] >= 10) & (self.MCInteractions_e[0] < 20))):
            self.is_complete_compton = True
        else:
            self.is_complete_compton = False

        # initialize e & p first interaction position
        if self.is_complete_compton:
            for idx in range(1, len(self.MCInteractions_p)):
                if 0 < self.MCInteractions_p[idx] < 10:
                    self.MCPosition_p_first = self.MCPosition_p[idx]
                    break
            for idx in range(0, len(self.MCInteractions_e)):
                if 10 <= self.MCInteractions_e[idx] < 20:
                    self.MCPosition_e_first = self.MCPosition_e[idx]
                    break
        else:
            self.MCPosition_p_first = TVector3(0, 0, 0)
            self.MCPosition_e_first = TVector3(0, 0, 0)

        # check if the event is a complete distributed Compton event
        # complete distributed Compton event= complete Compton event + each e and p go through a secondary
        # interaction in a different module of the SiFiCC
        if (self.is_complete_compton
                and scatterer.is_vec_in_module(self.MCPosition_e)
                and absorber.is_vec_in_module(self.MCPosition_p)):
            self.is_complete_distributed_compton = True
        else:
            self.is_complete_distributed_compton = False

        # check if the event is an ideal Compton event and what type is it (EP or PE)
        # ideal Compton event = complete distributed Compton event where the next interaction of both
        # e and p is in the different modules of SiFiCC
        if (self.is_complete_compton
                and scatterer.is_vec_in_module(self.MCPosition_e_first)
                and absorber.is_vec_in_module(self.MCPosition_p_first)
                and self.MCSimulatedEventType == 2):
            self.is_ideal_compton = True
            self.is_ep = True
            self.is_pe = False
        elif (self.is_complete_compton
              and scatterer.is_vec_in_module(self.MCPosition_p_first)
              and absorber.is_vec_in_module(self.MCPosition_e_first)
              and self.MCSimulatedEventType == 2):
            self.is_ideal_compton = True
            self.is_ep = False
            self.is_pe = True
        else:
            self.is_ideal_compton = False
            self.is_ep = False
            self.is_pe = False
        """

    ####################################################################################################################

    def unit_vec(self, vec):
        return vec / np.sqrt(np.dot(vec, vec))

    def vec_angle(self, vec1, vec2):
        return np.arccos(np.clip(np.dot(self.unit_vec(vec1), self.unit_vec(vec2)), -1.0, 1.0))

    def calculate_theta(self, e1, e2):
        """
        Calculate scattering angle theta in radiants.

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

    ####################################################################################################################
