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

        # Cut-Based information
        self.Identified = Identified
        # Cut-Based reco data can not be accessed in python due to the entries being branches

        # Cluster information
        self.RecoClusterPosition = RecoClusterPosition
        self.RecoClusterPosition_uncertainty = RecoClusterPosition_uncertainty
        self.RecoClusterEnergies_values = RecoClusterEnergies_values
        self.RecoClusterEnergies_uncertainty = RecoClusterEnergies_uncertainty
        self.RecoClusterEntries = RecoClusterEntries

        self.scatterer = scatterer
        self.absorber = absorber

        # event identification steps
        # positive events are described as ideal compton events
        # the needed tags will be defined below
        # TODO: check if Awals event tags are correct

        # check if the event is a valid event by considering the clusters associated with it
        # the event is considered distributed if there is at least one cluster within each module of the SiFiCC
        if (len(self.RecoClusterEnergies_values) >= 2
                and scatterer.is_vec_in_module(self.RecoClusterPosition)
                and absorber.is_vec_in_module(self.RecoClusterPosition)):
            self.is_distributed = True
        else:
            self.is_distributed = False

        # check if the event is a Compton event
        # Compton events have a positive MC electron energy
        self.is_compton = True if self.MCEnergy_e != 0 else False

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

    ####################################################################################################################

    def get_electron_energy(self):
        idx_scatterer, _ = self.sort_clusters_by_module(use_energy=True)
        return self.RecoClusterEnergies_values[idx_scatterer[0]], self.RecoClusterEnergies_uncertainty[idx_scatterer[0]]

    def get_photon_energy(self):
        _, idx_absorber = self.sort_clusters_by_module(use_energy=True)
        photon_energy_value = np.sum(self.RecoClusterEnergies_values[idx_absorber])
        photon_energy_uncertainty = np.sqrt(np.sum(self.RecoClusterEnergies_uncertainty[idx_absorber]**2))
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
