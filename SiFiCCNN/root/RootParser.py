# ##################################################################################################
# RootParser Class
#
# Class to load and process root files. Main goal is to iterate over all event entries of a root
# file and yield the events at each position. Additionally, this class determines what type of
# information is stored in the file and generate the corresponding event type
#
# ##################################################################################################


import uproot
import tqdm
import sys
import os

from SiFiCCNN.root.Event import Event, EventCluster, EventSiPM
from SiFiCCNN.root.Detector import Detector


class RootParser:
    """
    loading and preprocessing of root data for events and setup tree.

    """

    def __init__(self,
                 rootfile_path):

        # Base attributes of a root file
        self.file_base = os.path.basename(rootfile_path)
        self.file_name = os.path.splitext(self.file_base)[0]

        # open root file with uproot
        rootfile = uproot.open(rootfile_path)
        self.rootfile_path = rootfile_path
        self.events = rootfile[b"Events"]
        self.setup = rootfile[b"Setup"]
        self.events_entries = self.events.numentries
        self.events_keys = self.events.keys()

        # List of root leaves expected for all possible root file structures
        # Note: b in front of the string account for
        # byte stare of strings in root
        self.leaves_global = [b"EventNumber",
                              b'MCSimulatedEventType',
                              b'MCEnergy_Primary',
                              b'MCEnergy_e',
                              b'MCEnergy_p',
                              b'MCPosition_source',
                              b'MCDirection_source',
                              b'MCComptonPosition',
                              b'MCDirection_scatter',
                              b'MCPosition_e',
                              b'MCInteractions_e',
                              b'MCPosition_p',
                              b'MCInteractions_p']
        self.leaves_cluster = [b'RecoClusterPositions.position',
                               b'RecoClusterPositions.uncertainty',
                               b'RecoClusterEnergies.value',
                               b'RecoClusterEnergies.uncertainty',
                               b'RecoClusterEntries',
                               b"RecoClusterTimestamps"]
        self.leaves_sipm = [b"SiPMData.fSiPMTriggerTime",
                            b"SiPMData.fSiPMQDC",
                            b"SiPMData.fSiPMPosition",
                            b"SiPMData.fSiPMId",
                            b"FibreData.fFibreTime",
                            b"FibreData.fFibreEnergy",
                            b"FibreData.fFibrePosition",
                            b"FibreData.fFibreId"]
        self.leaves_cluster_keys = [b'RecoClusterPositions',
                                    b'RecoClusterEnergies',
                                    b'RecoClusterEntries',
                                    b"RecoClusterTimestamps"]
        self.leaves_sipm_keys = [b"SiPMData",
                                 b"FibreData"]
        self.leaves_reco = [b'Identified']

        # define information level in root-file:
        self.ifglobal = False
        self.ifcluster = False
        self.ifreco = False
        self.ifsipm = False
        self.list_leaves_final = []
        if set(self.leaves_global).issubset(self.events_keys):
            self.ifglobal = True
            self.list_leaves_final += self.leaves_global
        # try to catch exception
        if not self.ifglobal:
            self.leaves_global[2] = b"MCEnergyPrimary"
            if set(self.leaves_global).issubset(self.events_keys):
                self.ifglobal = True
                self.list_leaves_final += self.leaves_global

        if set(self.leaves_reco).issubset(self.events_keys):
            self.ifreco = True
            self.list_leaves_final += self.leaves_reco
        if set(self.leaves_cluster_keys).issubset(self.events_keys):
            self.ifcluster = True
            self.list_leaves_final += self.leaves_cluster
        if set(self.leaves_sipm_keys).issubset(self.events_keys):
            self.ifsipm = True
            self.list_leaves_final += self.leaves_sipm

        # create SIFICC-Module objects for scatterer and absorber
        self.scatterer = Detector(self.setup["ScattererPosition"].array()[0],
                                  self.setup["ScattererThickness_x"].array()[0],
                                  self.setup["ScattererThickness_y"].array()[0],
                                  self.setup["ScattererThickness_z"].array()[0])
        self.absorber = Detector(self.setup["AbsorberPosition"].array()[0],
                                 self.setup["AbsorberThickness_x"].array()[0],
                                 self.setup["AbsorberThickness_y"].array()[0],
                                 self.setup["AbsorberThickness_z"].array()[0])

    def iterate_events(self, n=None):
        """
        iteration over the events root tree

        Args:
            n (int) or (None):  total number of events being returned,
                                if None the maximum number will be iterated.

        Returns:
            yield event at every root tree entry

        """
        # evaluate parameter n
        if n is None:
            n = self.events_entries
        # TODO: exception for negative entries

        # define progress bar
        progbar = tqdm.tqdm(total=n, ncols=100, file=sys.stdout,
                            desc="iterating root tree")
        progbar_step = 0
        progbar_update_size = 1000

        for start, end, basket in self.events.iterate(self.list_leaves_final,
                                                      entrysteps=100000,
                                                      reportentries=True,
                                                      namedecode='utf-8',
                                                      entrystart=0,
                                                      entrystop=n):
            length = end - start
            for idx in range(length):
                # yield event
                yield self.__event_at_basket(basket, idx)

                progbar_step += 1
                if progbar_step % progbar_update_size == 0:
                    progbar.update(progbar_update_size)

        progbar.update(self.events_entries % progbar_update_size)
        progbar.close()

    def __event_at_basket(self, basket, idx):
        """
        create event object from a given position in a root basket

        Args:
            basket (obj: ???): root basket
            idx (int): position inside the root basket

        Return:
            event (obj: Event)

        """

        if self.ifcluster:
            event = EventCluster(EventNumber=basket["EventNumber"][idx],
                                 MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                                 MCEnergy_Primary=basket['MCEnergy_Primary'][idx],
                                 MCEnergy_e=basket['MCEnergy_e'][idx],
                                 MCEnergy_p=basket['MCEnergy_p'][idx],
                                 MCPosition_e=basket['MCPosition_e'][idx],
                                 MCInteractions_e=basket['MCInteractions_e'][idx],
                                 MCPosition_p=basket['MCPosition_p'][idx],
                                 MCInteractions_p=basket['MCInteractions_p'][idx],
                                 MCPosition_source=basket['MCPosition_source'][idx],
                                 MCDirection_source=basket['MCDirection_source'][idx],
                                 MCComptonPosition=basket['MCComptonPosition'][idx],
                                 MCDirection_scatter=basket['MCDirection_scatter'][idx],
                                 Identified=basket['Identified'][idx],
                                 RecoClusterPosition=basket['RecoClusterPositions.position'][idx],
                                 RecoClusterPosition_uncertainty=
                                 basket['RecoClusterPositions.uncertainty'][idx],
                                 RecoClusterEnergies_values=basket['RecoClusterEnergies.value'][
                                     idx],
                                 RecoClusterEnergies_uncertainty=
                                 basket['RecoClusterEnergies.uncertainty'][idx],
                                 RecoClusterEntries=basket['RecoClusterEntries'][idx],
                                 RecoClusterTimestamps=basket["RecoClusterTimestamps"][idx],
                                 module_scatterer=self.scatterer,
                                 module_absorber=self.absorber)

        if self.ifsipm:
            event = EventSiPM(EventNumber=basket["EventNumber"][idx],
                              MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                              MCEnergy_Primary=basket['MCEnergyPrimary'][idx],
                              MCEnergy_e=basket['MCEnergy_e'][idx],
                              MCEnergy_p=basket['MCEnergy_p'][idx],
                              MCPosition_e=basket['MCPosition_e'][idx],
                              MCInteractions_e=basket['MCInteractions_e'][idx],
                              MCPosition_p=basket['MCPosition_p'][idx],
                              MCInteractions_p=basket['MCInteractions_p'][idx],
                              MCPosition_source=basket['MCPosition_source'][idx],
                              MCDirection_source=basket['MCDirection_source'][idx],
                              MCComptonPosition=basket['MCComptonPosition'][idx],
                              MCDirection_scatter=basket['MCDirection_scatter'][idx],
                              SiPM_triggertime=basket["SiPMData.fSiPMTriggerTime"][idx],
                              SiPM_qdc=basket["SiPMData.fSiPMQDC"][idx],
                              SiPM_position=basket["SiPMData.fSiPMPosition"][idx],
                              SiPM_id=basket["SiPMData.fSiPMId"][idx],
                              fibre_time=basket["FibreData.fFibreTime"][idx],
                              fibre_energy=basket["FibreData.fFibreEnergy"][idx],
                              fibre_position=basket["FibreData.fFibrePosition"][idx],
                              fibre_id=basket["FibreData.fFibreId"][idx],
                              module_scatterer=self.scatterer,
                              module_absorber=self.absorber)

        else:
            event = Event(EventNumber=basket["EventNumber"][idx],
                          MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                          MCEnergy_Primary=basket['MCEnergy_Primary'][idx],
                          MCEnergy_e=basket['MCEnergy_e'][idx],
                          MCEnergy_p=basket['MCEnergy_p'][idx],
                          MCPosition_e=basket['MCPosition_e'][idx],
                          MCInteractions_e=basket['MCInteractions_e'][idx],
                          MCPosition_p=basket['MCPosition_p'][idx],
                          MCInteractions_p=basket['MCInteractions_p'][idx],
                          MCPosition_source=basket['MCPosition_source'][idx],
                          MCDirection_source=basket['MCDirection_source'][idx],
                          MCComptonPosition=basket['MCComptonPosition'][idx],
                          MCDirection_scatter=basket['MCDirection_scatter'][idx],
                          module_scatterer=self.scatterer,
                          module_absorber=self.absorber)

        return event

    def get_event(self, position):
        """
        Return event for a given position in the root file
        """
        for basket in self.events.iterate(self.list_leaves_final,
                                          entrystart=position,
                                          entrystop=position + 1,
                                          namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)

    '''
    # CURRENTLY DISABLED
    def export_npz_lookup(self, n=None, is_s1ax=False):
        """
        generates compressed npz file containing MC-Truth data and Cut-based
        reco data.

        Args:
            n (int or none):    number of events parsed from root tree,
                                None if all events are iterated
            is_s1ax (bool):     If true, skip events with more
                                than 1 scatterer cluster

        """

        # create empty arrays for full export to compressed .npz format
        # root data will be split into:
        # - Meta data: (EventNumber,
        #               MCSimulatedEventType,
        #               IdealCompton event tag,
        #               CB-identified)
        # - MonteCarlo-data: Monte-Carlo Event data
        # - CutBased-data: Cut-based reconstruction data

        ary_meta = np.zeros(shape=(self.events_entries, 4))
        ary_mc = np.zeros(shape=(self.events_entries, 10))
        ary_cb = np.zeros(shape=(self.events_entries, 10))
        ary_tags = np.zeros(shape=(self.events_entries, 5))

        # Fill Meta-data, Monte-Carlo data and Cluster data into empty arrays
        # Cut-based reco data is not iterable since uproot can't handle
        # the reco data stored in branches
        counter = 0
        for i, event in enumerate(self.iterate_events(n=n)):
            if is_s1ax:
                idx_scatterer, idx_absorber = event.sort_clusters_by_module(
                    use_energy=True)
                if not len(idx_scatterer) == 1:
                    continue

                if not len(idx_absorber) > 0:
                    continue

            ary_meta[counter, :] = [event.EventNumber,
                                    event.MCSimulatedEventType,
                                    event.is_ideal_compton * 1,
                                    event.Identified]

            ary_mc[counter, :] = [event.compton_tag * 1,
                                  event.target_energy_e,
                                  event.target_energy_p,
                                  event.target_position_e.x,
                                  event.target_position_e.y,
                                  event.target_position_e.z,
                                  event.target_position_p.x,
                                  event.target_position_p.y,
                                  event.target_position_p.z,
                                  event.target_angle_theta]

            e1, _ = event.get_electron_energy()
            e2, _ = event.get_photon_energy()
            p1, _ = event.get_electron_position()
            p2, _ = event.get_photon_position()
            ary_cb[counter, :] = [event.Identified,
                                  e1,
                                  e2,
                                  p1.x,
                                  p1.y,
                                  p1.z,
                                  p2.x,
                                  p2.y,
                                  p2.z,
                                  event.calculate_theta(e1, e2)]

            ary_tags[counter, :] = [event.is_real_coincidence * 1,
                                    event.is_compton * 1,
                                    event.is_compton_pseudo_complete * 1,
                                    event.is_compton_pseudo_distributed * 1,
                                    event.is_compton_distributed * 1]

            counter += 1

        # resize arrays
        ary_meta = ary_meta[:counter, :]
        ary_mc = ary_mc[:counter, :]
        ary_cb = ary_cb[:counter, :]
        ary_tags = ary_tags[:counter, :]

        # export dataframe to compressed .npz
        with open(self.file_name + "_lookup.npz", 'wb') as file:
            np.savez_compressed(file,
                                META=ary_meta,
                                MC_TRUTH=ary_mc,
                                CB_RECO=ary_cb,
                                TAGS=ary_tags)

        print("file saved: ", self.file_name + "_lookup.npz")
    '''