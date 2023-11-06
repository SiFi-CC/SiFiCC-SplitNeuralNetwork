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
    Loading and pre-processing of root data for events and setup tree. Main tasks include defining
    the amount of information inside the root tree, correctly setting up the necessary tree leaves,
    providing the methods needed for iteration over root files.

    """

    def __init__(self, path):

        # Base attributes of a root file
        self.file_base = os.path.basename(path)
        self.file_name = os.path.splitext(self.file_base)[0]

        # open root file with uproot
        rootfile = uproot.open(path)
        self.path = path
        self.events = rootfile[b"Events"]
        self.setup = rootfile[b"Setup"]
        self.events_entries = self.events.numentries
        self.events_keys = self.events.keys()

        # create SIFICC-Module objects for scatterer and absorber
        self.scatterer = Detector(self.setup["ScattererPosition"].array()[0],
                                  self.setup["ScattererThickness_x"].array()[0],
                                  self.setup["ScattererThickness_y"].array()[0],
                                  self.setup["ScattererThickness_z"].array()[0])
        self.absorber = Detector(self.setup["AbsorberPosition"].array()[0],
                                 self.setup["AbsorberThickness_x"].array()[0],
                                 self.setup["AbsorberThickness_y"].array()[0],
                                 self.setup["AbsorberThickness_z"].array()[0])

        # boolean values for information content of root file
        # This needed for the RootParser to define which type of event class is generated
        self.hasGlobal = False  # Event contains global prompt gamma track information
        self.hasCluster = False  # Event contains low level reconstruction clusters
        self.hasSiPM = False  # event contains SiPM (and fibre) response of detector
        self.set_information_content()

    def set_information_content(self):
        """
        Defines which type of event structure is in the ROOT-file. Possible types right now are:

        NOTE:   This is pretty much hard coded according to what is available from the simulation.
                If later on a low level reconstruction in the 4-to-1 coupling is possible, a new
                type and event subclass should be created.
        """

        # scan ROOT file information by checking which leave keys are in the tree
        # These are example keys, not all of them
        keys_global = [b"EventNumber", b'MCSimulatedEventType']
        keys_cluster = [b'RecoClusterPositions',
                        b'RecoClusterEnergies',
                        b'RecoClusterEntries',
                        b"RecoClusterTimestamps"]
        keys_sipm = [b"SiPMData",
                     b"FibreData"]

        if set(keys_global).issubset(self.events_keys):
            self.hasGlobal = True
        if set(keys_cluster).issubset(self.events_keys):
            self.hasCluster = True
        if set(keys_sipm).issubset(self.events_keys):
            self.hasSiPM = True

    def tree_leaves(self):
        """
        Generates a list of all leaves to be read out from the ROOT-file tree.
        Checks for all exceptions regarding legacy versions of simulation files.

        :return:
            list_leaves (list): list containing all leave names in binary strings

        NOTE: Strings are in binary because uproot
        """

        # initialize
        list_leaves = []

        # pre defined leaves
        leaves_global = [b"EventNumber",
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
        leaves_cluster = [b'RecoClusterPositions.position',
                          b'RecoClusterPositions.uncertainty',
                          b'RecoClusterEnergies.value',
                          b'RecoClusterEnergies.uncertainty',
                          b'RecoClusterEntries',
                          b"RecoClusterTimestamps",
                          b"Identified"]
        leaves_sipm = [b"SiPMData.fSiPMTimeStamp",
                       b"SiPMData.fSiPMPhotonCount",
                       b"SiPMData.fSiPMPosition",
                       b"SiPMData.fSiPMId",
                       b"FibreData.fFibreTime",
                       b"FibreData.fFibreEnergy",
                       b"FibreData.fFibrePosition",
                       b"FibreData.fFibreId"]

        # Exception "photonCount - timestamp"
        # Later simulation updates changed the naming convention for SiPM attributes from:
        # fSiPMTriggerTime -> fSiPMTimestamp
        # fSiPMQDC -> fSiPMPhotonCount
        if self.hasSiPM:
            if {b"SiPMData.fSiPMTriggerTime"}.issubset(self.events[b"SiPMData"].keys()):
                leaves_sipm[0] = b"SiPMData.fSiPMTriggerTime"
                leaves_sipm[1] = b"SiPMData.fSiPMQDC"

        # Exception "MCEnergyPrimary"
        # Older 1-to-1 coupling files names the primary energy "MCEnergy_Primary" while 4-to-1
        # coupling files name it "MCEnergyPrimary
        if {b"MCEnergyPrimary"}.issubset(self.events_keys):
            leaves_global[2] = b"MCEnergyPrimary"

        # Exception: "MCEnergyDeps"
        # Check if MCEnergyDeps_e and MCEnergyDeps_p are stored in the root file
        # These are leaves added later in the analysis, therefore older datasets won't contain them.
        if {b'MCEnergyDeps_e', b'MCEnergyDeps_p'}.issubset(self.events_keys):
            leaves_global += [b'MCEnergyDeps_e', b'MCEnergyDeps_p']

        # finalize leave list based on type string
        if self.hasGlobal:
            list_leaves += leaves_global
        if self.hasCluster:
            list_leaves += leaves_cluster
        if self.hasSiPM:
            list_leaves += leaves_sipm

        return list_leaves

    def iterate_events(self, n=None):
        """
        iteration over the events root tree

        Args:
            n (int or None):    total number of events being returned,
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

        for start, end, basket in self.events.iterate(self.tree_leaves(),
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
        # TODO: This needs to be cleaner

        list_leaves = self.tree_leaves()
        if self.hasCluster:
            event = EventCluster(EventNumber=basket["EventNumber"][idx],
                                 MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                                 MCEnergy_Primary=basket['MCEnergy_Primary'][idx],
                                 MCEnergy_e=basket['MCEnergy_e'][idx],
                                 MCEnergy_p=basket['MCEnergy_p'][idx],
                                 MCPosition_e=basket['MCPosition_e'][idx],
                                 MCInteractions_e=basket['MCInteractions_e'][idx],
                                 MCPosition_p=basket['MCPosition_p'][idx],
                                 MCInteractions_p=basket['MCInteractions_p'][idx],
                                 MCEnergyDeps_e=basket["MCEnergyDeps_e"][
                                     idx] if b"MCEnergyDeps_e" in list_leaves else None,
                                 MCEnergyDeps_p=basket["MCEnergyDeps_p"][
                                     idx] if b"MCEnergyDeps_p" in list_leaves else None,
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

        elif self.hasSiPM:
            event = EventSiPM(EventNumber=basket["EventNumber"][idx],
                              MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                              MCEnergy_Primary=basket['MCEnergyPrimary'][idx],
                              MCEnergy_e=basket['MCEnergy_e'][idx],
                              MCEnergy_p=basket['MCEnergy_p'][idx],
                              MCPosition_e=basket['MCPosition_e'][idx],
                              MCInteractions_e=basket['MCInteractions_e'][idx],
                              MCPosition_p=basket['MCPosition_p'][idx],
                              MCInteractions_p=basket['MCInteractions_p'][idx],
                              MCEnergyDeps_e=basket["MCEnergyDeps_e"][
                                  idx] if b"MCEnergyDeps_e" in list_leaves else None,
                              MCEnergyDeps_p=basket["MCEnergyDeps_p"][
                                  idx] if b"MCEnergyDeps_p" in list_leaves else None,
                              MCPosition_source=basket['MCPosition_source'][idx],
                              MCDirection_source=basket['MCDirection_source'][idx],
                              MCComptonPosition=basket['MCComptonPosition'][idx],
                              MCDirection_scatter=basket['MCDirection_scatter'][idx],
                              SiPM_timestamp=basket[
                                  "SiPMData.fSiPMTimeStamp" if b"SiPMData.fSiPMTimeStamp" in list_leaves else "SiPMData.fSiPMTriggerTime"][
                                  idx],
                              SiPM_photoncount=basket[
                                  "SiPMData.fSiPMPhotonCount" if b"SiPMData.fSiPMPhotonCount" in list_leaves else "SiPMData.fSiPMQDC"][
                                  idx],
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
                          MCEnergyDeps_e=basket["MCEnergyDeps_e"][
                              idx] if b"MCEnergyDeps_e" in list_leaves else None,
                          MCEnergyDeps_p=basket["MCEnergyDeps_p"][
                              idx] if b"MCEnergyDeps_p" in list_leaves else None,
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
        for basket in self.events.iterate(self.tree_leaves(),
                                          entrystart=position,
                                          entrystop=position + 1,
                                          namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)
