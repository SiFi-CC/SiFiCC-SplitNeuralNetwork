import numpy as np
import uproot
import tqdm
import sys

from classes import Event
from classes import Detector


########################################################################################################################

class RootParser:
    """loading and preprocessing of root data for events and setup tree.

    Attributes:
        rootfile_path (str): path to root file
        events (TTree): root tree containing event data
        setup (TTree): root tree containing detector dimensions
        events_entries (int): number of entries (events) inside the event tree
        scatterer (obj:SIFICC_Module): object containing scatterer dimensions
        absorber (obj:SIFICC_Module): object containing absorber dimensions
        leaves (list): list containing all root tree leaves that are loaded

    """

    def __init__(self, rootfile_path):
        # open root file with uproot
        rootfile = uproot.open(rootfile_path)
        self.rootfile_path = rootfile_path
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

        # list of root tree leaves used in this analysis
        self.leaves = ["EventNumber",
                       'MCEnergy_Primary',
                       'MCEnergy_e',
                       'MCEnergy_p',
                       'MCPosition_source',
                       'MCSimulatedEventType',
                       'MCDirection_source',
                       'MCComptonPosition',
                       'MCDirection_scatter',
                       'MCPosition_e',
                       'MCInteractions_e',
                       'MCPosition_p',
                       'MCInteractions_p',
                       'Identified',
                       'RecoClusterPositions.position',
                       'RecoClusterPositions.uncertainty',
                       'RecoClusterEnergies.value',
                       'RecoClusterEnergies.uncertainty',
                       'RecoClusterEntries']

    def iterate_events(self, n=None):
        """iteration over the events root tree

        Args:
            n (int) or (None): total number of events being returned, if None the maximum number
                               will be iterated.

        Returns:
            yield event at every root tree entry

        """
        # evaluate parameter n
        if n is None:
            n = self.events_entries
        # TODO: exception for negative entries

        # define progress bar
        progbar = tqdm.tqdm(total=n, ncols=100, file=sys.stdout, desc="iterating root tree")
        progbar_step = 0
        progbar_update_size = 1000

        for start, end, basket in self.events.iterate(self.leaves, entrysteps=100000,
                                                      reportentries=True, namedecode='utf-8',
                                                      entrystart=0, entrystop=n):
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
        """create event object from a given position in a root basket

        Args:
            basket (obj: ???): root basket
            idx (int): position inside the root basket

        Return:
            event (obj: Event)

        """
        event = Event(EventNumber=basket["EventNumber"][idx],
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
                      RecoClusterPosition_uncertainty=basket['RecoClusterPositions.uncertainty'][idx],
                      RecoClusterEnergies_values=basket['RecoClusterEnergies.value'][idx],
                      RecoClusterEnergies_uncertainty=basket['RecoClusterEnergies.uncertainty'][idx],
                      RecoClusterEntries=basket['RecoClusterEntries'][idx],
                      MCSimulatedEventType=basket['MCSimulatedEventType'][idx],
                      scatterer=self.scatterer,
                      absorber=self.absorber)
        return event

    def get_event(self, position):
        """Return event for a given position in the root file"""
        for basket in self.events.iterate(self.leaves, entrystart=position, entrystop=position + 1,
                                          namedecode='utf-8'):
            return self.__event_at_basket(basket, 0)

    ####################################################################################################################

    def export_npz(self, npz_filename, n=None):
        """generates compressed npz file containing MC-Truth data, cluster data and Cut-based reco data.

        Args:
            npz_filename (str): filename of the generated .npz file
            n (int or none): number of events parsed from root tree, None if all events are iterated

        """

        # create empty arrays for full export to compressed .npz format
        # root data will be split into:
        # - Meta data: (EventNumber, MCSimulatedEventType, IdealCompton event tag, CB-identified)
        # - MonteCarlo-data: Monte-Carlo Event data
        # - CutBased-data: Cut-based reconstruction data
        # - Cluster-data: Cluster data

        ary_meta = np.zeros(shape=(self.events_entries, 4))
        ary_mc = np.zeros(shape=(self.events_entries, 21))
        ary_cb = np.zeros(shape=(self.events_entries, 23))

        # temporary a list, converted to array later
        ary_cluster = np.empty(shape=(self.events_entries,), dtype=object)

        # Fill Meta-data, Monte-Carlo data and Cluster data into empty arrays
        # Cut-based reco data is not iterable since uproot can't handle the reco data stored in branches
        for i, event in enumerate(self.iterate_events(n=n)):
            ary_meta[i, :] = [event.EventNumber,
                              event.MCSimulatedEventType,
                              event.is_ideal_compton * 1,
                              event.Identified]

            ary_mc[i, :] = [event.MCEnergy_Primary,
                            event.MCEnergy_e,
                            event.MCEnergy_p,
                            event.MCPosition_source.x,
                            event.MCPosition_source.y,
                            event.MCPosition_source.z,
                            event.MCDirection_source.x,
                            event.MCDirection_source.y,
                            event.MCDirection_source.z,
                            event.MCComptonPosition.x,
                            event.MCComptonPosition.y,
                            event.MCComptonPosition.z,
                            event.MCDirection_scatter.x,
                            event.MCDirection_scatter.y,
                            event.MCDirection_scatter.z,
                            event.MCPosition_e_first.x,
                            event.MCPosition_e_first.y,
                            event.MCPosition_e_first.z,
                            event.MCPosition_p_first.x,
                            event.MCPosition_p_first.y,
                            event.MCPosition_p_first.z]

            ary_temp = np.zeros(shape=(len(event.RecoClusterEntries), 9))
            for j in range(len(event.RecoClusterEntries)):
                ary_temp[j, :] = [event.RecoClusterEnergies_values[j],
                                  event.RecoClusterPosition[j].x,
                                  event.RecoClusterPosition[j].y,
                                  event.RecoClusterPosition[j].z,
                                  event.RecoClusterEntries[j],
                                  event.RecoClusterEnergies_uncertainty[j],
                                  event.RecoClusterPosition_uncertainty[j].x,
                                  event.RecoClusterPosition_uncertainty[j].y,
                                  event.RecoClusterPosition_uncertainty[j].z]
            ary_cluster[i] = ary_temp

        # fill up Cut-Based reconstruction values manually due to them being stored in branches
        # TODO: find a better way to do this, why uproot????
        ary_cb[:, 1] = self.events["RecoEnergy_e"]["value"].array()
        ary_cb[:, 2] = self.events["RecoEnergy_p"]["value"].array()
        ary_cb[:, 3] = self.events["RecoPosition_e"]["position"].array().x
        ary_cb[:, 4] = self.events["RecoPosition_e"]["position"].array().y
        ary_cb[:, 5] = self.events["RecoPosition_e"]["position"].array().z
        ary_cb[:, 6] = self.events["RecoPosition_p"]["position"].array().x
        ary_cb[:, 7] = self.events["RecoPosition_p"]["position"].array().y
        ary_cb[:, 8] = self.events["RecoPosition_p"]["position"].array().z
        ary_cb[:, 9] = self.events["RecoDirection_scatter"]["position"].array().x
        ary_cb[:, 10] = self.events["RecoDirection_scatter"]["position"].array().y
        ary_cb[:, 11] = self.events["RecoDirection_scatter"]["position"].array().z

        ary_cb[:, 12] = self.events["RecoEnergy_e"]["uncertainty"].array()
        ary_cb[:, 13] = self.events["RecoEnergy_p"]["uncertainty"].array()
        ary_cb[:, 14] = self.events["RecoPosition_e"]["uncertainty"].array().x
        ary_cb[:, 15] = self.events["RecoPosition_e"]["uncertainty"].array().y
        ary_cb[:, 16] = self.events["RecoPosition_e"]["uncertainty"].array().z
        ary_cb[:, 17] = self.events["RecoPosition_p"]["uncertainty"].array().x
        ary_cb[:, 18] = self.events["RecoPosition_p"]["uncertainty"].array().y
        ary_cb[:, 19] = self.events["RecoPosition_p"]["uncertainty"].array().z
        ary_cb[:, 20] = self.events["RecoDirection_scatter"]["uncertainty"].array().x
        ary_cb[:, 21] = self.events["RecoDirection_scatter"]["uncertainty"].array().y
        ary_cb[:, 22] = self.events["RecoDirection_scatter"]["uncertainty"].array().z

        # TODO: fill reco cluster information

        # export dataframe to compressed .npz
        with open(npz_filename, 'wb') as file:
            np.savez_compressed(file,
                                META=ary_meta,
                                MC_TRUTH=ary_mc,
                                CB_RECO=ary_cb,
                                CLUSTER_RECO=ary_cluster)

        print("file saved: ", npz_filename)
