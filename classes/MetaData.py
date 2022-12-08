import numpy as np


class MetaData:

    def __init__(self, npz_file):
        # load npz file and extract all meta data features
        npz_data = np.load(npz_file)

        self.meta = npz_data["META"]
        self.mc_true = npz_data["MC_TRUTH"]
        # cluster reco data is disabled as it is rather large and rarely used
        # self.cluster_reco = npz_data["CLUSTER_RECO"]
        self.cb_reco = npz_data["CB_RECO"]

    def event_number(self):
        return self.meta[:, 0]

    def simulated_event_type(self):
        return self.meta[:, 1]

    def cb_identified(self):
        return self.meta[:, 3]

    def mc_primary_energy(self):
        return self.mc_true[:, 0]

    def mc_source_position_z(self):
        return self.mc_true[:, 5]
