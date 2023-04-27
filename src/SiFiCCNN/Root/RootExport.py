import numpy as np
import uproot

from src.SiFiCCNN import Utility


# ----------------------------------------------------------------------------------------------------------------------

def export_classicreco(RootParser,
                       destination):
    # create empty array for classical cut-based reconstruction
    ary_cb = np.zeros(shape=(RootParser.events_entries, 10))

    # fill up Cut-Based reconstruction values manually due to them being stored in branches
    ary_cb[:, 0] = RootParser.events["Identified"].array()
    ary_cb[:, 1] = RootParser.events["RecoEnergy_e"]["value"].array()
    ary_cb[:, 2] = RootParser.events["RecoEnergy_p"]["value"].array()
    ary_cb[:, 3] = RootParser.events["RecoPosition_e"]["position"].array().x
    ary_cb[:, 4] = RootParser.events["RecoPosition_e"]["position"].array().y
    ary_cb[:, 5] = RootParser.events["RecoPosition_e"]["position"].array().z
    ary_cb[:, 6] = RootParser.events["RecoPosition_p"]["position"].array().x
    ary_cb[:, 7] = RootParser.events["RecoPosition_p"]["position"].array().y
    ary_cb[:, 8] = RootParser.events["RecoPosition_p"]["position"].array().z
    # add compton scattering angle calculated from energy
    e = RootParser.events["RecoEnergy_e"]["value"].array()
    p = RootParser.events["RecoEnergy_p"]["value"].array()
    for i in range(len(e)):
        ary_cb[i, 9] = Utility.get_scattering_angle_energy(e[i], p[i])

    # export dataframe to compressed .npz

    with open(destination + "/" + RootParser.file_name + "_CBRECO.npz", 'wb') as file:
        np.savez_compressed(file, CB_RECO=ary_cb)

    print("file saved at: ", destination + "/" + RootParser.file_name + "_CBRECO.npz")
