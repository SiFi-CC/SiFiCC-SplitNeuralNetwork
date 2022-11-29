import numpy as np


########################################################################################################################

def print_event_summary(Rootdata, n=0):
    """Prints out a summary of one random event"""

    # grab event from root tree
    event = Rootdata.get_event(position=n)

    print("\nPrinting event summary\n")
    print("Event number: ", event.EventNumber)
    print("Event type: {}".format(event.MCSimulatedEventType))
    print("--------------")
    print("EnergyPrimary: {:.3f}".format(event.MCEnergy_Primary))
    print("RealEnergy_e: {:.3f}".format(event.MCEnergy_e))
    print("RealEnergy_p: {:.3f}".format(event.MCEnergy_p))
    print("RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCPosition_source.x,
                                                                    event.MCPosition_source.y,
                                                                    event.MCPosition_source.z))
    print("RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCDirection_source.x,
                                                                     event.MCDirection_source.y,
                                                                     event.MCDirection_source.z))
    print("RealComptonPosition: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCComptonPosition.x,
                                                                    event.MCComptonPosition.y,
                                                                    event.MCComptonPosition.z))
    print("RealDirection_scatter: ({:7.3f}, {:7.3f}, {:7.3f})".format(event.MCDirection_scatter.x,
                                                                      event.MCDirection_scatter.y,
                                                                      event.MCDirection_scatter.z))
    print("\nRealPosition_e / RealInteractions_e:")
    for i in range(len(event.MCInteractions_e)):
        print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_e.x[i],
                                                        event.MCPosition_e.y[i],
                                                        event.MCPosition_e.z[i],
                                                        event.MCInteractions_e[i]))
    print("\nRealPosition_p / RealInteractions_p:")
    for i in range(len(event.MCInteractions_p)):
        print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_p.x[i],
                                                        event.MCPosition_p.y[i],
                                                        event.MCPosition_p.z[i],
                                                        event.MCInteractions_p[i]))

    print("\n Cluster Entries: ")
    print("Energy / Position / Entries / Module")
    for i, cluster in enumerate(event.RecoClusterPosition):
        print("{:.3f} | {} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} ".format(i,
                                                                         event.RecoClusterEnergies_values[i],
                                                                         cluster.x,
                                                                         cluster.y,
                                                                         cluster.z,
                                                                         event.RecoClusterEntries[i]))

    RecoCluster_idx_scatterer, RecoCluster_idx_absorber = event.sort_clusters_by_module(use_energy=True)
    print("\nCluster in Scatterer: {} | Cluster idx: {}".format(len(RecoCluster_idx_scatterer),
                                                                RecoCluster_idx_scatterer))
    print("Cluster in Absorber: {} | Cluster idx: {}".format(len(RecoCluster_idx_absorber),
                                                             RecoCluster_idx_absorber))


def print_detector_summary(Rootdata):
    """print detector dimensions

    """
    scatterer = Rootdata.scatterer
    absorber = Rootdata.absorber
    print("\n##### Scatterer: ")
    print("Position: ({},{},{})".format(scatterer.pos.x, scatterer.pos.y, scatterer.pos.z))
    print("Dimensions: ({:.1f},{:.1f},{:.1f})".format(scatterer.dimx, scatterer.dimy, scatterer.dimz))

    print("\n##### Absorber: ")
    print("Position: ({},{},{})".format(absorber.pos.x, absorber.pos.y, absorber.pos.z))
    print("Dimensions: ({:.1f},{:.1f},{:.1f})".format(absorber.dimx, absorber.dimy, absorber.dimz))


########################################################################################################################

def update_npz_AWALNN(npz_file, root_file):
    """Adds Neural-Network classification data created by AwalNN to an existing npz file

    Args:
        npz_file (str): path to existing npz file
        root_file (root): root file created by AwalNN predictions

    return:
        None
    """

    import uproot

    # load existing npz file
    npz = np.load(npz_file)

    ary_MC = npz["MC_TRUTH"]
    ary_CB = npz["CB_RECO"]
    ary_cluster = npz["CLUSTER_RECO"]
    ary_nn = npz["NN_RECO"]

    root_data = uproot.open(root_file)
    root_data_tree = root_data[b"ConeList"]

    # TODO: optimize problem
    # grab entries from leaves "GlobalEventNumber" and "EventType" and store them in arrays
    ary_GlobalEventNumber = root_data_tree["GlobalEventNumber"].array()
    ary_EventType = root_data_tree["EventType"].array()
    ary_Energy_e = root_data_tree["E1"].array()
    ary_Energy_p = root_data_tree["E2"].array()
    ary_x1 = root_data_tree["x_1"].array()
    ary_y1 = root_data_tree["y_1"].array()
    ary_z1 = root_data_tree["z_1"].array()
    ary_x2 = root_data_tree["x_2"].array()
    ary_y2 = root_data_tree["y_2"].array()
    ary_z2 = root_data_tree["z_2"].array()

    # sort both arrays by GlobalEventNumber
    ary_EventType = ary_EventType[ary_GlobalEventNumber.argsort()]
    ary_Energy_e = ary_Energy_e[ary_GlobalEventNumber.argsort()]
    ary_Energy_p = ary_Energy_p[ary_GlobalEventNumber.argsort()]
    ary_x1 = ary_x1[ary_GlobalEventNumber.argsort()]
    ary_y1 = ary_y1[ary_GlobalEventNumber.argsort()]
    ary_z1 = ary_z1[ary_GlobalEventNumber.argsort()]
    ary_x2 = ary_x2[ary_GlobalEventNumber.argsort()]
    ary_y2 = ary_y2[ary_GlobalEventNumber.argsort()]
    ary_z2 = ary_z2[ary_GlobalEventNumber.argsort()]
    ary_GlobalEventNumber = ary_GlobalEventNumber[ary_GlobalEventNumber.argsort()]

    # iterate the dataframe and scan for matching entries
    # an extremely dumb iteration algorithm, but it works
    cidx = 0
    for i in range(ary_MC.shape[0]):
        # start at first entry of df
        if ary_MC[i, 0] == ary_GlobalEventNumber[cidx]:
            ary_MC[i, 4] = ary_EventType[cidx]

            # add neural network reco data
            ary_nn[i, 3] = ary_EventType[cidx]
            ary_nn[i, 4] = ary_Energy_e[cidx]
            ary_nn[i, 5] = ary_Energy_p[cidx]
            ary_nn[i, 6] = ary_x1[cidx]
            ary_nn[i, 7] = ary_y1[cidx]
            ary_nn[i, 8] = ary_z1[cidx]
            ary_nn[i, 9] = ary_x2[cidx]
            ary_nn[i, 10] = ary_y2[cidx]
            ary_nn[i, 11] = ary_z2[cidx]

            cidx += 1
            # end condition
            if cidx >= len(ary_EventType):
                break
        else:
            continue

    with open(npz_file, 'wb') as file:
        np.savez_compressed(file,
                            MC_TRUTH=ary_MC,
                            CB_RECO=ary_CB,
                            CLUSTER_RECO=ary_cluster,
                            NN_RECO=ary_nn)

    print("file saved: ", npz_file)


def update_npz_CBcorrect(npz_file):
    """Updates the identified tag of an npz file based on the "correctness" given by AwalNN criteria.

    Args:
        npz_file (str): path to existing npz file

    return:
        None
    """

    import uproot

    # load existing npz file
    npz = np.load(npz_file)

    ary_MC = npz["MC_TRUTH"]
    ary_CB = npz["CB_RECO"]
    ary_cluster = npz["CLUSTER_RECO"]
    ary_nn = npz["NN_RECO"]

    for i in range(ary_MC.shape[0]):
        if ary_MC[i, 3] not in [-1, -3, 1, 3]:
            continue

        # energy e
        if abs(ary_CB[i, 4] - ary_MC[i, 6]) > ary_MC[i, 6] * 0.06 * 2:
            continue
        # energy p
        if abs(ary_CB[i, 5] - ary_MC[i, 7]) > ary_MC[i, 7] * 0.06 * 2:
            continue
        # position e
        if abs(ary_CB[i, 6] - ary_MC[i, 20]) > 1.3 * 2:
            continue
        if abs(ary_CB[i, 7] - ary_MC[i, 21]) > 10 * 2:
            continue
        if abs(ary_CB[i, 8] - ary_MC[i, 22]) > 1.3 * 2:
            continue
        # position p
        if abs(ary_CB[i, 9] - ary_MC[i, 23]) > 1.3 * 2:
            continue
        if abs(ary_CB[i, 10] - ary_MC[i, 24]) > 10 * 2:
            continue
        if abs(ary_CB[i, 11] - ary_MC[i, 25]) > 1.3 * 2:
            continue

        # update identified tag
        ary_MC[i, 3] += int(ary_MC[i, 3] / abs(ary_MC[i, 3]) * 10)

    with open(npz_file, 'wb') as file:
        np.savez_compressed(file,
                            MC_TRUTH=ary_MC,
                            CB_RECO=ary_CB,
                            CLUSTER_RECO=ary_cluster,
                            NN_RECO=ary_nn)

    print("file saved: ", npz_file)


"""self.energy_factor_limit = .06 * 2
self.position_absolute_limit = np.array([1.3, 10, 1.3]) * 2"""
