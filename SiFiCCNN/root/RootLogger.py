def print_event_summary(Rootdata, n=0):
    """Prints out a summary of one event"""

    # grab event from root tree
    event = Rootdata.get_event(position=n)

    print("\n### Printing event summary\n")
    print("\n# Global Event data:")
    print("Event number: ", event.EventNumber)
    print("Event type: {}".format(event.MCSimulatedEventType))
    print("--------------")
    print("EnergyPrimary: {:.3f}".format(event.MCEnergy_Primary))
    print("RealEnergy_e: {:.3f}".format(event.MCEnergy_e))
    print("RealEnergy_p: {:.3f}".format(event.MCEnergy_p))
    print("RealPosition_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(
        event.MCPosition_source.x,
        event.MCPosition_source.y,
        event.MCPosition_source.z))
    print("RealDirection_source: ({:7.3f}, {:7.3f}, {:7.3f})".format(
        event.MCDirection_source.x,
        event.MCDirection_source.y,
        event.MCDirection_source.z))
    print("RealComptonPosition: ({:7.3f}, {:7.3f}, {:7.3f})".format(
        event.MCComptonPosition.x,
        event.MCComptonPosition.y,
        event.MCComptonPosition.z))
    print("RealDirection_scatter: ({:7.3f}, {:7.3f}, {:7.3f})".format(
        event.MCDirection_scatter.x,
        event.MCDirection_scatter.y,
        event.MCDirection_scatter.z))
    print("\nRealPosition_e / RealInteractions_e:")
    for i in range(len(event.MCInteractions_e)):
        print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_e.x[i],
                                                        event.MCPosition_e.y[i],
                                                        event.MCPosition_e.z[i],
                                                        event.MCInteractions_e[
                                                            i]))
    print("\nRealPosition_p / RealInteractions_p:")
    for i in range(len(event.MCInteractions_p)):
        print("({:7.3f}, {:7.3f}, {:7.3f}) | {}".format(event.MCPosition_p.x[i],
                                                        event.MCPosition_p.y[i],
                                                        event.MCPosition_p.z[i],
                                                        event.MCInteractions_p[
                                                            i]))

    if event.breco:
        print("\n# Reconstruction data: ")
        print("Cluster Entries: ")
        print("Energy / Position / Entries / Module")
        for i, cluster in enumerate(event.RecoClusterPosition):
            print(
                "{:.3f} | {:.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:3} | {:7.5}".format(
                    i,
                    event.RecoClusterEnergies_values[
                        i],
                    cluster.x,
                    cluster.y,
                    cluster.z,
                    event.RecoClusterEntries[i],
                    event.RecoClusterTimestamps_relative[
                        i]))

        RecoCluster_idx_scatterer, RecoCluster_idx_absorber = event.sort_clusters_by_module(
            use_energy=True)
        print("\nCluster in Scatterer: {} | Cluster idx: {}".format(
            len(RecoCluster_idx_scatterer),
            RecoCluster_idx_scatterer))
        print("Cluster in Absorber: {} | Cluster idx: {}".format(
            len(RecoCluster_idx_absorber),
            RecoCluster_idx_absorber))

    if event.bsipm:
        print("\n# Fibre Data: ")
        print("ID / Energy / Position / TriggerTime")
        for j in range(len(event.fibre_id)):
            print(
                "{:3.3f} | {:5.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:7.5}".format(
                    event.fibre_id[j],
                    event.fibre_energy[j],
                    event.fibre_position[j].x,
                    event.fibre_position[j].y,
                    event.fibre_position[j].z,
                    event.fibre_time[j]))

        print("\n# SiPM Data: ")
        print("ID / QDC / Position / TriggerTime")
        for j in range(len(event.SiPM_id)):
            print(
                "{:3.3f} | {:5.3f} | ({:7.3f}, {:7.3f}, {:7.3f}) | {:7.5}".format(
                    event.SiPM_id[j],
                    event.SiPM_qdc[j],
                    event.SiPM_position[j].x,
                    event.SiPM_position[j].y,
                    event.SiPM_position[j].z,
                    event.SiPM_triggertime[j]))


def print_detector_summary(Rootdata):
    """print detector dimensions

    """
    scatterer = Rootdata.scatterer
    absorber = Rootdata.absorber
    print("\n##### Scatterer: ")
    print("Position: ({},{},{})".format(scatterer.pos.x, scatterer.pos.y,
                                        scatterer.pos.z))
    print("Dimensions: ({:.1f},{:.1f},{:.1f})".format(scatterer.dimx,
                                                      scatterer.dimy,
                                                      scatterer.dimz))

    print("\n##### Absorber: ")
    print("Position: ({},{},{})".format(absorber.pos.x, absorber.pos.y,
                                        absorber.pos.z))
    print("Dimensions: ({:.1f},{:.1f},{:.1f})".format(absorber.dimx,
                                                      absorber.dimy,
                                                      absorber.dimz))
