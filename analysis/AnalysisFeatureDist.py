def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from fastROCAUC import fastROCAUC

    # save results in txt file
    dir_main = os.getcwd()
    dir_results = dir_main + "/results/" + SiFiCCNN.model_name + SiFiCCNN.model_tag
    if not os.path.isdir(dir_results):
        os.mkdir(dir_results)

    # predict test set
    y_pred = SiFiCCNN.predict(DataCluster.x_test())
    y_true = DataCluster.y_test()

    # run ROC curve and AUC score analysis
    auc, theta = fastROCAUC(y_pred, y_true, return_score=True)

    # best optimal threshold
    threshold = theta

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_pred[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    # collect MC-Truth data for predicted test events
    ary_test_mcsourceposz = MetaData.mc_source_position_z()[DataCluster.idx_test()]
    ary_test_mcenergye = MetaData.mc_energy_e()[DataCluster.idx_test()]
    ary_test_mcenergyp = MetaData.mc_energy_p()[DataCluster.idx_test()]
    ary_test_mcenergyprimary = MetaData.mc_primary_energy()[DataCluster.idx_test()]

    # plot MC Source Position z-direction
    bins = np.arange(-80, 20, 1.0)
    width = abs(bins[0] - bins[1])
    ary1 = [ary_test_mcsourceposz[i] for i in range(len(y_pred)) if y_pred[i] == 1 and ary_test_mcsourceposz[i] != 0.0]
    ary2 = MetaData.mc_source_position_z()[MetaData.meta[:, 2] == 1]
    hist1, _ = np.histogram(ary1, bins=bins)
    hist2, _ = np.histogram(ary2, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary2, bins=bins, histtype=u"step", color="black", label="Total Ideal Compton", density=True, alpha=0.5,
             linestyle="--")
    plt.hist(ary1, bins=bins, histtype=u"step", color="red", label="NN positives", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
                 np.sqrt(hist2) / np.sum(hist2) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
                 np.sqrt(hist1) / np.sum(hist1) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_results + "/mc_sourceposz.png")

    ####################################################################################################################

    # plot MC Energy e
    bins = np.arange(0.0, 12.0, 0.1)
    width = abs(bins[0] - bins[1])
    ary1 = [ary_test_mcenergye[i] for i in range(len(y_pred)) if y_pred[i] == 1 and ary_test_mcenergye[i] != 0.0]
    ary2 = MetaData.mc_energy_e()[MetaData.meta[:, 2] == 1]
    hist1, _ = np.histogram(ary1, bins=bins)
    hist2, _ = np.histogram(ary2, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy electron")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary2, bins=bins, histtype=u"step", color="black", label="Total Ideal Compton", density=True, alpha=0.5,
             linestyle="--")
    plt.hist(ary1, bins=bins, histtype=u"step", color="red", label="NN positives", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
                 np.sqrt(hist2) / np.sum(hist2) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
                 np.sqrt(hist1) / np.sum(hist1) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_results + "/mc_energye.png")

    ####################################################################################################################

    # plot MC Energy e
    bins = np.arange(0.0, 12.0, 0.1)
    width = abs(bins[0] - bins[1])
    ary1 = [ary_test_mcenergyp[i] for i in range(len(y_pred)) if y_pred[i] == 1 and ary_test_mcenergyp[i] != 0.0]
    ary2 = MetaData.mc_energy_p()[MetaData.meta[:, 2] == 1]
    hist1, _ = np.histogram(ary1, bins=bins)
    hist2, _ = np.histogram(ary2, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy photon")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary2, bins=bins, histtype=u"step", color="black", label="Total Ideal Compton", density=True, alpha=0.5,
             linestyle="--")
    plt.hist(ary1, bins=bins, histtype=u"step", color="red", label="NN positives", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
                 np.sqrt(hist2) / np.sum(hist2) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
                 np.sqrt(hist1) / np.sum(hist1) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_results + "/mc_energyp.png")

    ####################################################################################################################

    # plot MC Energy Primary
    bins = np.arange(0.0, 16.0, 0.1)
    width = abs(bins[0] - bins[1])
    ary1 = [ary_test_mcenergyprimary[i] for i in range(len(y_pred)) if
            y_pred[i] == 1 and ary_test_mcenergyprimary[i] != 0.0]
    ary2 = MetaData.mc_primary_energy()[MetaData.meta[:, 2] == 1]
    hist1, _ = np.histogram(ary1, bins=bins)
    hist2, _ = np.histogram(ary2, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy Primary")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    # total event histogram
    plt.hist(ary2, bins=bins, histtype=u"step", color="black", label="Total Ideal Compton", density=True, alpha=0.5,
             linestyle="--")
    plt.hist(ary1, bins=bins, histtype=u"step", color="red", label="NN positives", density=True, alpha=0.5,
             linestyle="--")
    plt.errorbar(bins[1:] - width / 2, hist2 / np.sum(hist2) / width,
                 np.sqrt(hist2) / np.sum(hist2) / width, color="black", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist1 / np.sum(hist1) / width,
                 np.sqrt(hist1) / np.sum(hist1) / width, color="red", fmt=".")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir_results + "/mc_energyprimary.png")
