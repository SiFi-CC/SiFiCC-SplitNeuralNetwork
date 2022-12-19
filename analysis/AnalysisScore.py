def analysis(SiFiCCNN, DataCluster, MetaData=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from fastROCAUC import fastROCAUC

    # predict test set
    y_scores = SiFiCCNN.predict(DataCluster.x_test())
    y_scores = np.reshape(y_scores, newshape=(len(y_scores),))
    y_true = DataCluster.y_test()

    # check directory for results
    dir_main = os.getcwd()
    dir_results = dir_main + "/results/" + SiFiCCNN.model_name + SiFiCCNN.model_tag
    if not os.path.isdir(dir_results):
        os.mkdir(dir_results)

    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 0]

    plt.figure()
    plt.xlabel("score")
    plt.ylabel("counts")
    plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step", color="orange", label="true positives")
    plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step", color="blue", label="true negatives")
    # plt.vlines(x=theta, ymin=0.0, ymax=len(ary_scores_neg)/2, color="red", linestyles="--", label="optimal threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir_results + "/score_dist.png")

    # score, primary energy 2d histogram
    ary_test_mcenergyprimary = MetaData.mc_primary_energy()[DataCluster.idx_test()]
    bins_energy = np.arange(0.0, 16.0, 0.1)
    plt.figure()
    plt.xlabel("score")
    plt.ylabel("MC Primary Energy [MeV]")
    h0 = plt.hist2d(np.array(y_scores), ary_test_mcenergyprimary, bins=[bins, bins_energy], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(dir_results + "/score_energyprimary.png")

    # score, primary energy 2d histogram
    ary_test_mcsourceposz = MetaData.mc_source_position_z()[DataCluster.idx_test()]
    bins_sourcepos = np.arange(-80, 20, 1.0)
    plt.figure()
    plt.xlabel("score")
    plt.ylabel("MC Source position z-axis [mm]")
    h0 = plt.hist2d(np.array(y_scores), ary_test_mcsourceposz, bins=[bins, bins_sourcepos], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(dir_results + "/score_sourceposz.png")
