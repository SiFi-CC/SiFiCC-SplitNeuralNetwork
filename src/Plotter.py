import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


def plot_score_dist(y_scores, y_true, figure_name):
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
    plt.savefig(figure_name + ".png")


def plot_history_classifier(nn_classifier, figure_name):
    # plot model performance
    loss = nn_classifier.history['loss']
    val_loss = nn_classifier.history['val_loss']
    mse = nn_classifier.history["accuracy"]
    val_mse = nn_classifier.history["val_accuracy"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_history_regression(nn_regression, figure_name):
    loss = nn_regression.history['loss']
    val_loss = nn_regression.history['val_loss']
    mse = nn_regression.history["mean_absolute_error"]
    val_mse = nn_regression.history["val_mean_absolute_error"]

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.plot(mse, label="Training", linestyle='--', color="blue")
    ax1.plot(val_mse, label="Validation", linestyle='-', color="red")
    ax1.set_ylabel("mean_absolute_error")
    ax1.legend()
    ax1.grid()

    ax2.plot(loss, label="Training", linestyle='--', color="blue")
    ax2.plot(val_loss, label="Validation", linestyle='-', color="red")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_regression_energy_error(y_pred, y_true, figure_name):
    bins = np.arange(-2.0, 2.0, 0.05)

    plt.figure()
    plt.title("Error Energy Electron")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins, histtype=u"step", color="blue")
    plt.savefig(figure_name + "_electron.png")

    plt.figure()
    plt.title("Error Energy Photon")
    plt.xlabel(r"$E_{Pred}$ - $E_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins, histtype=u"step", color="blue")
    plt.savefig(figure_name + "_photon.png")

    e_diff = []
    p_diff = []
    for i in range(y_pred.shape[0]):
        e_diff.append(y_pred[i, 0] - y_true[i, 0])
        p_diff.append(y_pred[i, 1] - y_true[i, 1])
    print(min(e_diff), max(e_diff))
    print(min(p_diff), max(p_diff))

    print("Electron Energy Resolution: {:.3f} MeV".format(np.std(e_diff)))
    print("Photon Energy Resolution: {:.3f} MeV".format(np.std(p_diff)))


def plot_regression_position_error(y_pred, y_true, figure_name):
    bins_x = np.arange(-20.5, 20.5, 0.5)
    bins_y = np.arange(-60.5, 60.5, 0.5)
    bins_z = np.arange(-20.5, 20.5, 0.5)

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$x_{Pred}$ - $x_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 0] - y_true[:, 0], bins=bins_x, histtype=u"step", color="blue", label=r"e_x")
    plt.hist(y_pred[:, 3] - y_true[:, 3], bins=bins_x, histtype=u"step", color="orange", label=r"p_x")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_x.png")

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$y_{Pred}$ - $y_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 1] - y_true[:, 1], bins=bins_y, histtype=u"step", color="blue", label=r"e_y")
    plt.hist(y_pred[:, 4] - y_true[:, 4], bins=bins_y, histtype=u"step", color="orange", label=r"p_y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_y.png")

    plt.figure()
    plt.title("Error Position x")
    plt.xlabel(r"$z_{Pred}$ - $z_{True}$")
    plt.ylabel("counts")
    plt.hist(y_pred[:, 2] - y_true[:, 2], bins=bins_z, histtype=u"step", color="blue", label=r"e_z")
    plt.hist(y_pred[:, 5] - y_true[:, 5], bins=bins_z, histtype=u"step", color="orange", label=r"p_z")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + "_z.png")


def plot_source_position(ary_sp_pos, ary_sp_tp, ary_sp_tot, figure_name):
    # plot MC Source Position z-direction
    bins = np.arange(-80, 20, 1.0)
    width = abs(bins[0] - bins[1])
    hist1, _ = np.histogram(ary_sp_pos, bins=bins)
    hist2, _ = np.histogram(ary_sp_tp, bins=bins)
    hist3, _ = np.histogram(ary_sp_tot, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(ary_sp_tot, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist3, np.sqrt(hist3), color="orange", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black", fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red", fmt=".", label="NN positive\nevents")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_primary_energy_dist(ary_pe_pos, ary_pe_tp, ary_pe_tot, figure_name):
    # plot MC Source Position z-direction
    bins = np.arange(0.0, 10.0, 0.1)
    width = abs(bins[0] - bins[1])
    hist1, _ = np.histogram(ary_pe_pos, bins=bins)
    hist2, _ = np.histogram(ary_pe_tp, bins=bins)
    hist3, _ = np.histogram(ary_pe_tot, bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy Primary")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    plt.xlim(0.0, 10.0)
    # total event histogram
    plt.hist(ary_pe_tot, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist3, np.sqrt(hist3), color="orange", fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black", fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red", fmt=".", label="NN positive\nevents")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_2dhist_score_sourcepos(ary_score, ary_sp, figure_name):
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_sp = np.arange(-80.0, 20.0, 1.0)

    list_score = []
    list_sp = []
    for i in range(len(ary_score)):
        if not ary_sp[i] == 0.0:
            list_score.append(ary_score[i])
            list_sp.append(ary_sp[i])

    plt.figure()
    plt.xlabel("MC Source Position z [mm]")
    plt.ylabel("Score")
    plt.hist2d(list_sp, list_score, bins=[bin_sp, bin_score])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_2dhist_score_eprimary(ary_score, ary_ep, figure_name):
    bin_score = np.arange(0.0, 1.0, 0.05)
    bin_sp = np.arange(0.0, 16.0, 0.1)

    list_score = []
    list_sp = []
    for i in range(len(ary_score)):
        if not ary_ep[i] == 0.0:
            list_score.append(ary_score[i])
            list_sp.append(ary_ep[i])

    plt.figure()
    plt.xlabel("MC Primary Energy [MeV]")
    plt.ylabel("Score")
    h0 = plt.hist2d(list_sp, list_score, bins=[bin_sp, bin_score], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_2dhist_score_regE_error(ary_score, ary_regE_err, figure_name):
    bin_score = np.arange(0.4, 1.0, 0.05)
    bin_sp = np.arange(-2.0, 2.0, 0.05)

    ary_score = np.reshape(ary_score, (len(ary_score),))
    ary_regE_err = np.reshape(ary_regE_err, (len(ary_regE_err),))

    plt.figure()
    plt.xlabel(r" $E_{Reco} - E_{True}$ [MeV]")
    plt.ylabel("Score")
    h0 = plt.hist2d(ary_regE_err, ary_score, bins=[bin_sp, bin_score], norm=LogNorm())
    plt.colorbar(h0[3])
    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_sourceposition_heatmap(ary_sp_z, ary_sp_y, figure_name):
    bins_z = np.arange(-80, 20, 1.0)
    bins_y = np.arange(-20, 20, 1.0)

    hist0, _, _ = np.histogram2d(x=ary_sp_z, y=ary_sp_y, bins=[bins_z, bins_y])
    hist1, _ = np.histogram(ary_sp_z, bins=bins_z)

    # normalize histogram to range [0.0, 1.0]
    min_val, max_val = np.min(hist0), np.max(hist0)
    hmap = (hist0 - min_val) / (max_val - min_val + 1e-10)

    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 8))
    axs[0].set_xlabel("MC Source Position z [mm]")
    axs[0].set_ylabel("MC Source Position y [mm]")
    im0 = axs[0].imshow(hmap)

    axs[1].set_xlabel("MC Source Position z [mm]")
    axs[1].set_ylabel("Counts")
    axs[1].hist(ary_sp_z, bins=bins_z, histtype=u"step", color="black", linestyle="--", alpha=0.7)
    axs[1].errorbar(bins_z[:-1] - 0.5, hist1, np.sqrt(hist1), fmt=".", color="black")

    # color-bars
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')

    plt.tight_layout()
    plt.savefig(figure_name + ".png")


def plot_backprojection(image, figure_name):
    plt.figure()
    plt.imshow(image)
    # plt.savefig(figure_name + ".png")
    plt.show()
