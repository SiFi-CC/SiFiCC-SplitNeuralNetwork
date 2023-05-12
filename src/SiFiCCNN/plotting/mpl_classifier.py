import numpy as np
import matplotlib.pyplot as plt


def score_distribution(y_scores, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if
                      y_true[i] == 0]

    plt.figure(figsize=(8, 6))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Signal score")
    plt.ylabel("counts")
    plt.hist(np.array(ary_scores_pos), bins=bins, color="orange",
             label="true positives", alpha=0.25)
    plt.hist(np.array(ary_scores_neg), bins=bins, color="blue",
             label="true negatives", alpha=0.25)
    h0, _, _ = plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step",
                        color="orange")
    h1, _, _ = plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step",
                        color="blue")
    plt.vlines(x=0.5, ymin=0.0, ymax=max([max(h0), max(h1)]), color="red",
               label="Decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def efficiencymap(y_pred, y_true, y_sp, figure_name, theta=0.5, sr=100):
    # determine contribution from each prediction
    # Done by giving correct prediction weight 1, else weight 0
    ary_w = np.zeros(shape=(len(y_pred),))
    for i in range(len(ary_w)):
        if y_true[i] == 1:
            if y_pred[i] > theta:
                ary_w[i] = 1

    # determine binning from source position array
    bin_min = int(min(y_sp))
    bin_max = int(max(y_sp))
    ary_bin = np.linspace(bin_min, bin_max, sr)
    width = abs(ary_bin[1] - ary_bin[0])
    ary_bin_err = np.ones(shape=(len(ary_bin) - 1,)) * width / 2

    # Create histogram from prediction and truth
    hist0, _ = np.histogram(y_sp, bins=ary_bin, weights=y_true)
    hist1, _ = np.histogram(y_sp, bins=ary_bin, weights=ary_w)

    # determine efficiency from histogram
    ary_eff = np.zeros(shape=(len(hist0),))
    ary_eff_err = np.zeros(shape=(len(hist0),))
    for i in range(len(ary_eff)):
        if hist1[i] < 10:
            continue
        else:
            ary_eff[i] = hist1[i] / hist0[i]
            ary_eff_err[i] = (np.sqrt((np.sqrt(hist1[i]) / hist0[i]) ** 2 +
                                      (hist1[i] * np.sqrt(hist0[i]) / hist0[
                                          i] ** 2) ** 2))

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Source position efficiency")
    axs[0].set_ylabel("Counts")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=y_true,
                color="black", alpha=0.5, label="Truth")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=ary_w,
                color="red", alpha=0.5, linestyle="--",
                label="Prediction")
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("Source Position z-axis [mm]")
    axs[1].set_ylabel("Efficiency")
    axs[1].errorbar(ary_bin[:-1] + width, ary_eff, ary_eff_err, ary_bin_err,
                    fmt=".", color="blue")
    # axs[1].plot(bins[:-1] + 0.5, ary_eff, ".", color="darkblue")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def roc_curve(list_fpr,
              list_tpr,
              figure_name,
              weighted=False):
    if weighted:
        auc_label = "weightedAUC"
    else:
        auc_label = "AUC"

    # calc area under ROC
    auc_score = 0
    for i in range(len(list_fpr) - 1):
        # Riemann-sum to calculate area under curve
        area = (list_fpr[i + 1] - list_fpr[i]) * list_tpr[i]
        # multiply result by -1, since x values are ordered from highest to
        # lowest
        auc_score += area * (-1)

    print("Plotting ROC curve and {}...".format(auc_label))
    plt.figure()
    plt.title("ROC Curve | " + auc_label)
    plt.plot(list_fpr, list_tpr, color="red",
             label="{0:}: {1:.3f}".format(auc_label, auc_score))
    plt.plot([0, 1], [0, 1], color="black", ls="--")
    # plt.plot(dot[0], dot[1], 'b+')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def sp_distribution(ary_sp,
                    ary_score,
                    ary_true,
                    figure_name):
    # plot MC Source Position z-direction
    width = 1.0
    bins = np.arange(int(min(ary_sp)), int(max(ary_sp)), width)

    idx_tp = []
    for i in range(len(ary_sp)):
        if ary_score[i] > 0.5 and ary_true[i] == 1.0:
            idx_tp.append(True)
        else:
            idx_tp.append(False)
    idx_tp = np.array(idx_tp)

    hist0, _ = np.histogram(ary_sp[ary_sp != 0.0], bins=bins)
    hist1, _ = np.histogram(ary_sp[idx_tp], bins=bins)
    hist2, _ = np.histogram(ary_sp[ary_true == 1.0], bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Source Position z")
    plt.xlabel("z-position [mm]")
    plt.ylabel("counts")
    plt.xlim(-80.0, 20.0)
    # total event histogram
    plt.hist(ary_sp[ary_sp != 0.0], bins=bins, color="orange", alpha=0.5,
             label="All events")
    plt.errorbar(bins[1:] - width / 2, hist0, np.sqrt(hist0), color="orange",
                 fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black",
                 fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red",
                 fmt=".", label="True Positive events")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_pe_distribution(ary_pe,
                         ary_score,
                         ary_true,
                         figure_name):
    # plot MC Source Position z-direction
    width = 0.1
    bins = np.arange(0.0, 10.0, width)

    idx_tp = []
    for i in range(len(ary_pe)):
        if ary_score[i] > 0.5 and ary_true[i] == 1.0:
            idx_tp.append(True)
        else:
            idx_tp.append(False)
    idx_tp = np.array(idx_tp)

    hist0, _ = np.histogram(ary_pe, bins=bins)
    hist1, _ = np.histogram(ary_pe[idx_tp], bins=bins)
    hist2, _ = np.histogram(ary_pe[ary_true == 1.0], bins=bins)

    # generate plots
    # MC Total / MC Ideal Compton
    plt.figure()
    plt.title("MC Energy Primary")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("counts (normalized)")
    plt.xlim(0.0, 10.0)
    # total event histogram
    plt.hist(ary_pe, bins=bins, color="orange", alpha=0.5, label="All events")
    plt.errorbar(bins[1:] - width / 2, hist0, np.sqrt(hist0), color="orange",
                 fmt=".")
    plt.errorbar(bins[1:] - width / 2, hist2, np.sqrt(hist2), color="black",
                 fmt=".", label="Ideal Compton events")
    plt.errorbar(bins[1:] - width / 2, hist1, np.sqrt(hist1), color="red",
                 fmt=".", label="True Positive events")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()
