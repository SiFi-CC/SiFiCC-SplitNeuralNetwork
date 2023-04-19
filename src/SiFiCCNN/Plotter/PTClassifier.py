import numpy as np
import matplotlib.pyplot as plt


def plot_score_distribution(y_scores, y_true, figure_name):
    plt.rcParams.update({'font.size': 16})

    # score distribution plot
    bins = np.arange(0.0, 1.0 + 0.05, 0.05)
    ary_scores_pos = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 1]
    ary_scores_neg = [float(y_scores[i]) for i in range(len(y_scores)) if y_true[i] == 0]

    plt.figure(figsize=(8, 6))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.xlabel("Signal score")
    plt.ylabel("counts")
    plt.hist(np.array(ary_scores_pos), bins=bins, color="orange", label="true positives", alpha=0.25)
    plt.hist(np.array(ary_scores_neg), bins=bins, color="blue", label="true negatives", alpha=0.25)
    h0, _, _ = plt.hist(np.array(ary_scores_pos), bins=bins, histtype=u"step", color="orange")
    h1, _, _ = plt.hist(np.array(ary_scores_neg), bins=bins, histtype=u"step", color="blue")
    plt.vlines(x=0.5, ymin=0.0, ymax=max([max(h0), max(h1)]), color="red", label="Decision boundary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_efficiencymap(y_pred, y_true, y_sp, figure_name, theta=0.5, sr=100):
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
        if hist1[i] == 0:
            continue
        else:
            ary_eff[i] = hist1[i] / hist0[i]
            ary_eff_err[i] = (np.sqrt((np.sqrt(hist1[i]) / hist0[i]) ** 2 +
                                      (hist1[i] * np.sqrt(hist0[i]) / hist0[i] ** 2) ** 2))

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Source position efficiency")
    axs[0].set_ylabel("Counts")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=y_true, color="black", alpha=0.5, label="Truth")
    axs[0].hist(y_sp, bins=ary_bin, histtype=u"step", weights=ary_w, color="red", alpha=0.5, linestyle="--",
                label="Prediction")
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("Source Position z-axis [mm]")
    axs[1].set_ylabel("Efficiency")
    axs[1].errorbar(ary_bin[:-1] + width, ary_eff, ary_eff_err, ary_bin_err, fmt=".", color="blue")
    # axs[1].plot(bins[:-1] + 0.5, ary_eff, ".", color="darkblue")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()
