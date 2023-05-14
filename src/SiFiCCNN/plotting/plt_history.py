import numpy as np
import matplotlib.pyplot as plt

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


def plot_history_classifier(history, figure_name):
    plt.rcParams.update({'font.size': 16})
    # plot model performance
    loss = history['loss']
    val_loss = history['val_loss']
    # mse = nn_classifier.history["accuracy"]
    # val_mse = nn_classifier.history["val_accuracy"]

    eff = history["precision"]
    val_eff = history["val_precision"]
    pur = history["recall"]
    val_pur = history["val_recall"]

    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax2.plot(eff, label="Efficiency", linestyle='-', color="red")
    ax2.plot(val_eff, label="Validation", linestyle='--', color="red")
    ax2.plot(pur, label="Purity", linestyle="-", color="green")
    ax2.plot(val_pur, label="Validation", linestyle="--", color="green")
    ax2.set_ylabel("%")
    ax2.legend(loc="upper right")
    ax2.grid()

    ax1.plot(loss, label="Loss", linestyle='-', color="blue")
    ax1.plot(val_loss, label="Validation", linestyle='--', color="blue")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_history_regression(history, figure_name):
    plt.rcParams.update({'font.size': 16})

    loss = history['loss']
    val_loss = history['val_loss']
    mse = history["mean_absolute_error"]
    val_mse = history["val_mean_absolute_error"]

    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Loss", linestyle='-', color="blue")
    plt.plot(val_loss, label="Validation", linestyle='--', color="blue")
    # plt.plot(mse, label="MAE", linestyle='-', color="red")
    # plt.plot(val_mse, linestyle='--', color="red")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.xlim(-5, 100)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()


def plot_efficiency_sourceposition(ary_sp_pred, ary_sp_true, figure_name):
    bins = np.arange(-100.0, 100.0, 2.0)

    ary_eff = np.zeros(shape=(len(bins) - 1,))
    list_eff_err = []
    ary_bin_err = np.ones(shape=(len(bins) - 1,)) * 0.1
    hist1, _ = np.histogram(ary_sp_pred, bins=bins)
    hist0, _ = np.histogram(ary_sp_true, bins=bins)
    for i in range(len(ary_eff)):
        if hist1[i] == 0:
            list_eff_err.append(0)
            continue
        else:
            ary_eff[i] = hist1[i] / hist0[i]
            list_eff_err.append(np.sqrt((np.sqrt(hist1[i]) / hist0[i]) ** 2 +
                                        (hist1[i] * np.sqrt(hist0[i]) / hist0[
                                            i] ** 2) ** 2))

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Source position efficiency")
    axs[0].set_ylabel("Counts")
    axs[0].hist(ary_sp_true, bins=bins, histtype=u"step", color="black",
                label="All Ideal Compton")
    axs[0].hist(ary_sp_pred, bins=bins, histtype=u"step", color="red",
                linestyle="--", label="True Positives")
    axs[0].legend(loc="lower center")
    axs[1].set_xlabel("Source Position z-axis [mm]")
    axs[1].set_ylabel("Efficiency")
    axs[1].errorbar(bins[:-1] + 0.1, ary_eff, list_eff_err, ary_bin_err,
                    fmt=".", color="blue")
    # axs[1].plot(bins[:-1] + 0.5, ary_eff, ".", color="darkblue")
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
    plt.close()
