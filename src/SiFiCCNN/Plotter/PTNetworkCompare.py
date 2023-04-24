import numpy as np
import matplotlib.pyplot as plt

# update matplotlib parameter for bigger font size
plt.rcParams.update({'font.size': 12})


# ----------------------------------------------------------------------------------------------------------------------

def plot_compare_classifier(nn1_loss,
                            nn1_val_loss,
                            nn1_eff,
                            nn1_val_eff,
                            nn1_pur,
                            nn1_val_pur,
                            nn2_loss,
                            nn2_val_loss,
                            nn2_eff,
                            nn2_val_eff,
                            nn2_pur,
                            nn2_val_pur,
                            labels,
                            figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})
    y_upperlim = max(np.max(nn1_val_eff), np.max(nn2_val_eff)) + 0.1
    y_lowerlim = min(np.min(nn1_val_pur), np.min(nn2_val_pur)) - 0.1

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_loss, label=labels[0] + " Loss", linestyle='-', color="blue")
    plt.plot(nn1_val_loss, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_loss, label=labels[1] + " Loss", linestyle='-', color="orange")
    plt.plot(nn2_val_loss, label="Validation", linestyle='--', color="orange")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()

    plt.fig, axs1 = plt.subplots(figsize=(8, 6))
    axs1.set_xlabel("Epochs")
    axs1.set_ylabel("Efficiency", color="red")
    axs1.tick_params(axis='y', labelcolor="red")
    axs1.set_ylim(0.2, 0.9)
    axs1.plot(nn1_eff, label=labels[0] + " Efficiency", linestyle="-", color="black")
    axs1.plot(nn1_val_eff, linestyle="--", color="black")
    axs1.plot(nn2_eff, label=labels[1] + " Efficiency", linestyle="-", color="red")
    axs1.plot(nn2_val_eff, linestyle="--", color="red")

    axs2 = axs1.twinx()
    axs2.set_ylabel("Purity", color="blue")
    axs2.tick_params(axis='y', labelcolor="blue")
    axs2.set_ylim(0.2, 0.9)
    axs2.plot(nn1_pur, label=labels[0] + " Purity", linestyle="-", color="black")
    axs2.plot(nn1_val_pur, linestyle="--", color="black")
    axs2.plot(nn2_pur, label=labels[1] + " Purity", linestyle="-", color="blue")
    axs2.plot(nn2_val_pur, linestyle="--", color="blue")

    axs1.grid()
    plt.tight_layout()
    axs1.legend(loc="lower left")
    axs2.legend(loc="lower right")
    plt.savefig(figure_name + "_metrics" + ".png")
    plt.close()


def plot_compare_regression_loss(nn1_mae,
                                 nn1_val_mae,
                                 nn2_mae,
                                 nn2_val_mae,
                                 labels,
                                 figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_mae, label=labels[0] + " MAE", linestyle='-', color="blue")
    plt.plot(nn1_val_mae, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_mae, label=labels[1] + " MAE", linestyle='-', color="orange")
    plt.plot(nn2_val_mae, label="Validation", linestyle='--', color="orange")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()


def plot_compare_regression_position(nn1_mae,
                                     nn1_val_mae,
                                     nn2_mae,
                                     nn2_val_mae,
                                     labels,
                                     figure_name):
    # global params
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(8, 6))
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(nn1_mae, label=labels[0] + " MAE", linestyle='-', color="blue")
    plt.plot(nn1_val_mae, label="Validation", linestyle='--', color="blue")
    plt.plot(nn2_mae, label=labels[1] + " MAE", linestyle='-', color="orange")
    plt.plot(nn2_val_mae, label="Validation", linestyle='--', color="orange")
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(figure_name + "_loss" + ".png")
    plt.close()
