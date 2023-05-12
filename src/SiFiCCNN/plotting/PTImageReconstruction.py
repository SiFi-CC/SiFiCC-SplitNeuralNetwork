import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------------------------------------------------------

def plot_beamprojection_dual(proj1,
                             proj1_err,
                             proj2,
                             proj2_err,
                             labels,
                             figure_name,
                             figure_title=""):
    xticks = np.arange(0, len(proj1) + 10.0, 10.0)
    xlabels = xticks - len(proj1) / 2
    x = np.arange(0, len(proj1), 1.0, dtype=int)

    plt.figure()
    plt.title(figure_title)
    plt.xlabel("z-position [mm]")
    plt.xticks(xticks, xlabels)

    plt.plot(x, proj1, label=labels[0], color="blue")
    plt.fill_between(x, proj1 - proj1_err, proj1 + proj1_err, color="blue", alpha=0.5)
    plt.plot(x, proj2, label=labels[1], color="orange")
    plt.fill_between(x, proj2 - proj2_err, proj2 + proj2_err, color="orange", alpha=0.5)

    plt.legend(loc="upper right")
    plt.grid()
    plt.tight_layout()
    plt.savefig(figure_name + ".png")
