import numpy as np
import uproot
import matplotlib.pyplot as plt


def process_eventtime(rootfile_path,
                      file_name,
                      plotting=True,
                      output_path=""):
    """
    Process TOFPET root files, marks beginning and end of every beam spills, dumps them into .txt file.
    If needed, creation of control plot can be enabled.

    Args:
        rootfile_path: (string) full path to root file
        file_name: (string) name of the run, tag will be added to .txt and .png file name
        plotting: (Bool) Toggle export of plots
        output_path: (string) Custom destination path for .txt and .png files

    Return:
         None
    """

    # open root file, open "events" tree and grab "time" leave from every event.
    # The first timestamp will be set as reference start time for relative time stamps
    rootfile = uproot.open(rootfile_path)
    events = rootfile[b"events"]
    ary_time = events["time"].array()
    start_time = min(ary_time)
    ary_time -= start_time

    # background noise level factor
    # beam signal needs to pass > f * noise_level
    f = 1.3

    # binning
    # times are given in pico-seconds -> step size of 1e12 = 1 second
    # Binning until end time is reached + 1 additional bin to cover full time range
    step = 5e10
    lim_max = int(max(ary_time) / step) + 1
    bins = np.arange(0, lim_max * step, step)

    # create histogram from relative time stamps
    # create empty lists for upper and lower time limits of beam spills
    # create empty list for upper and lower index of beam spills -> just used for better localization
    hist_hits, _ = np.histogram(ary_time, bins=bins)
    time_lower_lim = []
    time_upper_lim = []
    idx_lower_lim = []
    idx_upper_lim = []

    # noise level determination
    # last 50 entries of time histogram will be used for binning
    noise_lvl = np.mean(hist_hits[-50:])
    noise_lvl_std = np.std(hist_hits[-50:])

    # determine beam spill timings
    # First determine all bins above the noise level threshold
    list_signal_idx = []
    for i in range(len(hist_hits)):
        if hist_hits[i] > noise_lvl * f:
            list_signal_idx.append(i)

    # scan for sequential bin indices (these are the beam spill bins)
    # for each sequential bin group, take start and stop for lower/upper time bound for beam spills
    idx_lower_lim.append(list_signal_idx[0])
    time_lower_lim.append(bins[list_signal_idx[0]])
    for i in range(1, len(list_signal_idx) - 1):
        if list_signal_idx[i] == list_signal_idx[i + 1] - 1:
            continue
        else:
            idx_upper_lim.append(list_signal_idx[i])
            time_upper_lim.append(bins[list_signal_idx[i] + 1])

            idx_lower_lim.append(list_signal_idx[i + 1])
            time_lower_lim.append(bins[list_signal_idx[i + 1]])
            continue
    idx_upper_lim.append(list_signal_idx[-1])
    time_upper_lim.append(bins[list_signal_idx[-1] + 1])

    # determine background slope
    # slope determined by determining slope of first beam spill lower bound and last beam spill upper bound
    def linear(x, m, b):
        return m * x + b

    m = (hist_hits[idx_upper_lim[-1] + 2] - hist_hits[idx_lower_lim[0] - 2]) / (
            bins[idx_upper_lim[-1] + 2] - bins[idx_lower_lim[0] - 2])
    b = hist_hits[idx_lower_lim[0] - 2] - m * bins[idx_lower_lim[0] - 2]

    if plotting:
        times = np.linspace(min(ary_time), max(ary_time), 1000)

        plt.rcParams.update({'font.size': 18})
        plt.figure(figsize=(10, 6))
        plt.title(file_name)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.xlabel("Time [ps]")
        plt.ylabel("Counts per second")
        plt.hist(ary_time, bins=bins, histtype=u"step", color="black", label="signal")
        plt.plot(times, linear(times, m, b), color="red", label="offset fit")
        plt.fill_between(x=[time_lower_lim[0], time_upper_lim[0]], y1=[0, 0], y2=[max(hist_hits), max(hist_hits)],
                         color="red", alpha=0.2, label="beam time")
        for i in range(1, len(time_upper_lim)):
            plt.fill_between(x=[time_lower_lim[i], time_upper_lim[i]], y1=[0, 0], y2=[max(hist_hits), max(hist_hits)],
                             color="red", alpha=0.2)
        plt.hlines(y=noise_lvl * f, xmin=times[0], xmax=times[-1], color="blue", linestyles="--", alpha=0.2
                   , label="noise threshold")
        plt.legend(loc="lower right")
        plt.tight_layout()
        # plt.show()
        plt.savefig(output_path + file_name + "_beamtime.png")

    # time integral
    # for all beam spill bins, determine integral with subtracted background
    integral = hist_hits[list_signal_idx]
    bin_middles = bins[:-1] + step / 2
    bin_middles = bin_middles[list_signal_idx]
    for i in range(len(integral)):
        integral[i] -= int(linear(bin_middles[i], m, b))

    # determine signal height of all beam spills
    # pretty much historical, was needed for the first night of the beam time
    list_signal_height = []
    for i in range(len(integral)):
        if integral[i] >= max(integral) * 0.75:
            list_signal_height.append(integral[i])

    # print out all results
    print("### ", file_name, "\n")
    print("# General results:")
    print("Binning step size: {} [ps]".format(step))
    print("Noise level: {:.2f} +- {:.2f} (counts)".format(noise_lvl, noise_lvl_std))
    print("Integrated event number: {} (counts)".format(np.sum(integral)))
    print("Integral height: {:.2f} +- {:.2f} (counts)".format(np.mean(list_signal_height), np.std(list_signal_height)))
    print("# Slope parameter: ")
    print("x1: {} | x2: {}".format(bins[idx_lower_lim[0] - 2], bins[idx_upper_lim[-1] + 2]))
    print("y1: {} | y2: {}".format(hist_hits[idx_lower_lim[0] - 2], hist_hits[idx_upper_lim[-1] + 2]))
    print("Fitted function: y = p0 * x + p1")
    print("p0: {}".format(m))
    print("p1: {}".format(b))
    print("")
    print("\nABSOLUTE START; ABSOLUTE STOP; RELATIVE START; RELATIVE STOP")
    print(time_lower_lim[0] + start_time, time_lower_lim[0])
    print(time_upper_lim[-1] + start_time, time_upper_lim[-1])

    # dump results into txt file
    with open(file_name + "_beamtime.txt", "w") as f:
        f.write("# General results:\n")
        f.write("Binning step size: {} [ps]\n".format(step))
        f.write("Noise level: {:.2f} +- {:.2f} (counts)\n".format(noise_lvl, noise_lvl_std))
        f.write("Integrated event number: {} (counts)\n".format(np.sum(integral)))
        f.write("Integral height: {:.2f} +- {:.2f} (counts)\n".format(np.mean(list_signal_height), np.std(list_signal_height)))
        f.write("# Slope parameter: \n")
        f.write("x1: {} | x2: {}\n".format(bins[idx_lower_lim[0] - 2], bins[idx_upper_lim[-1] + 2]))
        f.write("y1: {} | y2: {}\n".format(hist_hits[idx_lower_lim[0] - 2], hist_hits[idx_upper_lim[-1] + 2]))
        f.write("Fitted function: y = p0 * x + p1\n")
        f.write("p0: {}\n".format(m))
        f.write("p1: {}\n".format(b))
        f.write("\n")
        f.write("ABSOLUTE START; ABSOLUTE STOP; RELATIVE START; RELATIVE STOP\n")
        for i in range(len(time_lower_lim)):
            f.write(str(time_lower_lim[i] + start_time) + ";" +
                    str(time_upper_lim[i] + start_time) + ";" +
                    str(time_lower_lim[i]) + ";" +
                    str(time_upper_lim[i]) + "\n")
        f.close()


process_eventtime("run00491_single.root", "run00491_single", plotting=True)
