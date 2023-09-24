# ##################################################################################################
# Script PostProcessingFit.py
#
# This script adds a post-processing routing for image reconstruction. From the final back
# projection the common background will be subtracted and the range shift will be determined by
# a chi2 method.
#
# ##################################################################################################

import numpy as np
import argparse
import uproot
import os
# interpolation: cubic smoothing spline method
from scipy import interpolate

import matplotlib.pyplot as plt


def RMSE(ary_y1, ary_y2):
    return np.sqrt(np.sum([(i - j) ** 2 for i, j in zip(ary_y1, ary_y2)]) / len(ary_y1))


def splining(ary_x, ary_y, width=0.2):
    binning_new = np.arange(-70, 70, width)
    binning_new_center = binning_new[:-1] + width / 2

    tck_ary = interpolate.splrep(ary_x, ary_y, s=0)
    ary_y_spline = interpolate.splev(binning_new_center, tck_ary, der=0)

    return binning_new_center, ary_y_spline


def load_template(file_sg):
    root_file_sg = uproot.open(file_sg)

    # Signal and background template files created by CC6 image reconstruction
    # splicing at the end of the projection is to cut out overflow and underflow bin of root
    image_iter5_sg = np.rot90(root_file_sg[b'image_iter5_smooth_2.00;1'].values)
    proj_iter5_sg = np.sum(image_iter5_sg, axis=0)[1:-1]
    root_file_sg.close()

    return proj_iter5_sg


def scale_by_statistics(n_protons, n_protons_template=4e9):
    return n_protons / n_protons_template


def bp_str_converter(bp):
    if bp == "BP0mm":
        return 0
    if bp == "BP5mm":
        return 5
    if bp == "BP10mm":
        return 10
    if bp == "BPm5mm":
        return -5


def RMSE_shifting(file,
                  iteration,
                  tmp_sg,
                  k_lower,
                  k_upper,
                  step_size,
                  steps,
                  do_plots=True):
    root_file = uproot.open(file)
    # grab image from root file and create copy, then root file can be close properly
    # image is rotated so beam axis (z-axis) is on the x-axis in the array
    # the splicing at the end removes the overlow and underflow bin from root histograms
    image_iter5 = np.rot90(root_file[b'image_iter5_smooth_2.00;1'].values)
    proj_iter5 = np.sum(image_iter5, axis=0)[1:-1]
    root_file.close()

    # define the signal projection from root file
    # re-create binning from ComptonCamera6, normal settings cover a range of -150mm to 150mm
    # with a bin width of 1mm
    sg = proj_iter5
    bins = np.linspace(-150, 150, 301)
    width = bins[1] - bins[0]
    x = bins[:-1] + width / 2

    # cubic splining method
    x_spline, sg_spline = splining(x, sg, width=step_size)
    _, tmp_spline = splining(x, tmp_sg, width=step_size)

    # select window of interest
    # final selected window: [k_lower:k_upper]
    if k_lower is None and k_upper is None:
        k_lower = int(len(x_spline) / 2) - 50
        k_upper = int(len(x_spline) / 2) + 5
    else:
        k_lower += int(len(x_spline) / 2)
        k_upper += int(len(x_spline) / 2)

    # RSME minimization algorithm
    # Use sliding windows method to scan -steps*step_size to steps*step_size window
    list_rmse = []
    list_shift = []
    list_kstep = []
    for j in range(-steps, steps):
        rmse_jk = RMSE(sg_spline[k_lower + j: k_upper + j],
                       tmp_spline[k_lower: k_upper])
        list_rmse.append(rmse_jk)
        list_shift.append(j * step_size)
        list_kstep.append(j)
    min_rmse = list_rmse[np.argmin(list_rmse)]
    min_shift = list_shift[np.argmin(list_rmse)]
    min_kstep = list_kstep[np.argmin(list_rmse)]

    # generate control plots
    if do_plots:
        if iteration % 10 == 0:
            plot_RMSE_fit(x_spline=x_spline,
                          sg_spline=sg_spline,
                          tmp_spline=tmp_spline,
                          k_upper=k_upper,
                          k_lower=k_lower,
                          min_kstep=min_kstep,
                          min_shift=min_shift,
                          min_rmse=min_rmse,
                          step_size=step_size,
                          file_name=file[:-5] + "_RMSE_fit.png")
            plot_RMSE_min(list_shift=list_shift,
                          list_rmse=list_rmse,
                          step_size=step_size,
                          file_name=file[:-5] + "_RMSE_min.png")

    return list_rmse, list_shift, list_kstep


def plot_RMSE_fit(x_spline,
                  sg_spline,
                  tmp_spline,
                  k_upper,
                  k_lower,
                  min_kstep,
                  min_shift,
                  min_rmse,
                  step_size,
                  file_name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[1].set_xlabel("Position along beam axis [mm]")
    axs[1].plot(x_spline[k_lower + min_kstep: k_upper + min_kstep],
                tmp_spline[k_lower: k_upper],
                ".", color="grey")
    axs[1].plot(x_spline[k_lower + min_kstep: k_upper + min_kstep],
                sg_spline[k_lower + min_kstep: k_upper + min_kstep],
                ".", color="red", label="Shift: {:.1f} +/- {:.1f}\nRMSE: {:.3f}".format(
            min_shift, step_size, min_rmse))
    axs[1].grid()
    axs[1].legend()
    axs[0].set_xlabel("Position along beam axis [mm]")
    axs[0].set_ylabel("Counts")
    axs[0].plot(x_spline, tmp_spline,
                linestyle="--", color="grey", alpha=0.3, label="Reference curve")
    axs[0].plot(x_spline, sg_spline,
                linestyle="--", color="red", alpha=0.3, label="Signal curve")
    axs[0].plot(x_spline[k_lower: k_upper],
                tmp_spline[k_lower: k_upper],
                ".", color="grey")
    axs[0].plot(x_spline[k_lower + min_kstep: k_upper + min_kstep],
                sg_spline[k_lower + min_kstep: k_upper + min_kstep],
                ".", color="red")
    axs[0].grid()
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_RMSE_min(list_shift,
                  list_rmse,
                  step_size,
                  file_name):
    min_rsme = list_rmse[np.argmin(list_rmse)]
    min_shift = list_shift[np.argmin(list_rmse)]

    plt.figure(figsize=(12, 2))
    plt.xlabel("Shift [mm]")
    plt.ylabel("RMSE")
    plt.plot(list_shift, list_rmse, ".", color="black")
    plt.vlines(x=min_shift, ymin=min_rsme, ymax=max(list_rmse), linestyle="--", color="red")
    plt.fill_between([min_shift - step_size, min_shift + step_size], [min_rsme, min_rsme],
                     [max(list_rmse), max(list_rmse)], color="red", alpha=0.3)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_RMSE_dist(list_rmse,
                   filename):
    plt.figure(figsize=(8, 6))
    plt.xlabel("RMSE")
    plt.ylabel("Counts")
    plt.hist(list_rmse, histtype=u"step", bins=20, color="blue")
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def dfop_fit_full(run_name,
                  list_bp,
                  list_protons,
                  file_sg,
                  k_lower,
                  k_upper,
                  step_size=0.2,
                  steps=50,
                  list_scale=None):
    # grab main working path
    path = os.getcwd()

    # initialize lists for chi2 calculations
    list_chi2 = []
    total_bp = []
    total_dfop = []
    total_dfop_err = []

    # main iteration over every Bragg-peak position and proton statistic
    for b, bp in enumerate(list_bp):
        # initialize local chi2 calculations
        local_bp = []
        local_dfop = []
        local_dfop_err = []

        for protons in list_protons:
            # grab correct root file
            path_run = "{}/{}/".format(path, run_name)
            path_target = path_run + "/{}/{}protons/".format(bp, protons)
            file_target = run_name + "_{}_{}protons".format(bp, protons)

            # load signal and background template
            tmp_sg = load_template(file_sg=path_run + file_sg)
            # rescale template curves accordingly

            f_scale = scale_by_statistics(float(protons), 4e9)
            if list_scale is not None:
                f_scale *= list_scale[b]

            list_dfop = []
            list_min_rmse = []
            for i in range(100):
                # grab correct root file iteration
                file = path_target + file_target + "_Subset_{}.root".format(i)

                # rescale template curves
                tmp_sg *= f_scale

                # define window of interest
                k_lower = int(350 + (bp_str_converter(bp) / step_size) - (5. / step_size))
                k_upper = int(350 + (bp_str_converter(bp) / step_size) + (5. / step_size))

                list_rmse, list_shift, list_kstep = RMSE_shifting(file=file,
                                                                  iteration=i,
                                                                  tmp_sg=tmp_sg,
                                                                  k_lower=k_lower,
                                                                  k_upper=k_upper,
                                                                  step_size=step_size,
                                                                  steps=steps)

                min_shift = list_shift[np.argmin(list_rmse)]
                min_rmse = list_rmse[np.argmin(list_rmse)]
                list_dfop.append(min_shift)
                list_min_rmse.append(min_rmse)

            # plot RMSE distribution
            plot_RMSE_dist(list_rmse=list_min_rmse,
                           filename=path_target + file_target + "_RMSE_dist.png")

            # additional steps for local chi2 test
            # (Local means for one Bragg-peak position)
            local_bp.append(bp_str_converter(bp))
            local_dfop.append(np.mean(list_dfop))
            local_dfop_err.append(np.std(list_dfop))

            print("DFOP FIT: {}, {}protons : {:.3f} +/- {:.3f}".format(
                bp, protons, np.mean(list_dfop), np.std(list_dfop)))
        # local chi2 test
        chi2 = 0
        for j in range(len(local_dfop)):
            chi2 += (local_bp[j] - local_dfop[j]) ** 2 / local_dfop_err[j]
        chi2 /= (len(local_dfop) - 1)
        list_chi2.append(chi2)

        total_bp = np.concatenate([total_bp, local_bp])
        total_dfop = np.concatenate([total_dfop, local_dfop])
        total_dfop_err = np.concatenate([total_dfop_err, local_dfop_err])

    # global chi2 test
    chi2 = 0
    for j in range(len(total_bp)):
        chi2 += (total_bp[j] - total_dfop[j]) ** 2 / total_dfop_err[j]
    chi2 /= (len(total_dfop) - len(list_bp))

    print("#######################################")
    for k in range(len(list_chi2)):
        print("Chi2 Fit {}: Chi2/ndof = {:.3f}".format(list_bp[k], list_chi2[k]))
    print("Total CHi2 Fit: Chi2/ndof = {:.3f}".format(chi2))


def scan_fit_range(run_name,
                   list_bp,
                   list_protons,
                   file_sg,
                   list_scale,
                   fit_start,
                   fit_finish,
                   fit_min,
                   step_size,
                   steps=50):
    # grab main working path
    path = os.getcwd()
    path_run = "{}/{}/".format(path, run_name)

    # convert fitting parameter to indexing
    k_start = int((fit_start / step_size))
    k_finish = int((fit_finish / step_size))
    k_min = int((fit_min / step_size))

    # generate final array containing chi2 values
    ary_size = int(k_finish - k_start)
    ary_chi2 = np.zeros(shape=(ary_size, ary_size))

    # generate steps
    k_steps = np.arange(k_start, k_finish, 1.0, dtype=int)

    # main iteration over all possible fit ranges
    for i, k0 in enumerate(k_steps):
        for j, k1 in enumerate(k_steps):
            if k1 <= k0:
                continue
            if (k1 - k0) < k_min:
                continue
            print("Iterating fit range: [{:.1f}, {:.1f}]".format(k0 * step_size, k1 * step_size))
            # initialize lists for chi2 calculations
            total_bp = []
            total_dfop = []
            total_dfop_err = []

            # main iteration over every Bragg-peak position and proton statistic
            for b, bp in enumerate(list_bp):
                for protons in list_protons:
                    # grab correct root file
                    path_target = path_run + "/{}/{}protons/".format(bp, protons)
                    file_target = run_name + "_{}_{}protons".format(bp, protons)

                    # load signal and background template
                    tmp_sg = load_template(file_sg=path_run + file_sg)
                    # rescale template curves accordingly
                    f_scale = scale_by_statistics(float(protons), 4e9)
                    if list_scale is not None:
                        f_scale *= list_scale[b]

                    list_dfop = []
                    list_min_rmse = []
                    for k in range(100):
                        # grab correct root file iteration
                        file = path_target + file_target + "_Subset_{}.root".format(k)

                        # rescale template curves
                        tmp_sg *= f_scale

                        # RMSE fitting
                        list_rmse, list_shift, list_kstep = RMSE_shifting(file=file,
                                                                          iteration=k,
                                                                          tmp_sg=tmp_sg,
                                                                          k_lower=k0,
                                                                          k_upper=k1,
                                                                          step_size=step_size,
                                                                          steps=steps,
                                                                          do_plots=False)

                        min_shift = list_shift[np.argmin(list_rmse)]
                        min_rmse = list_rmse[np.argmin(list_rmse)]
                        list_dfop.append(min_shift)
                        list_min_rmse.append(min_rmse)

                    # additional steps for local chi2 test
                    # (Local means for one Bragg-peak position)
                    total_bp.append(bp_str_converter(bp))
                    total_dfop.append(np.mean(list_dfop))
                    total_dfop_err.append(np.std(list_dfop))

            # global chi2 test
            chi2 = 0
            for m in range(len(total_bp)):
                chi2 += (total_bp[m] - total_dfop[m]) ** 2 / total_dfop_err[m]
            chi2 /= (len(total_dfop) - len(list_bp))

            ary_chi2[i, j] = chi2

    # save results
    np.save(path_run + "fit_range_scan", ary_chi2)


"""
if __name__ == "__main__":
    # configure argument parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument("--run", type=str, help="Name of run")
    parser.add_argument("--bp", type=str, help="Bragg peak position")
    parser.add_argument("--protons", type=str, help="Number of protons")
    args = parser.parse_args()

    main(run_name=args.run, bp=args.bp, n_protons=args.protons)
"""

if __name__ == "__main__":
    # test main function
    RUM_NAME = "CC6IR_ECRNSiPM"
    LIST_BP = ["BP0mm", "BP5mm", "BP10mm", "BPm5mm"]
    LIST_PROTONS = ["1e8", "2e8", "5e8", "1e9"]
    FILE_SG = "CC6_ECRNSiPM_4to1_simv4_0mm_4e9protons_SIGNAL.root"
    FILE_BG = "CC6_ECRNSiPM_4to1_simv4_0mm_4e9protons_SIGNAL.root"
    K_LOWER = None
    K_UPPER = None
    STEP_SIZE = 0.2
    STEPS = 50
    LIST_SCALE = [1.0, 1.020, 0.980, 0.835]
    """
    dfop_fit_full(run_name=RUM_NAME,
                  list_bp=LIST_BP,
                  list_protons=LIST_PROTONS,
                  file_sg=FILE_SG,
                  k_lower=K_LOWER,
                  k_upper=K_UPPER,
                  step_size=0.2,
                  steps=50,
                  list_scale=LIST_SCALE)
    """
    scan_fit_range(run_name=RUM_NAME,
                   list_bp=LIST_BP,
                   list_protons=LIST_PROTONS,
                   file_sg=FILE_SG,
                   step_size=0.2,
                   steps=50,
                   list_scale=LIST_SCALE,
                   fit_start=-5,
                   fit_finish=15,
                   fit_min=5)
