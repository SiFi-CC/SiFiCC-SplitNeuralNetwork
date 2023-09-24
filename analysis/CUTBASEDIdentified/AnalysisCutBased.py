import numpy as np
import os

from SiFiCCNN.root import RootParser, RootFiles
from SiFiCCNN.analysis.metrics import get_classifier_metrics
from SiFiCCNN.utils.plotter import plot_energy_error, plot_position_error, plot_energy_resolution


def main():
    n = None
    ROOTFILE = RootFiles.fourtoone_CONT_simv4
    RUN_NAME = "CutBasedIdentified"

    # go backwards in directory tree until the main repo directory is matched
    path = os.getcwd()
    while True:
        path = os.path.abspath(os.path.join(path, os.pardir))
        if os.path.basename(path) == "SiFiCC-SplitNeuralNetwork":
            break
    path_main = path
    path_root = path_main + "/root_files/"
    path_results = path_main + "/results/" + RUN_NAME + "/"

    # create subdirectory for run output
    if not os.path.isdir(path_results):
        os.mkdir(path_results)
    if not os.path.isdir(path_results + "/" + ROOTFILE + "/"):
        os.mkdir(path_results + "/" + ROOTFILE + "/")

    # generate root parser object
    root_parser = RootParser.RootParser(path_root + ROOTFILE)

    # fix number of events used
    if n is None:
        n = root_parser.events_entries

    # create empty arrays for analysis
    ary_cbreco = np.zeros(shape=(n, 9))
    ary_mctrue = np.zeros(shape=(n, 9))

    # main iteration over root file
    for i, event in enumerate(root_parser.iterate_events(n=n)):
        # cut based reconstruction
        identified = (event.Identified != 0) * 1
        reco_energy_e, reco_energy_p = event.get_reco_energy()
        reco_position_e, reco_position_p = event.get_reco_position()
        # monte carlo truth
        dist_compton = event.get_distcompton_tag() * 1
        true_energy_e, true_energy_p = event.get_target_energy()
        true_position_e, true_position_p = event.get_target_position()

        ary_mctrue[i, :] = [dist_compton,
                            true_energy_e,
                            true_energy_p,
                            true_position_e.x,
                            true_position_e.y,
                            true_position_e.z,
                            true_position_p.x,
                            true_position_p.y,
                            true_position_p.z]
        ary_cbreco[i, :] = [identified,
                            reco_energy_e,
                            reco_energy_p,
                            reco_position_e.x,
                            reco_position_e.y,
                            reco_position_e.z,
                            reco_position_p.x,
                            reco_position_p.y,
                            reco_position_p.z]
    # calculate classifier metrics
    acc, eff, pur, (tp, fp, tn, fn) = get_classifier_metrics(ary_cbreco[:, 0], ary_mctrue[:, 0])

    # print classifier statistic
    print("### Classifier statistics: ")
    print("Classifier Accuracy:    {:.1f} %".format(acc * 100))
    print("Classifier Efficiency:  {:.1f} %".format(eff * 100))
    print("Classifier Purity:      {:.1f} %".format(pur * 100))

    tp_mask = np.array([ary_mctrue[i, 0] * ary_cbreco[i, 0] for i in range(n)])
    # plotting of results
    os.chdir(path_results + ROOTFILE + "/")
    plot_energy_error(ary_cbreco[tp_mask == 1, 1:3],
                      ary_mctrue[tp_mask == 1, 1:3],
                      "error_regression_energy")
    plot_position_error(ary_cbreco[tp_mask == 1, 3:],
                        ary_mctrue[tp_mask == 1, 3:],
                        "error_regression_position")
    plot_energy_resolution(ary_cbreco[tp_mask == 1, 1:3],
                           ary_mctrue[tp_mask == 1, 1:3],
                           "resolution_regression_energy")


if __name__ == "__main__":
    main()
