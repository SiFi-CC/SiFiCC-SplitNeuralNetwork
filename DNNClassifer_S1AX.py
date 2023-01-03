import os
import sys
import numpy as np

from src import NPZParser
from src import MetaData
from src import NeuralNetwork
from src import NetworkEvaluation
from src import Plotter
from src import fastROCAUC
from src import MLEMExportCutBased


def train_model(NeuralNetwork, npz_file):
    # load npz file
    data_cluster = NPZParser.parse(npz_file)

    # update training-valid ration to increase validation set
    data_cluster.p_train = 0.7
    data_cluster.p_valid = 0.1
    data_cluster.p_test = 0.2

    # set class weights as sample weights
    data_cluster.weights *= data_cluster.get_classweights()

    # standardize input
    data_cluster.standardize()

    # update run settings
    NeuralNetwork.epochs = 30
    NeuralNetwork.batch_size = 256

    print("##### Training classifier model: ")
    print("NPZ_FILE: ", npz_file)
    print("Feature dimension: ({} ,{})".format(data_cluster.features.shape[0], data_cluster.features.shape[1]))
    print("")

    NeuralNetwork.train(data_cluster.x_train(),
                        data_cluster.y_train(),
                        data_cluster.w_train(),
                        data_cluster.x_valid(),
                        data_cluster.y_valid())

    # save model
    NeuralNetwork.save()

    # evaluate training process of the model
    y_scores = NeuralNetwork.predict(data_cluster.x_test())
    y_true = data_cluster.y_test()

    Plotter.plot_history_classifier(NeuralNetwork, "training_historyclassifier")
    NetworkEvaluation.write_metrics_classifier(y_scores, y_true)
    Plotter.plot_score_dist(y_scores, y_true, "training_scoredist")
    fastROCAUC.fastROCAUC(y_scores, y_true, save_fig="training_ROCAUC")


def evaluation(NeuralNetwork, npz_file):
    # load npz file
    data_cluster = NPZParser.parse(npz_file)

    # standardize input
    data_cluster.standardize()

    # evaluate test dataset
    y_scores = NeuralNetwork.predict(data_cluster.features)
    y_true = data_cluster.targets_clas

    NetworkEvaluation.write_metrics_classifier(y_scores, y_true)
    Plotter.plot_score_dist(y_scores, y_true, "scoredist")
    fastROCAUC.fastROCAUC(y_scores, y_true, save_fig="ROCAUC")

    ary_sourcepos_true = [data_cluster.meta[i, 2] for i in range(len(y_true)) if y_true[i] == 1]
    Plotter.plot_source_position(y_scores,
                                 data_cluster.meta[:, 2],
                                 ary_sourcepos_true,
                                 "dist_source_position")


def export_mlem(NeuralNetwork, npz_file):
    # settings
    theta = 0.5

    # load npz file
    data_cluster = NPZParser.parse(npz_file)

    # standardize input
    data_cluster.standardize()

    # evaluate test dataset
    y_scores = NeuralNetwork.predict(data_cluster.features)
    y_true = data_cluster.targets_clas

    # pre-define
    y_pred = np.zeros(shape=(len(y_scores, )))

    for i in range(len(y_pred)):
        # apply prediction threshold
        if y_scores[i] >= theta:
            y_pred[i] = 1
        else:
            y_pred[i] = 0

    list_idx_positives = y_pred == 1

    # denormalize features
    for i in range(data_cluster.features.shape[1]):
        data_cluster.features[:, i] *= data_cluster.list_std[i]
        data_cluster.features[:, i] += data_cluster.list_mean[i]

    # grab event kinematics from feature list
    ary_e = data_cluster.features[list_idx_positives, 1]
    ary_ex = data_cluster.features[list_idx_positives, 2]
    ary_ey = data_cluster.features[list_idx_positives, 3]
    ary_ez = data_cluster.features[list_idx_positives, 4]

    # select only absorber energies
    # select only positive events
    # replace -1. (NaN) values with 0.
    ary_p = data_cluster.features[:, [10, 19, 28, 37, 46]]
    ary_p = ary_p[list_idx_positives, :]
    for i in range(ary_p.shape[0]):
        for j in range(ary_p.shape[1]):
            if ary_p[i, j] == -1.:
                ary_p[i, j] = 0.0
    ary_p = np.sum(ary_p, axis=1)

    ary_px = data_cluster.features[list_idx_positives, 11]
    ary_py = data_cluster.features[list_idx_positives, 12]
    ary_pz = data_cluster.features[list_idx_positives, 13]

    MLEMExportCutBased.export_mlem(ary_e, ary_p, ary_ex, ary_ey, ary_ez, ary_px, ary_py, ary_pz,
                                   "OptimizedGeometry_BP0mm_2e10protons_DNN_S1AX_filter")


########################################################################################################################

def main():
    # GLOBAL SETTINGS
    RUN_NAME = "test"
    is_train = True
    is_export_mlem = False

    # define directory paths
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"

    # define file settings
    ROOT_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons.root"
    ROOT_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons.root"

    NPZ_FILE_TRAIN = "OptimisedGeometry_BP0mm_2e10protons_DNN_S1AX.npz"
    NPZ_FILE_BP0mm = "OptimisedGeometry_BP0mm_2e10protons_DNN_S1AX.npz"
    NPZ_FILE_BP5mm = "OptimisedGeometry_BP5mm_4e9protons_DNN_S1AX.npz"

    # generate needed directories in the "results" subdirectory
    # create subdirectory for run output
    if not os.path.isdir(dir_main + "/results/" + RUN_NAME + "/"):
        os.mkdir(dir_main + "/results/" + RUN_NAME + "/")

    if not os.path.isdir(
            dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP0mm[:-4] + "/"):
        os.mkdir(dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP0mm[:-4] + "/")

    if not os.path.isdir(
            dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP5mm[:-4] + "/"):
        os.mkdir(dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP5mm[:-4] + "/")

    # CHANGE DIRECTORY INTO THE NEWLY GENERATED RESULTS DIRECTORY
    # TODO: fix this pls
    os.chdir(dir_main + "/results/" + RUN_NAME + "/")

    # load up the Tensorflow model
    from models.DNN_base_classifier import return_model
    tf_model = return_model(54)
    neuralnetwork_classifier = NeuralNetwork.NeuralNetwork(model=tf_model,
                                                           model_name=RUN_NAME,
                                                           model_tag="")

    if is_train:
        train_model(neuralnetwork_classifier, dir_npz + NPZ_FILE_TRAIN)
    else:
        neuralnetwork_classifier.load()

    # evaluate both test datasets
    os.chdir(dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP0mm[:-4] + "/")
    evaluation(neuralnetwork_classifier, dir_npz + NPZ_FILE_BP0mm)
    if is_export_mlem:
        export_mlem(neuralnetwork_classifier, dir_npz + NPZ_FILE_BP0mm)

    os.chdir(dir_main + "/results/" + RUN_NAME + "/" + NPZ_FILE_BP5mm[:-4] + "/")
    evaluation(neuralnetwork_classifier, dir_npz + NPZ_FILE_BP5mm)
    if is_export_mlem:
        export_mlem(neuralnetwork_classifier, dir_npz + NPZ_FILE_BP5mm)


if __name__ == "__main__":
    main()
