import sys
import os
import numpy as np

from src import RepoStructure
from src import ArgParser
from src import ConfigFileParser
from src import NPZParser
from src import MetaData
from src import NeuralNetwork


def main():
    # set up the repo directory structure
    RepoStructure.setup_directory()

    # set main directory + subdirectories
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"

    # generate argument parser
    arg_parser = ArgParser.parser()

    # execute main functionality only if config file was given
    if arg_parser.cf is not None:
        print("Reading config file ...")
        # Parse content of config file and store parameter in ConfigData object
        config_data = ConfigFileParser.parse(arg_parser.cf)

        # create sub directory for run output
        if not os.path.isdir(dir_main + "/results/" + config_data.RUN_NAME + "/"):
            os.mkdir(dir_main + "/results/" + config_data.RUN_NAME + "/")

        if not os.path.isdir(
                dir_main + "/results/" + config_data.RUN_NAME + "/" + config_data.ROOT_FILE_NAME[:-5] + "/"):
            os.mkdir(dir_main + "/results/" + config_data.RUN_NAME + "/" + config_data.ROOT_FILE_NAME[:-5] + "/")

        # training input data can be overloaded here
        config_data.NN_INPUT = "OptimizedGeometry_BP05_S1AX.npz"

        # grab cluster data object and meta data object
        data_cluster = NPZParser.parse(dir_npz + config_data.NN_INPUT)
        meta_data = MetaData.MetaData(dir_npz + config_data.META_DATA)

        data_cluster.features = np.delete(data_cluster.features, -2, 1)

        # fill additional meta data attributes
        meta_data.root_file_name = config_data.ROOT_FILE_NAME

        # load neural network models from model names defined in config file
        # evaluate model expression
        method_model_classifier = __import__("models." + config_data.MODEL_NAME_CLASSIFIER, fromlist=[None])
        func_model_classifier = getattr(method_model_classifier, "return_model")
        model_classifier = func_model_classifier(data_cluster.num_features())
        nn_classifier = NeuralNetwork.NeuralNetwork(model=model_classifier,
                                                    model_name=config_data.MODEL_NAME_CLASSIFIER,
                                                    model_tag=config_data.RUN_NAME)

        method_model_regression1 = __import__("models." + config_data.MODEL_NAME_REGRESSION1, fromlist=[None])
        func_model_regression1 = getattr(method_model_regression1, "return_model")
        model_regression1 = func_model_regression1(data_cluster.num_features())
        nn_regression1 = NeuralNetwork.NeuralNetwork(model=model_regression1,
                                                     model_name=config_data.MODEL_NAME_REGRESSION1,
                                                     model_tag=config_data.RUN_NAME)

        method_model_regression2 = __import__("models." + config_data.MODEL_NAME_REGRESSION2, fromlist=[None])
        func_model_regression2 = getattr(method_model_regression2, "return_model")
        model_regression2 = func_model_regression2(data_cluster.num_features())
        nn_regression2 = NeuralNetwork.NeuralNetwork(model=model_regression2,
                                                     model_name=config_data.MODEL_NAME_REGRESSION2,
                                                     model_tag=config_data.RUN_NAME)

        method_training_schedule = __import__("src." + config_data.TRAINING_SCHEDULE_NAME, fromlist=[None])
        func_training_schedule = getattr(method_training_schedule, "train_schedule")

        # change directory for writing results
        os.chdir(dir_main + "/results/" + config_data.RUN_NAME + "/" + config_data.ROOT_FILE_NAME[:-5] + "/")

        func_training_schedule(data_cluster=data_cluster,
                               meta_data=meta_data,
                               nn_classifier=nn_classifier,
                               nn_regression1=nn_regression1,
                               nn_regression2=nn_regression2,
                               load_classifier=config_data.LOAD_CLASSIFIER,
                               load_regression1=config_data.LOAD_REGRESSION1,
                               load_regression2=config_data.LOAD_REGRESSION2
                               )



    else:
        print("No configfile found!")
        print("CODE WILL BE ABORTED!")
        sys.exit()


if __name__ == "__main__":
    main()
