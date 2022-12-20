from classes import ArgParser
from classes import ConfigFileParser
from classes import ConfigVerify
from classes import NPZParser
from classes import SiFiCCNNTF
from classes import MetaData
from classes import RootParser

import sys
import os


########################################################################################################################

def main():
    # generate argument parser
    args = ArgParser.parser()

    # set main directory + subdirectories
    dir_main = os.getcwd()
    dir_root = dir_main + "/root_files/"
    dir_npz = dir_main + "/npz_files/"

    if args.cf is not None:
        print("Reading config file ...")
        config_data = ConfigFileParser.parse(args.cf)

        if ConfigVerify.config_verify(ConfigData=config_data):
            print("\nStarting SiFi-CC Neural Network Framework")

            # TODO: rework this stuff as well
            if args.geninput:
                print("Generating input")
                # call RootParser with root file defined in config file
                root_data = RootParser(dir_root + config_data.ROOT_FILE_NAME)

                # call InputGenerator method
                method_inputgenerator = __import__(
                    "inputgenerator." + "InputGenerator" + config_data.INPUT_GENERATOR_NAME,
                    fromlist=[None])
                func_inputgenerator = getattr(method_inputgenerator, "gen_input")
                func_inputgenerator(root_data)

                # exit program
                sys.exit()

            # Standard framework procedure
            # TODO: rework this

            # grab cluster data object and meta data object
            data_cluster = NPZParser.parse(dir_npz + config_data.NN_INPUT)
            meta_data = MetaData.MetaData(dir_npz + config_data.META_DATA)
            # fill additional meta data attributes
            meta_data.root_file_name = config_data.ROOT_FILE_NAME

            # evaluate model expression
            model_method = __import__("models." + "Model" + config_data.MODEL_NAME, fromlist=[None])
            func1 = getattr(model_method, "return_model")
            model = func1(data_cluster.num_features())

            neuralnetwork = SiFiCCNNTF.SiFiCCNNTF(model=model,
                                                  model_name=config_data.MODEL_NAME,
                                                  model_tag=config_data.RUN_TAG)

            # evaluate training strategy expression
            trainingstrategy_method = __import__(
                "trainingstrategy." + "TrainingStrategy" + config_data.TRAINING_STRATEGY_NAME,
                fromlist=[None])
            func2 = getattr(trainingstrategy_method, "train_strategy")
            func2(neuralnetwork, data_cluster, config_data.LOAD_MODEL)

            # evaluate analysis expression
            # create subdirectory for analysis results
            dir_results_t1 = dir_main + "/results/" + neuralnetwork.model_name + neuralnetwork.model_tag
            dir_results_t2 = dir_results_t1 + "/" + meta_data.root_file_name
            if not os.path.isdir(dir_results_t1):
                os.mkdir(dir_results_t1)
            if not os.path.isdir(dir_results_t2):
                os.mkdir(dir_results_t2)
            os.chdir(dir_results_t2)
            # analysis expressions can be a list of analysis methods
            for i in range(len(config_data.ANALYSIS_LIST)):
                analysis_method = __import__("analysis." + "Analysis" + config_data.ANALYSIS_LIST[i], fromlist=[None])
                analysis = getattr(analysis_method, "analysis")
                analysis(neuralnetwork, data_cluster, meta_data)
            os.chdir(dir_main)

        else:
            print("Error found in configfile!")
            print("CODE WILL BE ABORTED!")
            sys.exit()


    else:
        print("No configfile found!")
        print("CODE WILL BE ABORTED!")
        sys.exit()


if __name__ == "__main__":
    main()
